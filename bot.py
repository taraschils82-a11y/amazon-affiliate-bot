import os
import logging
from typing import Dict, List

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import asyncio
import signal

# --- Gemini SDK ---
import google.generativeai as genai

# ---------- Config & Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("personal-gpt-bot")

MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")   # or gemini-1.5-pro
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "15"))
PROMPT_FILE = os.getenv("PROMPT_FILE", "system_prompt.txt")

def load_system_prompt() -> str:
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning("%s not found. Using fallback prompt.", PROMPT_FILE)
        return ("You are a friendly, helpful assistant who speaks casually, "
                "with light humor when appropriate. Be concise unless asked for depth.")

SYSTEM_PROMPT = load_system_prompt()
history: Dict[int, List[Dict[str, str]]] = {}

# ---------- Gemini init ----------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")
genai.configure(api_key=GOOGLE_API_KEY)
_gem = genai.GenerativeModel(MODEL)

def llm_reply(messages: List[Dict[str, str]], temperature: float = 0.6) -> str:
    """Generate a reply using Gemini. Keeps function name stable for the app."""
    # Flatten chat to a single prompt (simple + robust)
    prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    try:
        resp = _gem.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        text = getattr(resp, "text", "") or ""
        return text.strip() if text else "Empty response."
    except Exception as e:
        logger.exception("Gemini API error: %s", e)
        return "Brain server hiccuped. Try again in a sec."

# ---------- Utils ----------
async def send_typing(update: Update):
    try:
        await update.message.chat.send_action(action=ChatAction.TYPING)
    except Exception:
        pass

def build_messages(user_id: int, user_text: str) -> List[Dict[str, str]]:
    convo = history.get(user_id, [])[-MAX_HISTORY:]
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(convo)
    messages.append({"role": "user", "content": user_text})
    return messages

# ---------- Commands ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Yo! I'm your personal GPT (Gemini) 🤖\nSend me a message to chat.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history.pop(user_id, None)
    await update.message.reply_text("Memory cleared for this chat. 🧹")

async def setstyle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SYSTEM_PROMPT
    new_prompt = " ".join(context.args).strip()
    if not new_prompt:
        await update.message.reply_text("Usage: /setstyle <new system prompt>")
        return
    SYSTEM_PROMPT = new_prompt
    await update.message.reply_text("Updated vibe/style ✅ (in-memory only)")

async def reloadprompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = load_system_prompt()
    await update.message.reply_text("Reloaded system_prompt.txt ✅")

async def version(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ver = os.getenv("BOT_VERSION", "dev")
    await update.message.reply_text(f"Version: {ver}\nModel: {MODEL}\nLog: {LOG_LEVEL}")

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("OK")

# ---------- Message Handler ----------
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_id = update.effective_user.id
    text = update.message.text.strip()

    asyncio.create_task(send_typing(update))
    msgs = build_messages(user_id, text)

    reply = await asyncio.to_thread(llm_reply, msgs)
    history.setdefault(user_id, []).extend(
        [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    )
    await update.message.reply_text(reply, disable_web_page_preview=True)

# ---------- Graceful Shutdown ----------
shutdown_requested = asyncio.Event()

async def _graceful_shutdown(app: Application):
    logger.info("Shutting down gracefully…")
    await app.stop()
    await app.shutdown()
    shutdown_requested.set()

# ---------- Main ----------
async def run_bot():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var missing")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("setstyle", setstyle))
    app.add_handler(CommandHandler("reloadprompt", reloadprompt))
    app.add_handler(CommandHandler("version", version))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(_graceful_shutdown(app)))
        except NotImplementedError:
            pass  # Windows

    logger.info("Bot is running… (log %s)", LOG_LEVEL)
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await shutdown_requested.wait()
    await app.updater.stop()
    await app.stop()
    await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass
