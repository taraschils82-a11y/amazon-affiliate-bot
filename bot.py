import os
import logging
from typing import Dict, List

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import asyncio
import json
import signal
import requests

# ---------- Config & Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("personal-gpt-bot")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "15"))
PROMPT_FILE = os.getenv("PROMPT_FILE", "system_prompt.txt")

def load_system_prompt() -> str:
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning("%s not found. Using fallback prompt.", PROMPT_FILE)
        return (
            "You are a friendly, helpful assistant who speaks casually, with memes and light humor when appropriate. "
            "Be concise unless asked for depth. Avoid purple prose."
        )

SYSTEM_PROMPT = load_system_prompt()
history: Dict[int, List[Dict[str, str]]] = {}

# ---------- OpenAI ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def call_openai(messages: List[Dict[str, str]], model: str = MODEL, temperature: float = 0.6) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(OPENAI_URL, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

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
    await update.message.reply_text("Yo! I'm your personal GPT bruh 🤖\nSend me a message to chat.")

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

# ---------- Message handler ----------
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_id = update.effective_user.id
    text = update.message.text.strip()

    asyncio.create_task(send_typing(update))
    msgs = build_messages(user_id, text)

    try:
        reply = await asyncio.to_thread(call_openai, msgs)
    except Exception as e:
        logger.exception("OpenAI error: %s", e)
        reply = "Brain server hiccuped. Try again in a sec."

    history.setdefault(user_id, []).extend(
        [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    )
    await update.message.reply_text(reply, disable_web_page_preview=True)

# ---------- Graceful shutdown ----------
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
            pass

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
