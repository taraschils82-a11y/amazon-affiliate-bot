# Deal Alert AI - Amazon India Price Tracker (Async, Category Alerts, Price History, Proxies)
# Requirements:
#   pip install python-telegram-bot==20.* httpx beautifulsoup4 lxml tenacity matplotlib
# Optional (if using system CA issues behind proxies): certifi
# ---------------------------------------------------------------
# Features added over MVP:
#  - /trackkw <keyword> [target_price]  → Category/keyword alerts via Amazon search scraping
#  - /listkw, /untrackkw                 → Manage keyword alerts
#  - Price history table + /chart <ASIN> [days]  → Plot and send price chart; “lowest in 90 days” notifier
#  - Robust scraping with retries, jitter, rotating User-Agents
#  - Optional rotating proxies (env PROXIES or a file)
#  - Exponential backoff
#  - Affiliate disclosure footer in all outbound promos
#  - Dedupe + spam prevention

import asyncio
import logging
import os
import random
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import urllib.parse

import httpx
from bs4 import BeautifulSoup
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, AIORateLimiter
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- CONFIG -----------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
AFFILIATE_TAG = os.getenv("AFFILIATE_TAG", "aigyanfree-21")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", str(6 * 60 * 60)))  # seconds, default 6h

# Affiliate disclosure appended to messages (set empty string to disable)
DISCLOSURE_TEXT = os.getenv(
    "DISCLOSURE_TEXT",
    "\n\n<i>Disclosure: This post may contain affiliate links. If you buy through them, we may earn a commission at no extra cost to you.</i>"
)

DB_PATH = os.getenv("DB_PATH", "tracklist.db")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
]

HEADERS_BASE = {
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Proxies can be provided as comma-separated list in env PROXIES or via PROXY_FILE (one per line)
PROXIES: List[str] = []
if os.getenv("PROXIES"):
    PROXIES = [p.strip() for p in os.getenv("PROXIES").split(",") if p.strip()]
elif os.getenv("PROXY_FILE") and os.path.exists(os.getenv("PROXY_FILE")):
    with open(os.getenv("PROXY_FILE"), "r", encoding="utf-8") as f:
        PROXIES = [line.strip() for line in f if line.strip()]

# --- DATABASE ---------------------------------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute(
    """
CREATE TABLE IF NOT EXISTS tracked (
    user_id INTEGER NOT NULL,
    asin TEXT NOT NULL,
    title TEXT NOT NULL,
    last_price INTEGER NOT NULL,
    image TEXT,
    target_price INTEGER,
    last_notified_price INTEGER,
    PRIMARY KEY (user_id, asin)
)
"""
)

c.execute(
    """
CREATE TABLE IF NOT EXISTS price_history (
    asin TEXT NOT NULL,
    ts INTEGER NOT NULL,
    price INTEGER NOT NULL
)
"""
)

c.execute("CREATE INDEX IF NOT EXISTS idx_price_history_asin_ts ON price_history(asin, ts)")

c.execute(
    """
CREATE TABLE IF NOT EXISTS tracked_kw (
    user_id INTEGER NOT NULL,
    keyword TEXT NOT NULL,
    target_price INTEGER,
    PRIMARY KEY (user_id, keyword)
)
"""
)

c.execute(
    """
CREATE TABLE IF NOT EXISTS kw_seen (
    user_id INTEGER NOT NULL,
    keyword TEXT NOT NULL,
    asin TEXT NOT NULL,
    last_notified_price INTEGER,
    PRIMARY KEY (user_id, keyword, asin)
)
"""
)

conn.commit()

# --- HELPERS ----------------------------------------------------

def build_affiliate_link(asin: str) -> str:
    return f"https://www.amazon.in/dp/{asin}/?tag={AFFILIATE_TAG}"


def extract_asin(url: str) -> Optional[str]:
    m = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", url)
    return m.group(1) if m else None


def parse_price_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    try:
        return int(digits) if digits else None
    except Exception:
        return None


def fmt_price(p: Optional[int]) -> str:
    return f"₹{p:,}".replace(",", ",") if p is not None else "—"


def choose_proxy() -> Optional[str]:
    if not PROXIES:
        return None
    return random.choice(PROXIES)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError))
)
async def fetch_html(url: str) -> str:
    headers = dict(HEADERS_BASE)
    headers["User-Agent"] = random.choice(USER_AGENTS)
    # jitter to reduce burstiness
    await asyncio.sleep(random.uniform(0.2, 0.8))
    proxy = choose_proxy()
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, proxies=proxy) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.text


def extract_image(soup: BeautifulSoup) -> Optional[str]:
    sel = [
        "#landingImage",
        "img#imgTagWrapperId img",
        "div#imgTagWrapperId img",
        "img[data-old-hires]",
    ]
    for s in sel:
        el = soup.select_one(s)
        if el and el.get("src"):
            return el["src"]
        if el and el.get("data-old-hires"):
            return el["data-old-hires"]
    el = soup.select_one("img[src*='images/I/']")
    return el["src"] if el and el.get("src") else None


def extract_title(soup: BeautifulSoup) -> Optional[str]:
    el = soup.select_one("#productTitle")
    if el:
        return el.get_text(strip=True)
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else None


def extract_price(soup: BeautifulSoup) -> Optional[int]:
    candidates = [
        ".a-price .a-offscreen",
        "#priceblock_ourprice",
        "#priceblock_dealprice",
        "#priceblock_saleprice",
        ".a-price-whole",
        ".a-color-price",
    ]
    for css in candidates:
        el = soup.select_one(css)
        if el:
            p = parse_price_int(el.get_text(" ", strip=True))
            if p:
                return p
    text = soup.get_text(" ", strip=True)
    m = re.search(r"₹\s?([\d,]+)", text)
    if m:
        return parse_price_int(m.group(0))
    return None


async def scrape_product(asin: str) -> Optional[Dict]:
    url = f"https://www.amazon.in/dp/{asin}"
    try:
        html = await fetch_html(url)
    except Exception as e:
        logging.warning(f"Fetch failed for {asin}: {e}")
        return None
    soup = BeautifulSoup(html, "lxml")

    title = extract_title(soup)
    price = extract_price(soup)
    image = extract_image(soup)

    if title and price:
        return {"title": title, "price": price, "image": image}
    return None


async def search_keyword(keyword: str, max_items: int = 20) -> List[Dict]:
    """Scrape Amazon.in search results for a keyword and return products with price.
    Note: HTML structure can change; adjust selectors if needed.
    """
    q = urllib.parse.quote_plus(keyword)
    url = f"https://www.amazon.in/s?k={q}"
    try:
        html = await fetch_html(url)
    except Exception as e:
        logging.warning(f"Search fetch failed for '{keyword}': {e}")
        return []
    soup = BeautifulSoup(html, "lxml")
    items = []
    for card in soup.select("div.s-result-item[data-asin]"):
        asin = card.get("data-asin")
        if not asin or len(asin) != 10:
            continue
        title_el = card.select_one("h2 a span") or card.select_one(".a-size-medium.a-color-base.a-text-normal")
        price_el = card.select_one(".a-price .a-offscreen") or card.select_one(".a-price-whole")
        img_el = card.select_one("img.s-image")
        if not (title_el and price_el):
            continue
        title = title_el.get_text(strip=True)
        price = parse_price_int(price_el.get_text(" ", strip=True))
        if not price:
            continue
        image = img_el["src"] if img_el and img_el.get("src") else None
        items.append({"asin": asin, "title": title, "price": price, "image": image})
        if len(items) >= max_items:
            break
    return items


# --- HISTORY / ANALYTICS ---------------------------------------

def record_history(asin: str, price: int):
    with closing(conn.cursor()) as cur:
        cur.execute("INSERT INTO price_history (asin, ts, price) VALUES (?, ?, ?)", (asin, int(time.time()), price))
        conn.commit()


def lowest_in_days(asin: str, days: int = 90) -> Optional[Tuple[int, int]]:
    """Return (min_price, since_ts) within last 'days' days, or None if no data."""
    since = int(time.time()) - days * 86400
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT MIN(price), MIN(ts) FROM price_history WHERE asin = ? AND ts >= ?", (asin, since))
        row = cur.fetchone()
    if row and row[0] is not None:
        return int(row[0]), int(row[1])
    return None


# --- MESSAGING --------------------------------------------------

def append_disclosure(msg: str) -> str:
    return msg + (DISCLOSURE_TEXT if DISCLOSURE_TEXT else "")


# --- COMMANDS ---------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(
        "👋 <b>Welcome to Deal Alert AI</b>\n\n"
        "<b>Commands</b>:\n"
        "• <code>/track &lt;amazon.in link&gt; [target_price]</code> – Track a product.\n"
        "• <code>/list</code> – Your tracked products.\n"
        "• <code>/untrack &lt;ASIN&gt;</code> – Stop tracking a product.\n"
        "• <code>/trackkw &lt;keyword&gt; [target_price]</code> – Keyword alerts (e.g., iPhone, SSD 1TB).\n"
        "• <code>/listkw</code> – Your keyword alerts.\n"
        "• <code>/untrackkw &lt;keyword&gt;</code> – Remove a keyword alert.\n"
        "• <code>/chart &lt;ASIN&gt; [days]</code> – Price history chart (default 90d).\n\n"
        f"Checks run every {int(CHECK_INTERVAL/3600)} hours."
    )


async def track(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_html("⚠️ Send: <code>/track &lt;amazon.in product link&gt; [target_price]</code>")
        return

    url = context.args[0]
    asin = extract_asin(url)
    if not asin:
        await update.message.reply_html("❌ Couldn’t extract ASIN. Please send a proper Amazon.in product link.")
        return

    target_price = None
    if len(context.args) > 1:
        tp = parse_price_int(context.args[1])
        if tp and tp > 0:
            target_price = tp

    product = await scrape_product(asin)
    if not product:
        await update.message.reply_html("⚠️ Couldn’t fetch product info. Try again later or another link.")
        return

    with closing(conn.cursor()) as cur:
        try:
            cur.execute(
                """
                INSERT INTO tracked (user_id, asin, title, last_price, image, target_price, last_notified_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, asin) DO UPDATE SET
                    title=excluded.title,
                    last_price=excluded.last_price,
                    image=excluded.image,
                    target_price=COALESCE(excluded.target_price, tracked.target_price)
                """,
                (user_id, asin, product["title"], product["price"], product["image"], target_price, None)
            )
            conn.commit()
        except Exception as e:
            logging.exception(e)
            await update.message.reply_html("❌ DB error. Please try again.")
            return

    # record history
    if product.get("price"):
        record_history(asin, product["price"])

    msg = (
        f"✅ <b>Tracking:</b> {product['title']}\n"
        f"💰 <b>Current:</b> {fmt_price(product['price'])}\n"
        f"🎯 <b>Target:</b> {'—' if not target_price else fmt_price(target_price)}\n"
        f"🔗 <a href=\"{build_affiliate_link(asin)}\">View on Amazon</a>"
    )
    msg = append_disclosure(msg)

    if product.get("image"):
        await update.message.reply_photo(photo=product["image"], caption=msg, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_html(msg)


async def list_tracked(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT asin, title, last_price, target_price FROM tracked WHERE user_id = ?", (user_id,))
        rows = cur.fetchall()

    if not rows:
        await update.message.reply_html("📭 You aren’t tracking any items yet.")
        return

    lines = ["📋 <b>Your Tracked Products</b>\n"]
    for asin, title, price, target in rows:
        tgt = f"🎯 {fmt_price(target)}" if target else "—"
        lines.append(
            f"• <b>{title[:80]}</b>\n"
            f"  ASIN: <code>{asin}</code> · 💰 {fmt_price(price)} · {tgt} · "
            f"<a href=\"{build_affiliate_link(asin)}\">Link</a>"
        )
    await update.message.reply_html("\n".join(lines))


async def untrack(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_html("Usage: <code>/untrack &lt;ASIN&gt;</code>")
        return
    asin = context.args[0].upper()
    with closing(conn.cursor()) as cur:
        cur.execute("DELETE FROM tracked WHERE user_id = ? AND asin = ?", (user_id, asin))
        conn.commit()
    await update.message.reply_html(f"🗑️ Stopped tracking <code>{asin}</code>.")


async def clear_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    with closing(conn.cursor()) as cur:
        cur.execute("DELETE FROM tracked WHERE user_id = ?", (user_id,))
        conn.commit()
    await update.message.reply_html("🧹 Cleared all tracked items.")


# --- KEYWORD ALERTS --------------------------------------------
async def trackkw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_html("⚠️ Send: <code>/trackkw &lt;keyword phrase&gt; [target_price]</code>")
        return
    keyword = " ".join([a for a in context.args if not a.isdigit() and not re.match(r"^₹?\d+$", a)])
    if not keyword:
        keyword = context.args[0]
    target_price = None
    for a in context.args[1:]:
        tp = parse_price_int(a)
        if tp:
            target_price = tp
            break

    with closing(conn.cursor()) as cur:
        cur.execute(
            """
            INSERT INTO tracked_kw (user_id, keyword, target_price)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, keyword) DO UPDATE SET target_price=excluded.target_price
            """,
            (user_id, keyword, target_price)
        )
        conn.commit()
    await update.message.reply_html(
        f"🔎 Keyword alert set for <b>{keyword}</b> · Target: {fmt_price(target_price)}"
    )


async def listkw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT keyword, target_price FROM tracked_kw WHERE user_id = ?", (user_id,))
        rows = cur.fetchall()
    if not rows:
        await update.message.reply_html("📭 No keyword alerts yet. Try <code>/trackkw iPhone 50000</code>.")
        return
    lines = ["🗂️ <b>Your Keyword Alerts</b>\n"]
    for kw, tp in rows:
        lines.append(f"• <b>{kw}</b> · 🎯 {fmt_price(tp)}")
    await update.message.reply_html("\n".join(lines))


async def untrackkw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_html("Usage: <code>/untrackkw &lt;keyword&gt;</code>")
        return
    keyword = " ".join(context.args)
    with closing(conn.cursor()) as cur:
        cur.execute("DELETE FROM tracked_kw WHERE user_id = ? AND keyword = ?", (user_id, keyword))
        cur.execute("DELETE FROM kw_seen WHERE user_id = ? AND keyword = ?", (user_id, keyword))
        conn.commit()
    await update.message.reply_html(f"🗑️ Removed keyword alert for <b>{keyword}</b>.")


# --- CHART ------------------------------------------------------
async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        await update.message.reply_html("📉 Charting dependencies missing. Install matplotlib.")
        return

    if not context.args:
        await update.message.reply_html("Usage: <code>/chart &lt;ASIN&gt; [days]</code>")
        return
    asin = context.args[0].upper()
    days = 90
    if len(context.args) > 1:
        try:
            days = max(7, int(context.args[1]))
        except Exception:
            pass
    since = int(time.time()) - days * 86400
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT ts, price FROM price_history WHERE asin = ? AND ts >= ? ORDER BY ts ASC", (asin, since))
        rows = cur.fetchall()
    if not rows:
        await update.message.reply_html("No history yet for that ASIN. Please wait for scheduled checks.")
        return

    xs = [datetime.fromtimestamp(ts) for ts, _ in rows]
    ys = [p for _, p in rows]

    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.set_title(f"Price History – {asin} ({days}d)")
    fig.autofmt_xdate()
    out = f"/tmp/{asin}_chart.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    await update.message.reply_photo(photo=open(out, "rb"), caption=f"Price history for {asin} ({days}d)")


# --- SCHEDULERS -------------------------------------------------
async def check_prices(context: ContextTypes.DEFAULT_TYPE):
    # Iterate all tracked ASINs, update price, record history, and notify on drops/targets/lowest-in-90d
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT user_id, asin, title, last_price, image, target_price, last_notified_price FROM tracked")
        rows = cur.fetchall()

    for user_id, asin, title, last_price, image, target_price, last_notified in rows:
        product = await scrape_product(asin)
        if not product or not product.get("price"):
            continue
        current = int(product["price"])
        record_history(asin, current)

        # Determine lowest in 90 days
        low90 = lowest_in_days(asin, 90)
        is_low90 = False
        lowest_val = None
        if low90:
            lowest_val = low90[0]
            is_low90 = current <= lowest_val

        notify = False
        reasons = []
        if target_price and current <= target_price:
            notify = True
            reasons.append(f"🎯 Reached target ({fmt_price(target_price)})")
        if current < (last_price or 10**9):
            notify = True
            reasons.append(f"⬇️ Dropped: {fmt_price(last_price)} → {fmt_price(current)}")
        if is_low90:
            notify = True
            reasons.append("📉 Lowest in 90 days")

        # Avoid repeat spam at same price
        if notify and last_notified is not None and current == last_notified:
            notify = False

        # Update DB with latest price
        with closing(conn.cursor()) as cur:
            cur.execute(
                """
                UPDATE tracked
                SET title = ?, last_price = ?, image = ?, last_notified_price = CASE WHEN ? THEN ? ELSE last_notified_price END
                WHERE user_id = ? AND asin = ?
                """,
                (product["title"], current, product["image"], int(bool(notify)), current if notify else None, user_id, asin)
            )
            conn.commit()

        if notify:
            msg = (
                f"{' · '.join(reasons)}\n\n"
                f"<b>{product['title']}</b>\n"
                f"💰 <b>Now:</b> {fmt_price(current)}\n"
                f"🔗 <a href=\"{build_affiliate_link(asin)}\">View on Amazon</a>"
            )
            msg = append_disclosure(msg)
            try:
                if product.get("image"):
                    await context.bot.send_photo(chat_id=user_id, photo=product["image"], caption=msg, parse_mode=ParseMode.HTML)
                else:
                    await context.bot.send_message(chat_id=user_id, text=msg, parse_mode=ParseMode.HTML)
            except Exception as e:
                logging.warning(f"Notify failed for {user_id}/{asin}: {e}")


async def check_keywords(context: ContextTypes.DEFAULT_TYPE):
    # For each user+keyword, search and notify items that meet target or are compelling
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT user_id, keyword, target_price FROM tracked_kw")
        alerts = cur.fetchall()

    for user_id, keyword, target_price in alerts:
        items = await search_keyword(keyword, max_items=20)
        if not items:
            continue
        for item in items:
            asin = item["asin"]
            price = item["price"]
            # If a target is set, only notify when price <= target
            if target_price and price > target_price:
                continue
            # Check spam prevention: if we already notified this asin for this kw at same price, skip
            with closing(conn.cursor()) as cur:
                cur.execute("SELECT last_notified_price FROM kw_seen WHERE user_id = ? AND keyword = ? AND asin = ?", (user_id, keyword, asin))
                row = cur.fetchone()
            if row and row[0] == price:
                continue

            # Insert/update kw_seen
            with closing(conn.cursor()) as cur:
                cur.execute(
                    """
                    INSERT INTO kw_seen (user_id, keyword, asin, last_notified_price)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, keyword, asin) DO UPDATE SET last_notified_price=excluded.last_notified_price
                    """,
                    (user_id, keyword, asin, price)
                )
                conn.commit()

            caption = (
                f"🔎 <b>{keyword}</b>\n"
                f"<b>{item['title']}</b>\n"
                f"💰 <b>Price:</b> {fmt_price(price)}\n"
                f"🔗 <a href=\"{build_affiliate_link(asin)}\">View on Amazon</a>"
            )
            caption = append_disclosure(caption)
            try:
                if item.get("image"):
                    await context.bot.send_photo(chat_id=user_id, photo=item["image"], caption=caption, parse_mode=ParseMode.HTML)
                else:
                    await context.bot.send_message(chat_id=user_id, text=caption, parse_mode=ParseMode.HTML)
            except Exception as e:
                logging.warning(f"KW notify failed for {user_id}/{keyword}/{asin}: {e}")


# --- MAIN -------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("track", track))
    app.add_handler(CommandHandler("list", list_tracked))
    app.add_handler(CommandHandler("untrack", untrack))
    app.add_handler(CommandHandler("clear", clear_all))

    app.add_handler(CommandHandler("trackkw", trackkw))
    app.add_handler(CommandHandler("listkw", listkw))
    app.add_handler(CommandHandler("untrackkw", untrackkw))

    app.add_handler(CommandHandler("chart", chart))

    # schedule periodic checkers
    app.job_queue.run_repeating(check_prices, interval=timedelta(seconds=CHECK_INTERVAL), first=10)
    # keyword checks can be a bit staggered
    app.job_queue.run_repeating(check_keywords, interval=timedelta(seconds=CHECK_INTERVAL), first=60)

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
