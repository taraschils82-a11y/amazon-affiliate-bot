#!/usr/bin/env python3
"""
Deal Alert AI - Amazon.in Affiliate Bot
A Telegram bot for tracking Amazon products and keyword alerts with affiliate marketing.
"""

import asyncio
import logging
import os
import re
import sqlite3
import time
import random
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import html

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, wait_exponential, stop_after_attempt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes, JobQueue, MessageHandler, filters
)
from telegram.constants import ParseMode

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mdates = None
    FuncFormatter = None
    MATPLOTLIB_AVAILABLE = False

# Optional Gemini AI import
try:
    import google.genai as genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
AFFILIATE_TAG = os.getenv("AFFILIATE_TAG", "aigyanfree-21")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "21600"))  # 6 hours default
DB_PATH = os.getenv("DB_PATH", "tracklist.db")
DISCLOSURE_TEXT = os.getenv("DISCLOSURE_TEXT", "")
MAIN_CHANNEL_ID = os.getenv("MAIN_CHANNEL_ID", "")
PROXIES_ENV = os.getenv("PROXIES", "")
PROXY_FILE = os.getenv("PROXY_FILE", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize Gemini client if available
gemini_client = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# Price selectors to try in order
PRICE_SELECTORS = [
    ".a-price .a-offscreen",
    "#priceblock_ourprice",
    "#priceblock_dealprice", 
    "#priceblock_saleprice",
    ".a-price-whole",
    ".a-color-price"
]

# Image selectors
IMAGE_SELECTORS = [
    "#landingImage",
    "img#imgTagWrapperId img",
    "[data-old-hires]"
]

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tracked products table
            cursor.execute('''
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
            ''')
            
            # Price history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    asin TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    price INTEGER NOT NULL
                )
            ''')
            
            # Create index for price history
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_price_history 
                ON price_history(asin, ts)
            ''')
            
            # Tracked keywords table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracked_kw (
                    user_id INTEGER NOT NULL,
                    keyword TEXT NOT NULL,
                    target_price INTEGER,
                    PRIMARY KEY (user_id, keyword)
                )
            ''')
            
            # Keywords seen table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kw_seen (
                    user_id INTEGER NOT NULL,
                    keyword TEXT NOT NULL,
                    asin TEXT NOT NULL,
                    last_notified_price INTEGER,
                    PRIMARY KEY (user_id, keyword, asin)
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")

    def get_tracked_products(self, user_id: int) -> List[Dict]:
        """Get all tracked products for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT asin, title, last_price, image, target_price, last_notified_price
                FROM tracked WHERE user_id = ?
            ''', (user_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'asin': row[0],
                    'title': row[1],
                    'last_price': row[2],
                    'image': row[3],
                    'target_price': row[4],
                    'last_notified_price': row[5]
                })
            return results

    def add_tracked_product(self, user_id: int, asin: str, title: str, 
                           price: int, image: Optional[str] = None, target_price: Optional[int] = None):
        """Add or update a tracked product"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO tracked 
                (user_id, asin, title, last_price, image, target_price, last_notified_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, asin, title, price, image, target_price, None))
            conn.commit()

    def remove_tracked_product(self, user_id: int, asin: str) -> bool:
        """Remove a tracked product"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM tracked WHERE user_id = ? AND asin = ?
            ''', (user_id, asin))
            conn.commit()
            return cursor.rowcount > 0

    def clear_tracked_products(self, user_id: int) -> int:
        """Clear all tracked products for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM tracked WHERE user_id = ?', (user_id,))
            conn.commit()
            return cursor.rowcount

    def add_price_history(self, asin: str, price: int):
        """Add price history entry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO price_history (asin, ts, price)
                VALUES (?, ?, ?)
            ''', (asin, int(time.time()), price))
            conn.commit()

    def get_price_history(self, asin: str, days: int = 90) -> List[Tuple[int, int]]:
        """Get price history for an ASIN"""
        cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ts, price FROM price_history 
                WHERE asin = ? AND ts >= ?
                ORDER BY ts
            ''', (asin, cutoff_time))
            return cursor.fetchall()

    def get_lowest_price_days(self, asin: str, days: int = 90) -> Optional[int]:
        """Get lowest price in the last N days"""
        cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT MIN(price) FROM price_history 
                WHERE asin = ? AND ts >= ?
            ''', (asin, cutoff_time))
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None

    def update_tracked_product(self, user_id: int, asin: str, title: Optional[str] = None,
                              price: Optional[int] = None, image: Optional[str] = None, 
                              last_notified_price: Optional[int] = None):
        """Update tracked product fields"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            if price is not None:
                updates.append("last_price = ?")
                params.append(price)
            if image is not None:
                updates.append("image = ?")
                params.append(image)
            if last_notified_price is not None:
                updates.append("last_notified_price = ?")
                params.append(last_notified_price)
            
            if updates:
                params.extend([user_id, asin])
                cursor.execute(f'''
                    UPDATE tracked SET {", ".join(updates)}
                    WHERE user_id = ? AND asin = ?
                ''', params)
                conn.commit()

    def get_all_tracked_products(self) -> List[Dict]:
        """Get all tracked products across all users"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, asin, title, last_price, image, target_price, last_notified_price
                FROM tracked
            ''')
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'user_id': row[0],
                    'asin': row[1],
                    'title': row[2],
                    'last_price': row[3],
                    'image': row[4],
                    'target_price': row[5],
                    'last_notified_price': row[6]
                })
            return results

    # Keyword tracking methods
    def add_keyword_alert(self, user_id: int, keyword: str, target_price: Optional[int] = None):
        """Add keyword alert"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO tracked_kw (user_id, keyword, target_price)
                VALUES (?, ?, ?)
            ''', (user_id, keyword, target_price))
            conn.commit()

    def get_keyword_alerts(self, user_id: int) -> List[Dict]:
        """Get keyword alerts for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT keyword, target_price FROM tracked_kw WHERE user_id = ?
            ''', (user_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'keyword': row[0],
                    'target_price': row[1]
                })
            return results

    def remove_keyword_alert(self, user_id: int, keyword: str) -> bool:
        """Remove keyword alert"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM tracked_kw WHERE user_id = ? AND keyword = ?
            ''', (user_id, keyword))
            conn.commit()
            return cursor.rowcount > 0

    def get_all_keyword_alerts(self) -> List[Dict]:
        """Get all keyword alerts across all users"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, keyword, target_price FROM tracked_kw')
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'user_id': row[0],
                    'keyword': row[1],
                    'target_price': row[2]
                })
            return results

    def check_kw_seen(self, user_id: int, keyword: str, asin: str, price: int) -> bool:
        """Check if keyword/ASIN combination was already notified at this price"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT last_notified_price FROM kw_seen 
                WHERE user_id = ? AND keyword = ? AND asin = ?
            ''', (user_id, keyword, asin))
            result = cursor.fetchone()
            
            if result is None:
                return False
            
            return result[0] == price

    def mark_kw_seen(self, user_id: int, keyword: str, asin: str, price: int):
        """Mark keyword/ASIN as seen at this price"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO kw_seen (user_id, keyword, asin, last_notified_price)
                VALUES (?, ?, ?, ?)
            ''', (user_id, keyword, asin, price))
            conn.commit()

class AmazonScraper:
    """Handles Amazon product scraping"""
    
    def __init__(self):
        self.proxies = self._load_proxies()
        self.session = None
    
    def _load_proxies(self) -> List[str]:
        """Load proxies from environment or file"""
        proxies = []
        
        # Load from PROXIES env var
        if PROXIES_ENV:
            proxies.extend([p.strip() for p in PROXIES_ENV.split(',') if p.strip()])
        
        # Load from proxy file
        if PROXY_FILE and os.path.exists(PROXY_FILE):
            try:
                with open(PROXY_FILE, 'r') as f:
                    file_proxies = [line.strip() for line in f if line.strip()]
                    proxies.extend(file_proxies)
            except Exception as e:
                logger.warning(f"Failed to load proxy file: {e}")
        
        return proxies

    def _get_random_headers(self) -> Dict[str, str]:
        """Get randomized headers"""
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session"""
        if self.session is None or self.session.is_closed:
            headers = self._get_random_headers()
            timeout = httpx.Timeout(20.0)
            
            proxy = None
            if self.proxies:
                proxy = random.choice(self.proxies)
                logger.info(f"Using proxy: {proxy}")
            
            self.session = httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                proxies=proxy,
                follow_redirects=True
            )
        
        return self.session

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML with retry logic"""
        try:
            session = await self._get_session()
            response = await session.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            raise

    async def extract_asin(self, amazon_url: str) -> Optional[str]:
        """Extract ASIN from Amazon URL, handling short URLs and redirects"""
        original_url = amazon_url
        
        # Handle redirect URLs (DealsCrown, short URLs, etc.)
        if any(domain in amazon_url for domain in ['amzn.to', 'amzn.in', 'a.co', 'dealscrown.com', 'bit.ly', 'tinyurl.com']):
            try:
                session = await self._get_session()
                # Follow redirects to get the final URL
                response = await session.get(amazon_url, follow_redirects=True)
                amazon_url = str(response.url)
                logger.info(f"Resolved redirect URL {original_url} to: {amazon_url}")
            except Exception as e:
                logger.warning(f"Failed to resolve redirect URL {original_url}: {e}")
        
        # First try to extract ASIN from URL parameters (like hidden-keywords)
        import urllib.parse
        parsed_url = urllib.parse.urlparse(amazon_url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        # Check for ASINs in hidden-keywords parameter
        if 'hidden-keywords' in query_params:
            keywords = query_params['hidden-keywords'][0]
            # Decode URL encoding and split by | or %7C
            keywords = urllib.parse.unquote(keywords)
            asins = re.findall(r'([A-Z0-9]{10})', keywords)
            if asins:
                logger.info(f"Found ASINs in hidden-keywords: {asins}")
                return asins[0]  # Return first valid ASIN
        
        # Check for ASINs in other query parameters
        for param_name, param_values in query_params.items():
            for value in param_values:
                asins = re.findall(r'([A-Z0-9]{10})', value)
                if asins:
                    logger.info(f"Found ASIN in {param_name} parameter: {asins[0]}")
                    return asins[0]
        
        # Try standard Amazon URL patterns
        patterns = [
            r'/dp/([A-Z0-9]{10})',
            r'/gp/product/([A-Z0-9]{10})',
            r'asin=([A-Z0-9]{10})',
            r'/([A-Z0-9]{10})/?(?:\?|$)',
            r'product/([A-Z0-9]{10})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, amazon_url)
            if match:
                logger.info(f"Found ASIN using pattern {pattern}: {match.group(1)}")
                return match.group(1)
        
        # Last resort: search for any ASIN pattern in the entire URL
        asin_matches = re.findall(r'([A-Z0-9]{10})', amazon_url)
        if asin_matches:
            # Filter for valid ASIN format (starts with B and has mix of letters/numbers)
            valid_asins = [asin for asin in asin_matches if asin.startswith('B') and re.match(r'^[A-Z0-9]{10}$', asin)]
            if valid_asins:
                logger.info(f"Found valid ASIN in URL: {valid_asins[0]}")
                return valid_asins[0]
        
        logger.warning(f"Could not extract ASIN from URL: {original_url}")
        return None

    def parse_price(self, html: str) -> Optional[int]:
        """Parse price from HTML using multiple selectors"""
        soup = BeautifulSoup(html, 'lxml')
        
        # Try CSS selectors first
        for selector in PRICE_SELECTORS:
            elements = soup.select(selector)
            for element in elements:
                price_text = element.get_text(strip=True)
                price = self._extract_price_from_text(price_text)
                if price:
                    return price
        
        # Fallback: search for ₹ pattern in entire page
        rupee_pattern = r'₹\s*([0-9,]+(?:\.[0-9]{2})?)'
        matches = re.findall(rupee_pattern, html)
        
        for match in matches:
            try:
                price_str = match.replace(',', '')
                price_float = float(price_str)
                return int(price_float * 100)  # Convert to paise
            except ValueError:
                continue
        
        return None

    def _extract_price_from_text(self, text: str) -> Optional[int]:
        """Extract price from text string"""
        # Remove currency symbols and clean text
        cleaned = re.sub(r'[₹$,\s]', '', text)
        
        # Try to extract number
        price_match = re.search(r'([0-9]+(?:\.[0-9]{2})?)', cleaned)
        if price_match:
            try:
                price_float = float(price_match.group(1))
                return int(price_float * 100)  # Convert to paise
            except ValueError:
                pass
        
        return None

    def parse_title(self, html: str) -> Optional[str]:
        """Parse product title from HTML"""
        soup = BeautifulSoup(html, 'lxml')
        
        title_selectors = [
            '#productTitle',
            '.product-title',
            'h1.a-size-large',
            'h1 span'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 5:
                    return title[:200]  # Truncate long titles
        
        return None

    def parse_image(self, html: str) -> Optional[str]:
        """Parse product image URL from HTML"""
        soup = BeautifulSoup(html, 'lxml')
        
        for selector in IMAGE_SELECTORS:
            element = soup.select_one(selector)
            if element:
                # Try different attributes
                for attr in ['data-old-hires', 'src', 'data-src']:
                    img_url = element.get(attr)
                    if img_url and 'images/I/' in img_url:
                        return str(img_url)
        
        # Fallback: find any Amazon product image
        img_tags = soup.find_all('img')
        for img in img_tags:
            for attr in ['src', 'data-src', 'data-old-hires']:
                img_url = img.get(attr, '')
                if 'images/I/' in img_url and 'amazon' in img_url:
                    return img_url
        
        return None

    async def scrape_product(self, asin: str) -> Optional[Dict]:
        """Scrape product data by ASIN"""
        url = f"https://www.amazon.in/dp/{asin}"
        
        try:
            html = await self.fetch_html(url)
            if not html:
                return None
            
            title = self.parse_title(html)
            price = self.parse_price(html)
            image = self.parse_image(html)
            
            if not title or not price:
                logger.warning(f"Could not extract required data for ASIN {asin}")
                return None
            
            return {
                'asin': asin,
                'title': title,
                'price': price,
                'image': image
            }
            
        except Exception as e:
            logger.error(f"Error scraping ASIN {asin}: {e}")
            return None

    async def search_products(self, keyword: str, max_results: int = 20) -> List[Dict]:
        """Search for products by keyword"""
        search_url = f"https://www.amazon.in/s?k={keyword.replace(' ', '+')}"
        
        try:
            html = await self.fetch_html(search_url)
            if not html:
                return []
            
            soup = BeautifulSoup(html, 'lxml')
            products = []
            
            # Find product containers
            product_containers = soup.select('[data-asin]:not([data-asin=""])')
            
            for container in product_containers[:max_results]:
                asin = container.get('data-asin')
                if not asin or len(asin) != 10:
                    continue
                
                # Extract title
                title_elem = container.select_one('h2 a span, .s-size-mini span')
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)
                
                # Extract price
                price_elem = container.select_one('.a-price .a-offscreen')
                if not price_elem:
                    continue
                
                price = self._extract_price_from_text(price_elem.get_text())
                if not price:
                    continue
                
                # Extract image
                img_elem = container.select_one('img')
                image = None
                if img_elem:
                    image = img_elem.get('src') or img_elem.get('data-src')
                
                products.append({
                    'asin': asin,
                    'title': title,
                    'price': price,
                    'image': image
                })
            
            return products
            
        except Exception as e:
            logger.error(f"Error searching for keyword '{keyword}': {e}")
            return []

    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.is_closed:
            await self.session.aclose()

def format_price(price_paise: int) -> str:
    """Format price from paise to rupees"""
    rupees = price_paise / 100
    return f"₹{rupees:,.2f}"

def build_affiliate_link(asin: str) -> str:
    """Build Amazon affiliate link"""
    return f"https://www.amazon.in/dp/{asin}/?tag={AFFILIATE_TAG}"

def truncate_title(title: str, max_length: int = 80) -> str:
    """Truncate title sensibly"""
    if len(title) <= max_length:
        return title
    return title[:max_length-3] + "..."

# Initialize components
db = DatabaseManager(DB_PATH)
scraper = AmazonScraper()

# Keyboard Utilities
def get_main_menu_keyboard():
    """Main menu inline keyboard"""
    keyboard = [
        [InlineKeyboardButton("📦 Track Products", callback_data="menu_track"),
         InlineKeyboardButton("🔍 Keyword Alerts", callback_data="menu_keywords")],
        [InlineKeyboardButton("📊 My Lists", callback_data="menu_lists"),
         InlineKeyboardButton("📈 Price Charts", callback_data="menu_charts")],
        [InlineKeyboardButton("🤖 AI Price Compare", callback_data="menu_ai"),
         InlineKeyboardButton("📢 Manual Post", callback_data="menu_post")],
        [InlineKeyboardButton("📺 Channel Setup", callback_data="menu_channel"),
         InlineKeyboardButton("❓ Help", callback_data="menu_help")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_back_keyboard(callback_data="main_menu"):
    """Back button keyboard"""
    keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data=callback_data)]]
    return InlineKeyboardMarkup(keyboard)

def get_track_menu_keyboard():
    """Track products menu"""
    keyboard = [
        [InlineKeyboardButton("➕ Track New Product", callback_data="track_new")],
        [InlineKeyboardButton("📋 View Tracked Products", callback_data="track_list")],
        [InlineKeyboardButton("❌ Remove Product", callback_data="track_remove")],
        [InlineKeyboardButton("🗑️ Clear All", callback_data="track_clear")],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_keywords_menu_keyboard():
    """Keywords menu"""
    keyboard = [
        [InlineKeyboardButton("➕ Add Keyword Alert", callback_data="kw_new")],
        [InlineKeyboardButton("📋 View Keyword Alerts", callback_data="kw_list")],
        [InlineKeyboardButton("❌ Remove Keyword", callback_data="kw_remove")],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_channel_menu_keyboard():
    """Channel setup menu"""
    keyboard = [
        [InlineKeyboardButton("📢 Set Channel ID", callback_data="channel_set")],
        [InlineKeyboardButton("📋 View Current Channel", callback_data="channel_view")],
        [InlineKeyboardButton("🔧 Test Channel Posting", callback_data="channel_test")],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_message = """
🤖 <b>Welcome to Deal Alert AI - Amazon Affiliate Bot!</b>

I help you track Amazon.in products, find deals, and compare prices with AI assistance!

✨ <b>Choose an option from the menu below:</b>

🚀 <b>Quick Start:</b>
• Track products with target prices
• Set keyword alerts for deal discovery  
• Get AI-powered price comparisons
• Auto-post deals to your channel

I monitor prices every 6 hours and send smart notifications!
"""
    
    await update.message.reply_text(
        welcome_message, 
        parse_mode=ParseMode.HTML, 
        reply_markup=get_main_menu_keyboard()
    )

async def track_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /track command"""
    if not context.args:
        await update.message.reply_text(
            "Please provide an Amazon URL to track.\n"
            "Usage: <code>/track [Amazon URL] [target_price]</code>",
            parse_mode=ParseMode.HTML
        )
        return
    
    amazon_url = context.args[0]
    target_price = None
    
    # Parse target price if provided
    if len(context.args) > 1:
        try:
            target_price = int(float(context.args[1]) * 100)  # Convert to paise
        except ValueError:
            await update.message.reply_text("Invalid target price. Please use a number.")
            return
    
    # Extract ASIN
    asin = await scraper.extract_asin(amazon_url)
    if not asin:
        await update.message.reply_text("Could not extract product ID from the URL. Please check the link.")
        return
    
    await update.message.reply_text("🔍 Fetching product details...")
    
    try:
        # Scrape product data
        product_data = await scraper.scrape_product(asin)
        if not product_data:
            await update.message.reply_text(
                "❌ Could not fetch product details. Please try again later or check if the URL is valid."
            )
            return
        
        # Save to database
        user_id = update.effective_user.id
        db.add_tracked_product(
            user_id=user_id,
            asin=asin,
            title=product_data['title'],
            price=product_data['price'],
            image=product_data['image'],
            target_price=target_price
        )
        
        # Add to price history
        db.add_price_history(asin, product_data['price'])
        
        # Post to channel automatically
        await post_new_product_to_channel(context, update.effective_user, product_data, asin, target_price)
        
        # Build response message
        affiliate_link = build_affiliate_link(asin)
        truncated_title = truncate_title(product_data['title'])
        current_price = format_price(product_data['price'])
        
        message = f"✅ <b>Now tracking:</b>\n\n"
        message += f"📦 <b>{html.escape(truncated_title)}</b>\n"
        message += f"💰 Current Price: <b>{current_price}</b>\n"
        
        if target_price:
            target_formatted = format_price(target_price)
            message += f"🎯 Target Price: <b>{target_formatted}</b>\n"
        
        message += f"\n🛒 <a href='{affiliate_link}'>Buy on Amazon</a>\n\n"
        message += "📢 <i>Product posted to channel!</i> I'll notify about price changes."
        
        if DISCLOSURE_TEXT:
            message += f"\n\n{DISCLOSURE_TEXT}"
        
        await update.message.reply_text(message, parse_mode=ParseMode.HTML, disable_web_page_preview=False)
        
    except Exception as e:
        logger.error(f"Error in track_command: {e}")
        await update.message.reply_text("❌ An error occurred while tracking the product. Please try again.")

async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /list command"""
    user_id = update.effective_user.id
    tracked_products = db.get_tracked_products(user_id)
    
    if not tracked_products:
        await update.message.reply_text("📭 You're not tracking any products yet.\n\nUse <code>/track [Amazon URL]</code> to start!", parse_mode=ParseMode.HTML)
        return
    
    message = f"📦 <b>Your Tracked Products ({len(tracked_products)}):</b>\n\n"
    
    for i, product in enumerate(tracked_products, 1):
        truncated_title = truncate_title(product['title'], 60)
        current_price = format_price(product['last_price'])
        affiliate_link = build_affiliate_link(product['asin'])
        
        message += f"{i}. <b>{html.escape(truncated_title)}</b>\n"
        message += f"   💰 <b>{current_price}</b>"
        
        if product['target_price']:
            target_formatted = format_price(product['target_price'])
            message += f" (🎯 {target_formatted})"
        
        message += f"\n   🛒 <a href='{affiliate_link}'>Buy Now</a> | ASIN: <code>{product['asin']}</code>\n\n"
    
    if DISCLOSURE_TEXT:
        message += f"{DISCLOSURE_TEXT}\n\n"
    
    message += "Use <code>/untrack [ASIN]</code> to stop tracking a product."
    
    await update.message.reply_text(
        message, 
        parse_mode=ParseMode.HTML, 
        disable_web_page_preview=True,
        reply_markup=get_back_keyboard()
    )

async def untrack_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /untrack command"""
    if not context.args:
        await update.message.reply_text("Please provide an ASIN to untrack.\nUsage: <code>/untrack [ASIN]</code>", parse_mode=ParseMode.HTML)
        return
    
    asin = context.args[0].upper()
    user_id = update.effective_user.id
    
    if db.remove_tracked_product(user_id, asin):
        await update.message.reply_text(f"✅ Stopped tracking product with ASIN: <code>{asin}</code>", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(f"❌ Product with ASIN <code>{asin}</code> was not being tracked.", parse_mode=ParseMode.HTML)

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command"""
    user_id = update.effective_user.id
    removed_count = db.clear_tracked_products(user_id)
    
    if removed_count > 0:
        await update.message.reply_text(f"✅ Removed {removed_count} tracked product(s).")
    else:
        await update.message.reply_text("📭 You weren't tracking any products.")

async def trackkw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /trackkw command"""
    if not context.args:
        await update.message.reply_text(
            "Please provide a keyword to track.\n"
            "Usage: <code>/trackkw [keyword] [target_price]</code>\n"
            "Example: <code>/trackkw iPhone 15 50000</code>",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Parse keyword and target price
    if len(context.args) == 1:
        keyword = context.args[0]
        target_price = None
    else:
        # Last argument might be target price
        try:
            target_price = int(float(context.args[-1]) * 100)  # Convert to paise
            keyword = " ".join(context.args[:-1])
        except ValueError:
            # Last argument is not a price, treat all as keyword
            keyword = " ".join(context.args)
            target_price = None
    
    user_id = update.effective_user.id
    db.add_keyword_alert(user_id, keyword, target_price)
    
    message = f"✅ <b>Keyword alert set:</b>\n\n"
    message += f"🔍 Keyword: <b>{html.escape(keyword)}</b>\n"
    
    if target_price:
        target_formatted = format_price(target_price)
        message += f"🎯 Target Price: <b>{target_formatted}</b>\n"
        message += "\nI'll notify you when I find products matching this keyword at or below your target price."
    else:
        message += "\nI'll notify you when I find new products matching this keyword."
    
    await update.message.reply_text(message, parse_mode=ParseMode.HTML)

async def listkw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /listkw command"""
    user_id = update.effective_user.id
    keyword_alerts = db.get_keyword_alerts(user_id)
    
    if not keyword_alerts:
        await update.message.reply_text(
            "🔍 You don't have any keyword alerts set.\n\n"
            "Use <code>/trackkw [keyword]</code> to create one!",
            parse_mode=ParseMode.HTML
        )
        return
    
    message = f"🔍 <b>Your Keyword Alerts ({len(keyword_alerts)}):</b>\n\n"
    
    for i, alert in enumerate(keyword_alerts, 1):
        message += f"{i}. <b>{html.escape(alert['keyword'])}</b>"
        
        if alert['target_price']:
            target_formatted = format_price(alert['target_price'])
            message += f" (🎯 {target_formatted})"
        
        message += "\n"
    
    message += "\nUse <code>/untrackkw [keyword]</code> to remove an alert."
    
    await update.message.reply_text(message, parse_mode=ParseMode.HTML)

async def untrackkw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /untrackkw command"""
    if not context.args:
        await update.message.reply_text("Please provide a keyword to remove.\nUsage: <code>/untrackkw [keyword]</code>", parse_mode=ParseMode.HTML)
        return
    
    keyword = " ".join(context.args)
    user_id = update.effective_user.id
    
    if db.remove_keyword_alert(user_id, keyword):
        await update.message.reply_text(f"✅ Removed keyword alert: <b>{html.escape(keyword)}</b>", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(f"❌ Keyword alert <b>{html.escape(keyword)}</b> was not found.", parse_mode=ParseMode.HTML)

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /chart command"""
    if not MATPLOTLIB_AVAILABLE:
        await update.message.reply_text(
            "📊 Price charts are not available in this deployment.\n"
            "Charts require matplotlib to be installed and configured properly."
        )
        return
    
    if not context.args:
        await update.message.reply_text("Please provide an ASIN to chart.\nUsage: <code>/chart [ASIN] [days]</code>", parse_mode=ParseMode.HTML)
        return
    
    asin = context.args[0].upper()
    days = 90
    
    if len(context.args) > 1:
        try:
            days = int(context.args[1])
            if days < 1 or days > 365:
                days = 90
        except ValueError:
            pass
    
    # Get price history
    price_history = db.get_price_history(asin, days)
    
    if not price_history:
        await update.message.reply_text(f"📊 No price history available for ASIN: <code>{asin}</code>", parse_mode=ParseMode.HTML)
        return
    
    try:
        # Create chart
        dates = [datetime.fromtimestamp(ts) for ts, _ in price_history]
        prices = [price / 100 for _, price in price_history]  # Convert to rupees
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, prices, marker='o', linewidth=2, markersize=4)
        plt.title(f'Price History - ASIN: {asin}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (₹)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
        plt.xticks(rotation=45)
        
        # Format y-axis to show prices
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f"chart_{asin}_{int(time.time())}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Send chart
        with open(chart_path, 'rb') as chart_file:
            caption = f"📊 <b>Price History for ASIN: {asin}</b>\n"
            caption += f"📅 Last {days} days | {len(price_history)} data points\n"
            caption += f"💰 Current: {format_price(price_history[-1][1])}\n"
            
            if len(price_history) > 1:
                min_price = min(price for _, price in price_history)
                max_price = max(price for _, price in price_history)
                caption += f"📈 Range: {format_price(min_price)} - {format_price(max_price)}"
            
            affiliate_link = build_affiliate_link(asin)
            caption += f"\n\n🛒 <a href='{affiliate_link}'>Buy on Amazon</a>"
            
            if DISCLOSURE_TEXT:
                caption += f"\n\n{DISCLOSURE_TEXT}"
            
            await update.message.reply_photo(
                photo=chart_file,
                caption=caption,
                parse_mode=ParseMode.HTML
            )
        
        # Clean up
        os.remove(chart_path)
        
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        await update.message.reply_text("❌ Error generating price chart. Please try again later.")

def get_user_display_name(user) -> str:
    """Get user's display name for channel posting"""
    if user.username:
        return f"@{user.username}"
    elif user.first_name:
        full_name = user.first_name
        if user.last_name:
            full_name += f" {user.last_name}"
        return full_name
    else:
        return f"User {user.id}"

async def post_new_product_to_channel(context: ContextTypes.DEFAULT_TYPE, user, product_data: Dict, asin: str, target_price: Optional[int] = None):
    """Post newly tracked product to channel with user info"""
    if not MAIN_CHANNEL_ID:
        return
    
    try:
        user_name = get_user_display_name(user)
        truncated_title = truncate_title(product_data['title'])
        current_price = format_price(product_data['price'])
        affiliate_link = build_affiliate_link(asin)
        
        message = f"📦 <b>NEW PRODUCT TRACKED</b>\n\n"
        message += f"👤 <b>Tracked by:</b> {html.escape(user_name)}\n\n"
        message += f"🛍️ <b>{html.escape(truncated_title)}</b>\n"
        message += f"💰 <b>Price:</b> {current_price}\n"
        
        if target_price:
            target_formatted = format_price(target_price)
            message += f"🎯 <b>Target Price:</b> {target_formatted}\n"
        
        message += f"\n🛒 <a href='{affiliate_link}'>Buy on Amazon</a>\n"
        message += f"📋 ASIN: <code>{asin}</code>\n\n"
        message += "💡 <i>I'll notify when price drops!</i>\n\n"
        message += "📢 @DealAlertAi"
        
        if DISCLOSURE_TEXT:
            message += f"\n\n{DISCLOSURE_TEXT}"
        
        await context.bot.send_message(
            chat_id=MAIN_CHANNEL_ID,
            text=message,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=False
        )
        
        logger.info(f"Posted new product {asin} to channel by user {user.id}")
        
    except Exception as e:
        logger.warning(f"Failed to post new product to channel {MAIN_CHANNEL_ID}: {e}")

async def send_notification(context: ContextTypes.DEFAULT_TYPE, user_id: int, message: str, send_to_channel: bool = False):
    """Send notification to user and optionally to channel"""
    try:
        # Send to user
        await context.bot.send_message(
            chat_id=user_id,
            text=message,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=False
        )
        
        # Send to main channel if configured and requested
        if send_to_channel and MAIN_CHANNEL_ID:
            try:
                await context.bot.send_message(
                    chat_id=MAIN_CHANNEL_ID,
                    text=f"🔥 <b>DEAL ALERT</b>\n\n{message}\n\n📢 @DealAlertAi",
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=False
                )
            except Exception as e:
                logger.warning(f"Failed to send to channel {MAIN_CHANNEL_ID}: {e}")
    
    except Exception as e:
        logger.error(f"Failed to send notification to user {user_id}: {e}")

async def check_tracked_products(context: ContextTypes.DEFAULT_TYPE):
    """Scheduled job to check tracked products"""
    logger.info("Starting tracked products check...")
    
    all_products = db.get_all_tracked_products()
    
    for product in all_products:
        try:
            # Scrape current data
            current_data = await scraper.scrape_product(product['asin'])
            if not current_data:
                logger.warning(f"Could not fetch data for ASIN {product['asin']}")
                continue
            
            old_price = product['last_price']
            new_price = current_data['price']
            
            # Add to price history
            db.add_price_history(product['asin'], new_price)
            
            # Check for notifications
            should_notify = False
            notification_reason = ""
            send_to_channel = False
            
            # Price dropped
            if new_price < old_price:
                should_notify = True
                notification_reason = "📉 Price Drop"
                send_to_channel = True
            
            # Target price hit
            elif product['target_price'] and new_price <= product['target_price']:
                if product['last_notified_price'] is None or new_price < product['last_notified_price']:
                    should_notify = True
                    notification_reason = "🎯 Target Price Hit"
                    send_to_channel = True
            
            # Lowest price in 90 days
            else:
                lowest_90_days = db.get_lowest_price_days(product['asin'], 90)
                if lowest_90_days and new_price <= lowest_90_days and new_price < old_price:
                    should_notify = True
                    notification_reason = "📊 90-Day Low"
                    send_to_channel = True
            
            # Send notification if needed
            if should_notify:
                truncated_title = truncate_title(current_data['title'])
                affiliate_link = build_affiliate_link(product['asin'])
                
                message = f"🚨 <b>{notification_reason}!</b>\n\n"
                message += f"📦 <b>{html.escape(truncated_title)}</b>\n\n"
                
                if old_price != new_price:
                    old_formatted = format_price(old_price)
                    new_formatted = format_price(new_price)
                    savings = format_price(old_price - new_price)
                    message += f"💰 <s>{old_formatted}</s> → <b>{new_formatted}</b>\n"
                    message += f"💵 You save: <b>{savings}</b>\n\n"
                else:
                    message += f"💰 Price: <b>{format_price(new_price)}</b>\n\n"
                
                message += f"🛒 <a href='{affiliate_link}'>Buy Now on Amazon</a>"
                
                if DISCLOSURE_TEXT:
                    message += f"\n\n{DISCLOSURE_TEXT}"
                
                await send_notification(context, product['user_id'], message, send_to_channel)
                
                # Update last notified price
                db.update_tracked_product(
                    user_id=product['user_id'],
                    asin=product['asin'],
                    last_notified_price=new_price
                )
            
            # Update product data
            db.update_tracked_product(
                user_id=product['user_id'],
                asin=product['asin'],
                title=current_data['title'],
                price=new_price,
                image=current_data['image']
            )
            
            # Small delay between requests
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error checking product {product['asin']}: {e}")
    
    logger.info(f"Completed check of {len(all_products)} tracked products")

async def check_keyword_alerts(context: ContextTypes.DEFAULT_TYPE):
    """Scheduled job to check keyword alerts"""
    logger.info("Starting keyword alerts check...")
    
    all_keywords = db.get_all_keyword_alerts()
    
    for kw_alert in all_keywords:
        try:
            keyword = kw_alert['keyword']
            target_price = kw_alert['target_price']
            user_id = kw_alert['user_id']
            
            # Search for products
            products = await scraper.search_products(keyword, max_results=20)
            
            for product in products:
                asin = product['asin']
                price = product['price']
                
                # Check if target price is met (if set)
                if target_price and price > target_price:
                    continue
                
                # Check if already notified at this price
                if db.check_kw_seen(user_id, keyword, asin, price):
                    continue
                
                # Send notification
                truncated_title = truncate_title(product['title'])
                affiliate_link = build_affiliate_link(asin)
                
                message = f"🔍 <b>Keyword Alert: {html.escape(keyword)}</b>\n\n"
                message += f"📦 <b>{html.escape(truncated_title)}</b>\n"
                message += f"💰 Price: <b>{format_price(price)}</b>\n"
                
                if target_price:
                    message += f"🎯 Target: <b>{format_price(target_price)}</b>\n"
                
                message += f"\n🛒 <a href='{affiliate_link}'>Buy Now on Amazon</a>"
                
                if DISCLOSURE_TEXT:
                    message += f"\n\n{DISCLOSURE_TEXT}"
                
                await send_notification(context, user_id, message, send_to_channel=True)
                
                # Mark as seen
                db.mark_kw_seen(user_id, keyword, asin, price)
                
                # Small delay between notifications
                await asyncio.sleep(1)
            
            # Delay between keywords
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error checking keyword alert '{kw_alert['keyword']}': {e}")
    
    logger.info(f"Completed check of {len(all_keywords)} keyword alerts")

# New commands and features

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_message = """
🤖 <b>Deal Alert AI - Complete Guide</b>

<b>📦 Product Tracking:</b>
• Track Amazon.in products with price alerts
• Set target prices for notifications
• Monitor price drops and historical lows

<b>🔍 Keyword Discovery:</b>
• Set keyword alerts for categories
• Discover new deals automatically
• Filter by target prices

<b>🤖 AI Features:</b>
• Compare prices with AI analysis
• Get market insights and recommendations
• Smart deal evaluation

<b>📢 Channel Broadcasting:</b>
• Auto-post deals to your Telegram channel
• Customize posting settings
• Test channel connectivity

<b>📊 Analytics:</b>
• View price history charts
• Track 90-day lows
• Monitor savings over time

<b>⚡ Quick Commands:</b>
• <code>/start</code> - Main menu
• <code>/help</code> - This guide
• <code>/track [URL] [price]</code> - Quick track
• <code>/list</code> - View tracked items
• <code>/ai [query]</code> - AI price comparison

<b>💡 Pro Tips:</b>
• Use inline buttons for easier navigation
• Set realistic target prices
• Check your channel setup for auto-posting
• Use AI compare for best deals

Need more help? Use the menu buttons below!
"""
    
    await update.message.reply_text(
        help_message,
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_menu_keyboard()
    )

async def post_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /post command for manual channel posting"""
    if not MAIN_CHANNEL_ID:
        await update.message.reply_text(
            "❌ <b>Channel Not Configured</b>\n\n"
            "Please set MAIN_CHANNEL_ID environment variable first to enable manual posting.",
            parse_mode=ParseMode.HTML,
            reply_markup=get_back_keyboard()
        )
        return
    
    if not context.args:
        await update.message.reply_text(
            "📢 <b>Manual Channel Posting</b>\n\n"
            "Post deals directly to your channel!\n\n"
            "<b>Usage:</b>\n"
            "<code>/post [Amazon URL] [optional custom message]</code>\n\n"
            "<b>Examples:</b>\n"
            "• <code>/post https://amzn.in/d/eH3ADFx</code>\n"
            "• <code>/post https://amazon.in/dp/B08N5WRWNW Great deal alert!</code>\n\n"
            "💡 <i>The bot will fetch product details and post to channel immediately.</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=get_back_keyboard()
        )
        return
    
    amazon_url = context.args[0]
    custom_message = " ".join(context.args[1:]) if len(context.args) > 1 else ""
    
    # Extract ASIN
    asin = await scraper.extract_asin(amazon_url)
    if not asin:
        await update.message.reply_text(
            "❌ Could not extract product ID from URL. Please check the link format.",
            reply_markup=get_back_keyboard()
        )
        return
    
    await update.message.reply_text("🔍 Fetching product details for manual post...")
    
    try:
        # Scrape product data
        product_data = await scraper.scrape_product(asin)
        if not product_data:
            await update.message.reply_text(
                "❌ Could not fetch product details. Please try again later.",
                reply_markup=get_back_keyboard()
            )
            return
        
        # Build manual post message
        user_name = get_user_display_name(update.effective_user)
        truncated_title = truncate_title(product_data['title'])
        current_price = format_price(product_data['price'])
        affiliate_link = build_affiliate_link(asin)
        
        message = f"📢 <b>MANUAL POST</b>\n\n"
        message += f"👤 <b>Posted by:</b> {html.escape(user_name)}\n\n"
        
        if custom_message:
            message += f"💬 <i>{html.escape(custom_message)}</i>\n\n"
        
        message += f"🛍️ <b>{html.escape(truncated_title)}</b>\n"
        message += f"💰 <b>Price:</b> {current_price}\n\n"
        message += f"🛒 <a href='{affiliate_link}'>Buy on Amazon</a>\n"
        message += f"📋 ASIN: <code>{asin}</code>"
        
        if DISCLOSURE_TEXT:
            message += f"\n\n{DISCLOSURE_TEXT}"
        
        # Post to channel
        await context.bot.send_message(
            chat_id=MAIN_CHANNEL_ID,
            text=message,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=False
        )
        
        # Confirm to user
        await update.message.reply_text(
            f"✅ <b>Posted to Channel!</b>\n\n"
            f"📦 {html.escape(truncated_title)}\n"
            f"💰 {current_price}\n\n"
            f"📢 Successfully posted to: <code>{MAIN_CHANNEL_ID}</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=get_main_menu_keyboard()
        )
        
        logger.info(f"Manual post to channel by user {update.effective_user.id}: {asin}")
        
    except Exception as e:
        logger.error(f"Error in manual post: {e}")
        await update.message.reply_text(
            "❌ Failed to post to channel. Please try again or check channel permissions.",
            reply_markup=get_back_keyboard()
        )

async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ai command for price comparison"""
    if not GEMINI_AVAILABLE or not gemini_client:
        await update.message.reply_text(
            "🤖 <b>AI Features Unavailable</b>\n\n"
            "AI price comparison is not available. Please ask the administrator to set up Gemini API.",
            parse_mode=ParseMode.HTML,
            reply_markup=get_back_keyboard()
        )
        return
    
    if not context.args:
        await update.message.reply_text(
            "🤖 <b>AI Price Comparison</b>\n\n"
            "Ask me to compare prices or analyze deals!\n\n"
            "<b>Examples:</b>\n"
            "• <code>/ai compare iPhone 15 prices</code>\n"
            "• <code>/ai is 50000 good for iPhone 15</code>\n"
            "• <code>/ai best laptop under 80000</code>\n\n"
            "I'll help you make smart buying decisions!",
            parse_mode=ParseMode.HTML,
            reply_markup=get_back_keyboard()
        )
        return
    
    query = " ".join(context.args)
    await update.message.reply_text("🤖 Analyzing with AI... Please wait...")
    
    try:
        # Prepare AI prompt
        ai_prompt = f"""
You are an expert e-commerce price analyst for Amazon.in. A user asked: "{query}"

Provide helpful insights about:
1. Current market prices in India (₹)
2. Whether mentioned prices are good deals
3. Best alternatives or recommendations
4. When to buy vs wait
5. Key features to consider

Keep response concise, practical, and focused on Indian market. Use ₹ for prices.
"""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=ai_prompt
        )
        
        ai_response = response.text if response.text else "Sorry, I couldn't analyze that right now."
        
        message = f"🤖 <b>AI Price Analysis</b>\n\n{ai_response}\n\n"
        message += "💡 <i>This analysis is AI-generated. Always verify prices before purchasing.</i>"
        
        await update.message.reply_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=get_back_keyboard()
        )
        
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        await update.message.reply_text(
            "❌ AI analysis failed. Please try again later.",
            reply_markup=get_back_keyboard()
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages, especially Amazon URLs"""
    message_text = update.message.text
    
    # Check if message contains Amazon URL
    amazon_patterns = [
        r'amazon\.in',
        r'amzn\.to',
        r'amzn\.in',
        r'/dp/',
        r'/gp/product/'
    ]
    
    is_amazon_url = any(pattern in message_text.lower() for pattern in amazon_patterns)
    
    if is_amazon_url:
        # Extract ASIN from the URL
        asin = await scraper.extract_asin(message_text)
        if not asin:
            await update.message.reply_text(
                "❌ Could not extract product ID from the URL. Please check the link format.",
                reply_markup=get_back_keyboard()
            )
            return
        
        await update.message.reply_text("🔍 Fetching product details...")
        
        try:
            # Scrape product data
            product_data = await scraper.scrape_product(asin)
            if not product_data:
                await update.message.reply_text(
                    "❌ Could not fetch product details. Please try again later or check if the URL is valid.",
                    reply_markup=get_back_keyboard()
                )
                return
            
            # Save to database
            user_id = update.effective_user.id
            db.add_tracked_product(
                user_id=user_id,
                asin=asin,
                title=product_data['title'],
                price=product_data['price'],
                image=product_data['image']
            )
            
            # Add to price history
            db.add_price_history(asin, product_data['price'])
            
            # Post to channel automatically
            await post_new_product_to_channel(context, update.effective_user, product_data, asin)
            
            # Build response message
            affiliate_link = build_affiliate_link(asin)
            truncated_title = truncate_title(product_data['title'])
            current_price = format_price(product_data['price'])
            
            message = f"✅ <b>Now tracking:</b>\n\n"
            message += f"📦 <b>{html.escape(truncated_title)}</b>\n"
            message += f"💰 Current Price: <b>{current_price}</b>\n"
            message += f"\n🛒 <a href='{affiliate_link}'>Buy on Amazon</a>\n\n"
            message += "📢 <i>Product posted to channel!</i> To set a target price, use:\n"
            message += f"<code>/track {message_text} [target_price]</code>"
            
            if DISCLOSURE_TEXT:
                message += f"\n\n{DISCLOSURE_TEXT}"
            
            await update.message.reply_text(
                message, 
                parse_mode=ParseMode.HTML, 
                disable_web_page_preview=False,
                reply_markup=get_main_menu_keyboard()
            )
            
        except Exception as e:
            logger.error(f"Error in handle_message: {e}")
            await update.message.reply_text(
                "❌ An error occurred while tracking the product. Please try again.",
                reply_markup=get_back_keyboard()
            )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    callback_data = query.data
    
    async def safe_edit_message(text, reply_markup=None, parse_mode=ParseMode.HTML):
        """Safely edit message with error handling"""
        try:
            await query.edit_message_text(
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
        except Exception as e:
            # Handle "message is not modified" error silently
            if "message is not modified" not in str(e).lower():
                logger.error(f"Error editing message: {e}")
                # Send new message if editing fails
                await query.message.reply_text(
                    text=text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup
                )
    
    if callback_data == "main_menu":
        await safe_edit_message(
            "🤖 <b>Deal Alert AI - Main Menu</b>\n\n"
            "Choose an option below to get started:",
            reply_markup=get_main_menu_keyboard()
        )
    
    elif callback_data == "menu_track":
        await safe_edit_message(
            "📦 <b>Product Tracking</b>\n\n"
            "Manage your Amazon product tracking:",
            reply_markup=get_track_menu_keyboard()
        )
    
    elif callback_data == "menu_keywords":
        await safe_edit_message(
            "🔍 <b>Keyword Alerts</b>\n\n"
            "Set up keyword-based deal discovery:",
            reply_markup=get_keywords_menu_keyboard()
        )
    
    elif callback_data == "menu_lists":
        # Show tracked products
        user_id = query.from_user.id
        tracked_products = db.get_tracked_products(user_id)
        tracked_keywords = db.get_keyword_alerts(user_id)
        
        message = f"📋 <b>Your Lists</b>\n\n"
        message += f"📦 Tracked Products: {len(tracked_products)}\n"
        message += f"🔍 Keyword Alerts: {len(tracked_keywords)}\n\n"
        message += "Use the menu to manage your lists:"
        
        await safe_edit_message(
            message,
            reply_markup=get_main_menu_keyboard()
        )
    
    elif callback_data == "menu_charts":
        await safe_edit_message(
            "📊 <b>Price Charts</b>\n\n"
            "Use <code>/chart [ASIN] [days]</code> to view price history.\n\n"
            "<b>Example:</b>\n"
            "<code>/chart B08N5WRWNW 30</code>",
            reply_markup=get_back_keyboard()
        )
    
    elif callback_data == "menu_ai":
        await safe_edit_message(
            "🤖 <b>AI Price Comparison</b>\n\n"
            "Get smart price analysis and recommendations!\n\n"
            "Use <code>/ai [your question]</code>\n\n"
            "<b>Examples:</b>\n"
            "• <code>/ai compare iPhone 15 prices</code>\n"
            "• <code>/ai is 50000 good for iPhone 15</code>\n"
            "• <code>/ai best laptop under 80000</code>",
            reply_markup=get_back_keyboard()
        )
    
    elif callback_data == "menu_post":
        await safe_edit_message(
            "📢 <b>Manual Channel Posting</b>\n\n"
            "Post deals directly to your channel instantly!\n\n"
            "Use <code>/post [Amazon URL] [optional message]</code>\n\n"
            "<b>Examples:</b>\n"
            "• <code>/post https://amzn.in/d/eH3ADFx</code>\n"
            "• <code>/post https://amazon.in/dp/B08N5WRWNW Hot deal!</code>\n\n"
            "💡 <i>Perfect for sharing hot deals instantly!</i>",
            reply_markup=get_back_keyboard()
        )
    
    elif callback_data == "menu_channel":
        await safe_edit_message(
            "📢 <b>Channel Broadcasting</b>\n\n"
            "Manage auto-posting to your Telegram channel:",
            reply_markup=get_channel_menu_keyboard()
        )
    
    elif callback_data == "menu_help":
        help_message = """
🤖 <b>Deal Alert AI - Quick Help</b>

<b>🚀 Getting Started:</b>
1. Use /track [Amazon URL] to track products
2. Set /trackkw [keyword] for deal discovery
3. Configure channel for auto-posting

<b>⚡ Quick Commands:</b>
• /start - Main menu
• /help - Full guide
• /track [URL] - Track product
• /list - View tracked items
• /ai [query] - AI assistance

<b>💡 Tips:</b>
• Use realistic target prices
• Check your channel setup
• Try AI for price comparisons

Use the buttons for easy navigation!
"""
        await safe_edit_message(
            help_message,
            reply_markup=get_back_keyboard()
        )
    
    elif callback_data == "channel_view":
        channel_status = "✅ Configured" if MAIN_CHANNEL_ID else "❌ Not Set"
        await safe_edit_message(
            f"📢 <b>Channel Status</b>\n\n"
            f"Current Channel: <code>{MAIN_CHANNEL_ID or 'Not configured'}</code>\n"
            f"Status: {channel_status}\n\n"
            f"To set up channel broadcasting:\n"
            f"1. Create a Telegram channel\n"
            f"2. Add your bot as admin\n"
            f"3. Set MAIN_CHANNEL_ID in environment\n\n"
            f"Channel auto-posting {'enabled' if MAIN_CHANNEL_ID else 'disabled'}.",
            reply_markup=get_channel_menu_keyboard()
        )
    
    elif callback_data == "track_new":
        await safe_edit_message(
            "➕ <b>Track New Product</b>\n\n"
            "Send me an Amazon.in product URL to start tracking!\n\n"
            "<b>Usage:</b>\n"
            "• Just send the Amazon URL\n"
            "• Or use: <code>/track [URL] [target_price]</code>\n\n"
            "<b>Example:</b>\n"
            "<code>/track https://amazon.in/dp/B08N5WRWNW 50000</code>",
            reply_markup=get_track_menu_keyboard()
        )
    
    elif callback_data == "track_list":
        user_id = query.from_user.id
        tracked_products = db.get_tracked_products(user_id)
        
        if not tracked_products:
            await safe_edit_message(
                "📭 <b>No Products Tracked</b>\n\n"
                "You're not tracking any products yet.\n\n"
                "Use the '➕ Track New Product' button to start!",
                reply_markup=get_track_menu_keyboard()
            )
        else:
            message = f"📦 <b>Your Tracked Products ({len(tracked_products)}):</b>\n\n"
            
            for i, product in enumerate(tracked_products[:10], 1):  # Limit to 10 for display
                truncated_title = truncate_title(product['title'], 50)
                current_price = format_price(product['last_price'])
                
                message += f"{i}. <b>{html.escape(truncated_title)}</b>\n"
                message += f"   💰 {current_price}"
                
                if product['target_price']:
                    target_formatted = format_price(product['target_price'])
                    message += f" (🎯 {target_formatted})"
                
                message += f"\n   ASIN: <code>{product['asin']}</code>\n\n"
            
            if len(tracked_products) > 10:
                message += f"... and {len(tracked_products) - 10} more products."
            
            await safe_edit_message(
                message,
                reply_markup=get_track_menu_keyboard()
            )
    
    elif callback_data == "track_remove":
        await safe_edit_message(
            "❌ <b>Remove Product</b>\n\n"
            "Use the command: <code>/untrack [ASIN]</code>\n\n"
            "<b>Example:</b>\n"
            "<code>/untrack B08N5WRWNW</code>\n\n"
            "You can find ASINs in your tracked products list.",
            reply_markup=get_track_menu_keyboard()
        )
    
    elif callback_data == "track_clear":
        await safe_edit_message(
            "🗑️ <b>Clear All Products</b>\n\n"
            "Use the command: <code>/clear</code>\n\n"
            "⚠️ This will remove ALL your tracked products!",
            reply_markup=get_track_menu_keyboard()
        )
    
    elif callback_data == "kw_new":
        await safe_edit_message(
            "➕ <b>Add Keyword Alert</b>\n\n"
            "Set up keyword-based deal discovery!\n\n"
            "<b>Usage:</b>\n"
            "<code>/trackkw [keyword] [target_price]</code>\n\n"
            "<b>Examples:</b>\n"
            "• <code>/trackkw iPhone 15</code>\n"
            "• <code>/trackkw laptop 50000</code>\n"
            "• <code>/trackkw headphones 5000</code>",
            reply_markup=get_keywords_menu_keyboard()
        )
    
    elif callback_data == "kw_list":
        user_id = query.from_user.id
        keyword_alerts = db.get_keyword_alerts(user_id)
        
        if not keyword_alerts:
            await safe_edit_message(
                "🔍 <b>No Keyword Alerts</b>\n\n"
                "You don't have any keyword alerts set.\n\n"
                "Use '➕ Add Keyword Alert' to create one!",
                reply_markup=get_keywords_menu_keyboard()
            )
        else:
            message = f"🔍 <b>Your Keyword Alerts ({len(keyword_alerts)}):</b>\n\n"
            
            for i, alert in enumerate(keyword_alerts, 1):
                message += f"{i}. <b>{html.escape(alert['keyword'])}</b>"
                
                if alert['target_price']:
                    target_formatted = format_price(alert['target_price'])
                    message += f" (🎯 {target_formatted})"
                
                message += "\n"
            
            await safe_edit_message(
                message,
                reply_markup=get_keywords_menu_keyboard()
            )
    
    elif callback_data == "kw_remove":
        await safe_edit_message(
            "❌ <b>Remove Keyword Alert</b>\n\n"
            "Use the command: <code>/untrackkw [keyword]</code>\n\n"
            "<b>Example:</b>\n"
            "<code>/untrackkw iPhone 15</code>",
            reply_markup=get_keywords_menu_keyboard()
        )

    elif callback_data == "channel_test":
        if not MAIN_CHANNEL_ID:
            await safe_edit_message(
                "❌ <b>Channel Not Configured</b>\n\n"
                "Please set MAIN_CHANNEL_ID environment variable first.",
                reply_markup=get_channel_menu_keyboard()
            )
        else:
            try:
                await context.bot.send_message(
                    chat_id=MAIN_CHANNEL_ID,
                    text="🧪 <b>Test Message</b>\n\nYour Deal Alert AI bot is connected and ready to broadcast deals!",
                    parse_mode=ParseMode.HTML
                )
                await safe_edit_message(
                    "✅ <b>Channel Test Successful!</b>\n\n"
                    f"Test message sent to: <code>{MAIN_CHANNEL_ID}</code>\n\n"
                    "Your channel is ready for auto-posting!",
                    reply_markup=get_channel_menu_keyboard()
                )
            except Exception as e:
                await safe_edit_message(
                    f"❌ <b>Channel Test Failed</b>\n\n"
                    f"Error: {str(e)}\n\n"
                    "Please check:\n"
                    "• Bot is admin in channel\n"
                    "• Channel ID is correct\n"
                    "• Bot has posting permissions",
                    reply_markup=get_channel_menu_keyboard()
                )

def main():
    """Main function to run the bot"""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN environment variable is required!")
        return
    
    logger.info("Starting Deal Alert AI Bot...")
    
    # Create application with job queue
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("track", track_command))
    application.add_handler(CommandHandler("list", list_command))
    application.add_handler(CommandHandler("untrack", untrack_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("trackkw", trackkw_command))
    application.add_handler(CommandHandler("listkw", listkw_command))
    application.add_handler(CommandHandler("untrackkw", untrackkw_command))
    application.add_handler(CommandHandler("chart", chart_command))
    application.add_handler(CommandHandler("ai", ai_command))
    application.add_handler(CommandHandler("post", post_command))
    
    # Add callback query handler for inline buttons
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Add message handler for Amazon URLs
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Schedule jobs
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(check_tracked_products, interval=CHECK_INTERVAL, first=60)
        job_queue.run_repeating(check_keyword_alerts, interval=CHECK_INTERVAL, first=120)
        logger.info(f"Scheduled jobs set up successfully")
    else:
        logger.warning("JobQueue not available - scheduled monitoring disabled")
    
    logger.info(f"Bot started! Checking products every {CHECK_INTERVAL/3600:.1f} hours")
    if MAIN_CHANNEL_ID:
        logger.info(f"Channel broadcasting enabled: {MAIN_CHANNEL_ID}")
    
    # Run bot with long polling
    application.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
