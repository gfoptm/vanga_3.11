import os
import logging
import datetime
import ccxt.async_support as ccxt
from openai import OpenAI

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ ---
ALLOWED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
ALLOWED_EXCHANGES = ["binance", "bybit", "gateio"]
ALLOWED_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

INTERVAL_MAPPING = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

window_size = 60
POPULATION_SIZE = 10
GENERATIONS = 5

# --- API‑ключи и инициализация асинхронных клиентов CCXT ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
GATEIO_API_KEY = os.getenv("GATEIO_API_KEY", "")
GATEIO_API_SECRET = os.getenv("GATEIO_API_SECRET", "")

# Асинхронные клиенты для бирж
EXCHANGE_CLIENTS = {
    "binance": ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "enableRateLimit": True,
    }),
    "bybit": ccxt.bybit({
        "apiKey": BYBIT_API_KEY,
        "secret": BYBIT_API_SECRET,
        "enableRateLimit": True,
    }),
    "gateio": ccxt.gateio({
        "apiKey": GATEIO_API_KEY,
        "secret": GATEIO_API_SECRET,
        "enableRateLimit": True,
    }),
}


async def get_exchange_client(exchange: str):
    """
    Возвращает асинхронный клиент CCXT для указанной биржи.
    Если биржа не разрешена, возвращается клиент Binance по умолчанию.
    """
    exchange = exchange.lower()
    if exchange not in EXCHANGE_CLIENTS:
        logging.warning(f"Exchange {exchange} не разрешена, используется binance.")
        exchange = "binance"
    return EXCHANGE_CLIENTS[exchange]


# Пример асинхронной функции для получения OHLCV данных
async def fetch_ohlcv(exchange: str, symbol: str, timeframe: str = "1h", limit: int = 100):
    client = await get_exchange_client(exchange)
    try:
        ohlcv = await client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return ohlcv
    finally:
        await client.close()


#---Api OenAI---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY",
                           "sk"
                           "-9DbUr1fbG3C9F1ZLnV_nF5OpxKZ6086GyfqKRV71tWT3BlbkFJIJPdGtdNXvO4UXm6AgLndnjsl3sLicp7jtix5zYMYA")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

#---Api News---
NEWS_API_CONFIGS = [
    {
        "name": "NewsAPI",
        "url": "https://newsapi.org/v2/everything",
        "api_keys": ["newsapi_key_1", "newsapi_key_2"],
        "params_base": {
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "sources": ",".join([
                "bloomberg", "reuters", "cnbc",
                "financial-times", "forbes",
                "techcrunch", "the-wall-street-journal"
            ])
        }
    },
    {
        "name": "GNews",
        "url": "https://gnews.io/api/v4/search",
        "api_keys": ["gnews_key_1"],
        "params_base": {
            "lang": "en",
            "max": 10,
            "sort": "publishedAt"
        }
    }
    # Дополнительно: можно добавить другие API (например, ContextualWeb, Bing News и т.д.)
]

# RSS-источники крипто-новостей
CRYPTO_RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed"
}

# Рейтинг надёжности источников (от 0 до 1)
SOURCE_RELIABILITY = {
    "bloomberg": 1.0,
    "reuters": 1.0,
    "cnbc": 0.9,
    "financial-times": 0.95,
    "forbes": 0.85,
    "techcrunch": 0.8,
    "the-wall-street-journal": 0.95,
    # Средние значения для RSS источников
    "coindesk": 0.9,
    "cointelegraph": 0.85,
    "decrypt": 0.8,
}
