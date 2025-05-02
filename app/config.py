import os
import logging
import datetime
import ccxt
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

# --- API-ключи и инициализация синхронных клиентов CCXT ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
GATEIO_API_KEY = os.getenv("GATEIO_API_KEY", "")
GATEIO_API_SECRET = os.getenv("GATEIO_API_SECRET", "")

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


def get_exchange_client(exchange: str):
    """
    Синхронно возвращает CCXT-клиент для указанной биржи.
    Если биржа не в ALLOWED_EXCHANGES – падаем на binance.
    """
    exch = exchange.lower()
    if exch not in EXCHANGE_CLIENTS:
        logging.warning(f"Exchange '{exchange}' не разрешена, используется 'binance'.")
        exch = "binance"
    return EXCHANGE_CLIENTS[exch]


# (если вдруг нужен отдельный простой fetch_ohlcv – но лучше использовать fetch_data_from_exchange)
def fetch_ohlcv(exchange: str, symbol: str, timeframe: str = "1h", limit: int = 100):
    """
    Пример синхронного дёргания OHLCV, возвращает список или [].
    """
    client = get_exchange_client(exchange)
    try:
        return client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        logging.error(f"[fetch_ohlcv] Ошибка {exchange}/{symbol}: {e}")
        return []


#---Api OenAI---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY",
                           "sk"
                           "-9DbUr1fbG3C9F1ZLnV_nF5OpxKZ6086GyfqKRV71tWT3BlbkFJIJPdGtdNXvO4UXm6AgLndnjsl3sLicp7jtix5zYMYA")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Конфиги новостных API и RSS – без изменений ---
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
]

CRYPTO_RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed"
}

SOURCE_RELIABILITY = {
    "bloomberg": 1.0,
    "reuters": 1.0,
    "cnbc": 0.9,
    "financial-times": 0.95,
    "forbes": 0.85,
    "techcrunch": 0.8,
    "the-wall-street-journal": 0.95,
    "coindesk": 0.9,
    "cointelegraph": 0.85,
    "decrypt": 0.8,
}
