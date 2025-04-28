import logging
import pandas as pd
import datetime

from config import ALLOWED_SYMBOLS, ALLOWED_EXCHANGES, ALLOWED_INTERVALS, INTERVAL_MAPPING, get_exchange_client

def validate_symbol(symbol: str) -> str:
    return symbol if symbol in ALLOWED_SYMBOLS else "BTCUSDT"

def validate_exchange(exchange: str) -> str:
    return exchange if exchange in ALLOWED_EXCHANGES else "binance"

def validate_interval(interval: str) -> str:
    return interval if interval in ALLOWED_INTERVALS else "1h"

def interval_to_timedelta(interval: str) -> datetime.timedelta:
    if interval.endswith("m"):
        return datetime.timedelta(minutes=int(interval[:-1]))
    elif interval.endswith("h"):
        return datetime.timedelta(hours=int(interval[:-1]))
    elif interval.endswith("d"):
        return datetime.timedelta(days=int(interval[:-1]))
    return datetime.timedelta(hours=1)

def fetch_data_from_exchange(exchange="binance", symbol="BTCUSDT", interval="1h", limit=500) -> pd.DataFrame:
    """
    Забирает OHLCV данные через ccxt.
    Формат: [[timestamp, open, high, low, close, volume], ...]
    """
    exchange = validate_exchange(exchange)
    symbol = validate_symbol(symbol)
    interval = validate_interval(interval)
    ccxt_timeframe = INTERVAL_MAPPING[interval]

    logging.info(f"[Data] fetch_data_from_exchange: {exchange}, {symbol}, {interval}, limit={limit}")

    client = get_exchange_client(exchange)
    if not client:
        logging.error(f"[Data] Клиент для биржи {exchange} не найден.")
        return pd.DataFrame()

    try:
        data = client.fetch_ohlcv(symbol, timeframe=ccxt_timeframe, limit=limit)
        if not data or len(data) == 0:
            logging.warning(f"[Data] Нет данных OHLCV от {exchange}")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        logging.info(f"[Data] Получена форма данных от {exchange}: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"[Data] Ошибка при получении данных от {exchange}: {e}")
        return pd.DataFrame()
