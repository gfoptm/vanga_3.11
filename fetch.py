import logging
import pandas as pd
import datetime
from typing import Optional, Dict, List

from app.config import (
    ALLOWED_EXCHANGES,
    ALLOWED_SYMBOLS,
    ALLOWED_INTERVALS,
    INTERVAL_MAPPING,
    get_exchange_client
)


def validate_symbol(symbol: str) -> str:
    return symbol if symbol in ALLOWED_SYMBOLS else ALLOWED_SYMBOLS[0]


def validate_exchange(exchange: str) -> str:
    return exchange if exchange in ALLOWED_EXCHANGES else ALLOWED_EXCHANGES[0]


def validate_interval(interval: str) -> str:
    return interval if interval in ALLOWED_INTERVALS else ALLOWED_INTERVALS[0]


def interval_to_timedelta(interval: str) -> datetime.timedelta:
    if interval.endswith("m"):
        return datetime.timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return datetime.timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return datetime.timedelta(days=int(interval[:-1]))
    return datetime.timedelta(hours=1)


def fetch_data_from_exchange(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500
) -> pd.DataFrame:
    """
    Синхронно забирает OHLCV данные через ccxt.
    Всегда возвращает один pd.DataFrame (или пустой).
    """
    exchange = validate_exchange(exchange)
    symbol = validate_symbol(symbol)
    interval = validate_interval(interval)
    timeframe = INTERVAL_MAPPING[interval]

    logging.info(f"[Data] fetch_data_from_exchange: {exchange}, {symbol}, {interval}, limit={limit}")

    client = get_exchange_client(exchange)
    if client is None:
        logging.error(f"[Data] Клиент для биржи {exchange} не найден.")
        return pd.DataFrame()

    try:
        raw = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            logging.warning(f"[Data] Нет данных OHLCV от {exchange} для {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # конвертим типы
        df = df.astype({
            "timestamp": int,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float
        })
        logging.info(f"[Data] Получена форма данных от {exchange}: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"[Data] Ошибка при получении данных от {exchange} для {symbol}: {e}")
        return pd.DataFrame()


def fetch_all_data(
    exchanges: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    interval: str = "1h",
    limit: int = 500
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Обёртка для массового запроса:
      - exchanges: список бирж (по умолчанию ALLOWED_EXCHANGES)
      - symbols: список тикеров (по умолчанию ALLOWED_SYMBOLS)
      - возвращает {exchange: {symbol: DataFrame}}
    """
    exchs = exchanges or ALLOWED_EXCHANGES
    syms = symbols or ALLOWED_SYMBOLS

    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for exch in exchs:
        results[exch] = {}
        for sym in syms:
            df = fetch_data_from_exchange(exch, sym, interval, limit)
            results[exch][sym] = df
    return results
