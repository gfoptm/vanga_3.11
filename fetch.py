import logging
import pandas as pd
import datetime
from typing import List, Dict

from config import (
    ALLOWED_SYMBOLS,
    ALLOWED_EXCHANGES,
    ALLOWED_INTERVALS,
    INTERVAL_MAPPING,
    get_exchange_client
)


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


def fetch_data_from_exchange(
        exchange: str = "binance",
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500
) -> pd.DataFrame:
    """
    Забирает OHLCV данные через ccxt.
    Если на бирже нет запрошенной пары — сразу пропускает (возвращает пустой DataFrame).
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
        # загрузим список рынков и проверим поддержку символа
        client.load_markets()
        if symbol not in client.symbols:
            logging.warning(f"[Data] Пара {symbol} не поддерживается на {exchange}. Пропускаем.")
            return pd.DataFrame()

        data = client.fetch_ohlcv(symbol, timeframe=ccxt_timeframe, limit=limit)
        if not data:
            logging.warning(f"[Data] Нет OHLCV-данных от {exchange} для {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        logging.info(f"[Data] Получена форма данных от {exchange}: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"[Data] Ошибка при получении данных от {exchange}: {e}")
        return pd.DataFrame()


def fetch_data_from_exchanges(
        exchanges: List[str],
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500
) -> Dict[str, pd.DataFrame]:
    """
    Запрашивает данные по одному символу на нескольких биржах.
    Возвращает словарь {exchange: DataFrame}, пропуская биржи без пары или при ошибках.
    """
    results: Dict[str, pd.DataFrame] = {}
    for ex in exchanges:
        df = fetch_data_from_exchange(exchange=ex, symbol=symbol, interval=interval, limit=limit)
        if not df.empty:
            results[ex] = df
        else:
            logging.info(f"[Data] Пропущена биржа {ex} для {symbol} (нет данных).")
    return results
