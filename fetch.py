import logging
import pandas as pd
import datetime
from typing import Dict, List, Union

from app.config import (
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
    exchange: Union[str, List[str]] = "all",
    symbol: Union[str, List[str]] = "all",
    interval: str = "1h",
    limit: int = 500
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Синхронно забирает OHLCV данные через ccxt.

    Параметры:
      - exchange: название биржи, список бирж или "all" для всех из ALLOWED_EXCHANGES
      - symbol: тикер, список тикеров или "all" для всех из ALLOWED_SYMBOLS
      - interval: временной интервал (по умолчанию "1h")
      - limit: число свечей

    Возвращает вложенный словарь {exchange: {symbol: DataFrame}}.
    Для каждой биржи и каждого символа, при ошибке или отсутствии данных,
    возвращает пустой DataFrame().
    """
    # Формируем список бирж
    if isinstance(exchange, str) and exchange.lower() == "all":
        exchanges = ALLOWED_EXCHANGES
    elif isinstance(exchange, list):
        exchanges = [validate_exchange(ex) for ex in exchange]
    else:
        exchanges = [validate_exchange(exchange)]

    # Формируем список символов
    if isinstance(symbol, str) and symbol.lower() == "all":
        symbols = ALLOWED_SYMBOLS
    elif isinstance(symbol, list):
        symbols = [sym for sym in symbol if sym in ALLOWED_SYMBOLS]
    else:
        symbols = [validate_symbol(symbol)]

    validated_interval = validate_interval(interval)
    ccxt_timeframe = INTERVAL_MAPPING[validated_interval]

    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ex in exchanges:
        results[ex] = {}
        client = get_exchange_client(ex)
        if not client:
            logging.error(f"[Data] Клиент для биржи {ex} не найден.")
            for sym in symbols:
                results[ex][sym] = pd.DataFrame()
            continue

        for sym in symbols:
            logging.info(f"[Data] fetch_data_from_exchange: {ex}, {sym}, {validated_interval}, limit={limit}")
            try:
                raw = client.fetch_ohlcv(sym, timeframe=ccxt_timeframe, limit=limit)
                if not raw:
                    logging.warning(f"[Data] Нет данных OHLCV от {ex} для {sym}")
                    results[ex][sym] = pd.DataFrame()
                    continue
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
                results[ex][sym] = df
            except Exception as e:
                logging.error(f"[Data] Ошибка при получении данных от {ex} для {sym}: {e}")
                results[ex][sym] = pd.DataFrame()

        try:
            client.close()
        except Exception:
            pass

    return results
