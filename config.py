import os
import logging
import datetime
import ccxt

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, BigInteger
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- ЛОГИРОВАНИЕ ---
LOG_FILE = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# --- БАЗА ДАННЫХ (PostgreSQL) ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://gleb:secret@localhost:5432/crypto"
)

# создаём движок без sqlite‑специфичных опций
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Возвращает сессию подключения к БД для FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- МОДЕЛИ SQLALCHEMY ---

class ForecastDB(Base):
    __tablename__ = "forecasts"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    interval = Column(String)
    signal = Column(String)
    confidence = Column(Float)
    price = Column(Float)
    volatility = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    exchange = Column(String)
    forecast_time = Column(Integer, nullable=False)  # метка времени будущей свечи (сек)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class SignalDB(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    interval = Column(String)
    signal = Column(String)
    confidence = Column(Float)
    price = Column(Float)
    volatility = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    exchange = Column(String)
    forecast_time = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class PredictionDB(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    interval = Column(String)
    forecast_time = Column(Integer, nullable=False)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)
    exchange = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class ForecastComparison(Base):
    __tablename__ = "forecast_comparisons"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    forecast_time = Column(Integer, nullable=False)
    forecast_close = Column(Float)
    actual_close = Column(Float)
    diff_percentage = Column(Float)
    status = Column(String)
    exchange = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class FundamentalForecastDB(Base):
    __tablename__ = "fundamental_forecasts"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    exchange = Column(String, index=True)
    forecast_time = Column(Integer, index=True)
    signal = Column(String)
    confidence = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class TrendForecastDB(Base):
    __tablename__ = "trend_forecasts"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    exchange = Column(String, index=True)
    interval = Column(String, index=True)
    forecast_time = Column(Integer, index=True)
    trend = Column(String)  # "uptrend" или "downtrend"
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)



class CombinedForecast(Base):
    __tablename__ = "combined_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    exchange = Column(String, nullable=False)
    interval = Column(String, nullable=False)

    # Прогноз цены и сигнала
    forecast_price = Column(Float, nullable=True)
    forecast_signal = Column(String, nullable=True)

    # Традиционный сигнал
    signal = Column(String, nullable=True)

    # Фундаментальный прогноз
    fundamental_price = Column(Float, nullable=True)
    fundamental_signal = Column(String, nullable=True)

    # Прогноз тренда
    trend = Column(String, nullable=True)
    trend_confidence = Column(Float, nullable=True)

    # Новостная тональность
    news_sentiment = Column(Float, nullable=True)

    # LLM‑прогноз
    llm_signal = Column(String, nullable=True)
    llm_confidence = Column(Float, nullable=True)

    # Фактические данные по свече
    actual_open = Column(Float, nullable=True)
    actual_close = Column(Float, nullable=True)

    # Общая метрика качества
    overall_correlation = Column(Float, nullable=True)

    # Время свечи и время сохранения
    candle_time = Column(String, nullable=True)
    forecast_time = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class NewsSentimentForecast(Base):
    __tablename__ = "news_sentiment_forecast"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    forecast_time = Column(Integer, index=True)
    sentiment_score = Column(Float)
    source = Column(String, default="aggregated")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class LLMForecast(Base):
    __tablename__ = "llm_forecasts"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    interval = Column(String, index=True, nullable=False)
    exchange = Column(String, index=True, nullable=False)
    forecast_time = Column(BigInteger, index=True, nullable=False)
    signal = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    meta_features = Column(JSONB, nullable=False)  # используем JSONB для PostgreSQL
    created_at = Column(BigInteger, nullable=False)


# создаём все таблицы (при первом запуске)
Base.metadata.create_all(bind=engine)


# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ ---
ALLOWED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
ALLOWED_EXCHANGES = ["binance", "bybit", "kraken"]
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

# --- API‑ключи и инициализация клиентов CCXT ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

EXCHANGE_CLIENTS = {
    "binance": ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "enableRateLimit": True
    }),
    "bybit": ccxt.bybit({
        "apiKey": BYBIT_API_KEY,
        "secret": BYBIT_API_SECRET,
        "enableRateLimit": True
    }),
    "kraken": ccxt.kraken({
        "apiKey": KRAKEN_API_KEY,
        "secret": KRAKEN_API_SECRET,
        "enableRateLimit": True
    }),
}


def get_exchange_client(exchange: str):
    """
    Возвращает клиент ccxt для указанной биржи.
    Если биржа не разрешена, возвращается клиент Binance по умолчанию.
    """
    exchange = exchange.lower()
    if exchange not in ALLOWED_EXCHANGES:
        logging.warning(f"Exchange {exchange} не разрешена, используется binance.")
        exchange = "binance"
    return EXCHANGE_CLIENTS.get(exchange)
