import os

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, BigInteger
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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


# создаём все таблицы (при первом запуске)
Base.metadata.create_all(bind=engine)
