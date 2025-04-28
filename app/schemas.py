from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class ForecastComparisonInput(BaseModel):
    symbol: str
    interval: str
    actual_close: Optional[float] = None
    exchange: str
    forecast_time: Optional[int] = None


class LLMForecastOut(BaseModel):
    id: int
    symbol: str
    interval: str
    exchange: str
    forecast_time: int
    signal: str
    confidence: float
    meta_features: dict
    created_at: int

    class Config:
        orm_mode = True


class CombinedForecastOut(BaseModel):
    symbol: str
    exchange: str
    interval: str

    forecast_price: Optional[float] = None
    forecast_signal: Optional[str] = None

    signal: Optional[str] = None

    fundamental_price: Optional[float] = None
    fundamental_signal: Optional[str] = None

    trend: Optional[str] = None
    trend_confidence: Optional[float] = None

    news_sentiment: Optional[float] = None

    llm_signal: Optional[str] = None
    llm_confidence: Optional[float] = None

    actual_open: Optional[float] = None
    actual_close: Optional[float] = None

    overall_correlation: Optional[float] = None

    # В БД вы сохраняете как строку 'YYYY-MM-DD HH:MM:SS'
    candle_time: str

    # UNIX-таймстамп целиком, без дробной части
    forecast_time: int

    class Config:
        orm_mode = True
