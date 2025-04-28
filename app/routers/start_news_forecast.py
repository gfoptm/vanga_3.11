import time

from fastapi import Query, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app.database import get_db
from app.dbmodels import NewsSentimentForecast
from app.services.news import fetch_enhanced_news_sentiment

router = APIRouter(prefix="/start_news_forecast", tags=["start_news_forecast"])


@router.post("", response_class=JSONResponse)
def start_news_forecast(
        symbol: str = Query("BTCUSDT"),
        forecast_time: int = Query(None),
        db: Session = Depends(get_db)
) -> dict:
    """
    Вычисляет прогноз новостной тональности для заданного символа с расширенным анализом.
    Если forecast_time не передан, используется ближайшее округление (начало следующего часа).
    Результат сохраняется в БД (NewsSentimentForecast).
    """
    now_ts = int(time.time())
    if forecast_time is None:
        interval_sec = 3600  # прогноз на ближайший час
        forecast_time = ((now_ts // interval_sec) + 1) * interval_sec

    sentiment = fetch_enhanced_news_sentiment(symbol)

    # Обновление существующей записи или создание новой
    existing = db.query(NewsSentimentForecast).filter(
        NewsSentimentForecast.symbol == symbol,
        NewsSentimentForecast.forecast_time == forecast_time
    ).first()
    if existing:
        existing.sentiment_score = sentiment
        db.commit()
        db.refresh(existing)
        record = existing
    else:
        record = NewsSentimentForecast(
            symbol=symbol,
            forecast_time=forecast_time,
            sentiment_score=sentiment
        )
        db.add(record)
        db.commit()
        db.refresh(record)

    return {
        "message": "Новостной прогноз создан",
        "result": {
            "symbol": record.symbol,
            "forecast_time": record.forecast_time,
            "sentiment_score": record.sentiment_score,
            "timestamp": record.timestamp.isoformat()
        }
    }
