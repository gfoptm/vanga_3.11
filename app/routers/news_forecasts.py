from fastapi import Query, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app.database import get_db
from app.dbmodels import NewsSentimentForecast

router = APIRouter(prefix="/news_forecasts", tags=["news_forecasts"])

@router.get("", response_class=JSONResponse)
def get_news_forecasts(
        symbol: str = Query(None, description="Фильтр по символу, напр. BTCUSDT"),
        limit: int = Query(100, ge=1, le=1000),
        db: Session = Depends(get_db)
) -> list:
    """
    Эндпоинт для получения сохранённых новостных прогнозов.
    Позволяет фильтровать по символу.
    """
    query = db.query(NewsSentimentForecast)
    if symbol:
        query = query.filter(NewsSentimentForecast.symbol == symbol)
    forecasts = query.order_by(NewsSentimentForecast.timestamp.desc()).limit(limit).all()

    result = [{
        "id": f.id,
        "symbol": f.symbol,
        "forecast_time": f.forecast_time,
        "sentiment_score": f.sentiment_score,
        "timestamp": f.timestamp.isoformat()
    } for f in forecasts]
    return result