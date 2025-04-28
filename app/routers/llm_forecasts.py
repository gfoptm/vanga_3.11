from typing import Optional, List

from fastapi import Query, Depends, APIRouter
from sqlalchemy.orm import Session

from app.database import get_db
from app.dbmodels import LLMForecast
from app.schemas import LLMForecastOut

router = APIRouter(prefix="/llm_forecasts", tags=["llm_forecasts"])


# 2. Эндпоинт для получения данных
@router.get("", response_model=List[LLMForecastOut])
def get_llm_forecasts(
        symbol: Optional[str] = Query(None, description="Фильтр по символу"),
        interval: Optional[str] = Query(None, description="Фильтр по интервалу"),
        exchange: Optional[str] = Query(None, description="Фильтр по бирже"),
        limit: int = Query(100, gt=0, le=1000, description="Максимум записей"),
        db: Session = Depends(get_db),
):
    q = db.query(LLMForecast)
    if symbol:
        q = q.filter(LLMForecast.symbol == symbol)
    if interval:
        q = q.filter(LLMForecast.interval == interval)
    if exchange:
        q = q.filter(LLMForecast.exchange == exchange)

    forecasts = (
        q.order_by(LLMForecast.forecast_time.desc())
        .limit(limit)
        .all()
    )
    return forecasts
