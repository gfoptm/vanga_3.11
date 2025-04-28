import logging
from typing import Optional, Any

from fastapi import Query, Depends, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.dbmodels import CombinedForecast
from app.services.combined import update_and_store_combined_forecasts
from app.schemas import CombinedForecastOut
router = APIRouter(prefix="/api/combined_forecasts", tags=["api_combined_forecasts"])


@router.get("", response_model=List[CombinedForecastOut])
def get_stored_combined_forecasts(
    symbol: Optional[str] = Query(None),
    exchange: Optional[str] = Query(None),
    interval: Optional[str] = Query("1h"),
    limit: int = Query(100, ge=1, le=1000),
    include_old: bool = Query(False),
    db: Session = Depends(get_db)
) -> Any:
    # обновляем для текущего часа
    update_and_store_combined_forecasts(symbol, exchange, interval, db, include_old)

    # забираем из БД
    q = db.query(CombinedForecast)
    if symbol:
        q = q.filter(CombinedForecast.symbol == symbol)
    if exchange:
        q = q.filter(CombinedForecast.exchange == exchange)
    q = q.filter(CombinedForecast.interval == interval)
    q = q.order_by(CombinedForecast.forecast_time.desc()).limit(limit)

    return q.all()
