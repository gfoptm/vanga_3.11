import logging
from typing import Optional, Any

from fastapi import Query, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app.database import get_db
from app.dbmodels import TrendForecastDB

router = APIRouter(prefix="/trend_forecasts", tags=["trend_forecasts"])


@router.get("", response_class=JSONResponse)
def get_trend_forecasts(
        symbol: Optional[str] = Query(None, description="Торговый символ, например BTCUSDT"),
        exchange: Optional[str] = Query(None, description="Биржа, например binance"),
        interval: Optional[str] = Query(None, description="Интервал свечи, например 1h"),
        limit: int = Query(100, ge=1, le=1000, description="Количество записей"),
        db: Session = Depends(get_db)
) -> Any:
    try:
        query = db.query(TrendForecastDB)
        if symbol:
            query = query.filter(TrendForecastDB.symbol == symbol)
        if exchange:
            query = query.filter(TrendForecastDB.exchange == exchange)
        if interval:
            query = query.filter(TrendForecastDB.interval == interval)
        forecasts = query.order_by(TrendForecastDB.timestamp.desc()).limit(limit).all()
        result = [{
            "id": fc.id,
            "symbol": fc.symbol,
            "exchange": fc.exchange,
            "interval": fc.interval,
            "forecast_time": fc.forecast_time,
            "trend": fc.trend,
            "confidence": fc.confidence,
            "timestamp": fc.timestamp.isoformat()
        } for fc in forecasts]
        return result
    except Exception as e:
        logging.error(f"Ошибка получения прогнозов: {e}")
        return JSONResponse({"message": "Ошибка получения прогнозов", "error": str(e)}, status_code=500)
