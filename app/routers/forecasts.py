import logging
from typing import Optional, Any

from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dbmodels import ForecastDB

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.get("", response_class=JSONResponse)
def get_forecasts(
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        interval: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        db: Session = Depends(get_db)
) -> Any:
    try:
        query = db.query(ForecastDB)
        if symbol:
            query = query.filter(ForecastDB.symbol == symbol)
        if exchange:
            query = query.filter(ForecastDB.exchange == exchange)
        if interval:
            query = query.filter(ForecastDB.interval == interval)
        if start_time:
            query = query.filter(ForecastDB.forecast_time >= start_time)
        if end_time:
            query = query.filter(ForecastDB.forecast_time <= end_time)
        forecasts = query.order_by(ForecastDB.timestamp.desc()).all()
        result = [{
            "id": f.id,
            "symbol": f.symbol,
            "exchange": f.exchange,
            "interval": f.interval,
            "forecast_time": f.forecast_time,
            "signal": f.signal,
            "confidence": f.confidence,
            "price": f.price,
            "volatility": f.volatility,
            "atr": f.atr,
            "volume": f.volume,
            "timestamp": f.timestamp.isoformat()
        } for f in forecasts]
        return result
    except Exception as e:
        logging.error(f"Ошибка получения прогнозов: {e}")
        return JSONResponse({"message": "Ошибка получения прогнозов", "error": str(e)}, status_code=500)
