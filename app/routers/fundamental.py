import logging
from typing import Optional, Any

from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app.database import get_db
from app.dbmodels import FundamentalForecastDB

router = APIRouter(prefix="/fundamental_forecasts", tags=["fundamental_forecasts"])


# Эндпоинт для получения сохранённых фундаментальных прогнозов
@router.get("", response_class=JSONResponse)
def get_fundamental_forecasts(
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: int = 100,
        db: Session = Depends(get_db)
) -> Any:
    try:
        query = db.query(FundamentalForecastDB)
        if symbol:
            query = query.filter(FundamentalForecastDB.symbol == symbol)
        if exchange:
            query = query.filter(FundamentalForecastDB.exchange == exchange)
        forecasts = query.order_by(FundamentalForecastDB.timestamp.desc()).limit(limit).all()
        result = [{
            "id": f.id,
            "symbol": f.symbol,
            "exchange": f.exchange,
            "forecast_time": f.forecast_time,
            "signal": f.signal,
            "confidence": f.confidence,
            "price": f.price,
            "timestamp": f.timestamp.isoformat()
        } for f in forecasts]
        return result
    except Exception as e:
        logging.error(f"[get_fundamental_forecasts] Ошибка: {e}")
        return JSONResponse({"message": "Ошибка получения фундаментальных прогнозов", "error": str(e)}, status_code=500)
