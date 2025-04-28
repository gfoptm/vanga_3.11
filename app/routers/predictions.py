import logging
from typing import Optional, Any

from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dbmodels import PredictionDB

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_class=JSONResponse)
def get_predictions(
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        db: Session = Depends(get_db)
) -> Any:
    try:
        query = db.query(PredictionDB)
        if start_time:
            query = query.filter(PredictionDB.forecast_time >= start_time)
        if end_time:
            query = query.filter(PredictionDB.forecast_time <= end_time)
        if symbol:
            query = query.filter(PredictionDB.symbol == symbol)
        if exchange:
            query = query.filter(PredictionDB.exchange == exchange)
        preds = query.order_by(PredictionDB.timestamp.desc()).all()
        return [{
            "id": p.id,
            "symbol": p.symbol,
            "exchange": p.exchange,
            "interval": p.interval,
            "forecast_time": p.forecast_time,
            "open": p.open,
            "close": p.close,
            "high": p.high,
            "low": p.low,
            "volume": p.volume,
            "timestamp": p.timestamp.isoformat()
        } for p in preds]
    except Exception as e:
        logging.error(f"[get_predictions] Ошибка: {e}")
        return JSONResponse({"message": "Ошибка получения предсказаний", "error": str(e)}, status_code=500)


@router.get("/{prediction_id}", response_class=JSONResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)) -> Any:
    try:
        pred = db.query(PredictionDB).filter(PredictionDB.id == prediction_id).first()
        if not pred:
            return JSONResponse({"message": "Предсказание не найдено"}, status_code=404)
        return {
            "id": pred.id,
            "symbol": pred.symbol,
            "exchange": pred.exchange,
            "interval": pred.interval,
            "forecast_time": pred.forecast_time,
            "open": pred.open,
            "close": pred.close,
            "high": pred.high,
            "low": pred.low,
            "volume": pred.volume,
            "timestamp": pred.timestamp.isoformat()
        }
    except Exception as e:
        logging.error(f"[get_prediction] Ошибка: {e}")
        return JSONResponse({"message": "Ошибка получения предсказания", "error": str(e)}, status_code=500)
