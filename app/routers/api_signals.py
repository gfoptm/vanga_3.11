from starlette.responses import JSONResponse, HTMLResponse
import logging
from app.config import ALLOWED_SYMBOLS
from app.database import get_db, SessionLocal
from app.dbmodels import ForecastDB, ForecastComparison, TrendForecastDB, PredictionDB, SignalDB

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Query, Depends, HTTPException, Form, APIRouter
import datetime
import time
from app.services.predictions import compute_and_save_prediction, compare_single_forecast_job, \
    fetch_actual_close_from_exchange, calculate_status
from app.state import scheduler, templates
from app.utils.time import get_interval_seconds
from fetch import validate_interval, validate_exchange, validate_symbol, fetch_data_from_exchange, interval_to_timedelta
from sqlalchemy.orm import Session
import asyncio
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import json

router = APIRouter(prefix="/api/signals", tags=["api_signals"])


@router.get("", response_class=JSONResponse)
def get_api_signals(db: Session = Depends(get_db)) -> Any:
    try:
        signals_list = db.query(SignalDB).order_by(SignalDB.timestamp.desc()).all()
        return [{
            "id": s.id,
            "timestamp": s.timestamp.isoformat(),
            "symbol": s.symbol,
            "interval": s.interval,
            "signal": s.signal,
            "confidence": s.confidence,
            "price": s.price,
            "volatility": s.volatility,
            "atr": s.atr,
            "volume": s.volume,
            "exchange": s.exchange,
            "forecast_time": s.forecast_time,
        } for s in signals_list]
    except Exception as e:
        logging.error(f"Ошибка получения сигналов: {e}")
        return JSONResponse({"message": "Ошибка получения сигналов", "error": str(e)}, status_code=500)
