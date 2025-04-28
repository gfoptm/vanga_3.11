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

router = APIRouter(prefix="/candles", tags=["candles"])


@router.get("", response_class=JSONResponse)
def get_candles(
        symbol: str = Query(..., description="Торговая пара, например BTCUSDT"),
        interval: str = Query(..., description="Интервал свечи, например 1h, 15m"),
        exchange: str = Query(..., description="Биржа, например binance"),
        limit: int = Query(60, ge=1, le=1000, description="Количество свечей")
) -> Any:
    try:
        symbol = validate_symbol(symbol)
        interval = validate_interval(interval)
        exchange = validate_exchange(exchange)
        df = fetch_data_from_exchange(exchange, symbol, interval, limit=limit)
        if df.empty:
            return JSONResponse({"message": f"Нет данных для {symbol}@{interval} на {exchange}"}, status_code=404)
        candles = [{
            "time": int(row["timestamp"] / 1000),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "timestamp": int(row["timestamp"])
        } for _, row in df.iterrows()]
        return candles
    except Exception as e:
        logging.error(f"Ошибка в /candles: {e}")
        return JSONResponse({"message": "Ошибка получения данных свечей", "error": str(e)}, status_code=500)
