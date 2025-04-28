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

router = APIRouter(prefix="/live_predictions", tags=["live_predictions"])


@router.get("")
async def live_predictions() -> StreamingResponse:
    async def event_generator():
        while True:
            with SessionLocal() as db:
                preds = db.query(PredictionDB).order_by(PredictionDB.timestamp.desc()).limit(10).all()
                pred_list = [{
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
            data = json.dumps(pred_list)
            yield f"data: {data}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
