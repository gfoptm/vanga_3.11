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

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("", response_class=HTMLResponse)
async def signals_view(request: Request, db: Session = Depends(get_db)) -> Any:
    db_signals = db.query(SignalDB).order_by(SignalDB.timestamp.desc()).limit(200).all()
    return templates.TemplateResponse("signals.html", {"request": request, "signals": db_signals})
