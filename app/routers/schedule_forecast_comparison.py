import datetime
import logging
import time
from typing import Optional, Any

from fastapi import Query, HTTPException, APIRouter
from fastapi.responses import JSONResponse

from app.database import SessionLocal
from app.dbmodels import ForecastDB
from app.services.predictions import compare_single_forecast_job
from app.state import scheduler
from fetch import validate_interval, validate_exchange, interval_to_timedelta

router = APIRouter(prefix="/schedule_forecast_comparison", tags=["schedule_forecast_comparison"])


@router.post("", response_class=JSONResponse)
def schedule_forecast_comparison(
        symbol: str = Query(..., description="Тикер, напр. BTCUSDT"),
        interval: str = Query("1h", description="Тайм‑фрейм (1h, 15m, …)"),
        exchange: str = Query("binance", description="Биржа"),
        forecast_time: Optional[int] = Query(None,
                                             description="UNIX‑время открытия свечи (если не задано — берется последний Forecast)")
) -> Any:
    interval = validate_interval(interval)
    exchange = validate_exchange(exchange)
    int_sec = int(interval_to_timedelta(interval).total_seconds())
    buffer_sec = 30

    with SessionLocal() as db:
        q = db.query(ForecastDB).filter(
            ForecastDB.symbol == symbol,
            ForecastDB.interval == interval,
            ForecastDB.exchange == exchange
        )
        if forecast_time is not None:
            q = q.filter(ForecastDB.forecast_time == forecast_time)
        else:
            q = q.order_by(ForecastDB.forecast_time.desc())
        forecast = q.first()

    if forecast is None:
        raise HTTPException(404, "Forecast not found — возможно, еще не создан")

    forecast_time = forecast.forecast_time
    run_ts = forecast_time + int_sec + buffer_sec
    run_dt = datetime.datetime.fromtimestamp(run_ts, tz=datetime.timezone.utc)

    if run_ts <= int(time.time()):
        compare_single_forecast_job(symbol, interval, exchange, forecast_time)
        return {"message": "Свеча уже закрыта — сравнение выполнено сразу."}

    job_id = f"future_cmp_{symbol}_{interval}_{exchange}_{forecast_time}"
    if scheduler.get_job(job_id):
        return {"message": "Job already scheduled", "job_id": job_id, "run_at": run_dt.isoformat()}

    scheduler.add_job(
        compare_single_forecast_job,
        trigger="date",
        run_date=run_dt,
        args=[symbol, interval, exchange, forecast_time],
        id=job_id
    )
    logging.info(f"[schedule_forecast_comparison] Job scheduled ({job_id}) at {run_dt}")
    return {"message": "Сравнение запланировано", "job_id": job_id, "run_at": run_dt.isoformat()}
