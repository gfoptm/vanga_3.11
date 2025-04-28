import datetime
import logging
import time
from typing import Dict, Any, List

from fastapi import Query, Depends, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dbmodels import ForecastDB
from app.services.predictions import fetch_actual_close_from_exchange, calculate_status
from app.utils.time import get_interval_seconds

router = APIRouter(prefix="/forecast_comparison_page_data", tags=["forecast_comparison_page_data"])


@router.get("", response_class=JSONResponse)
def forecast_cmp_data_api(
        symbol: str = Query(...),
        exchange: str = Query(...),
        interval: str = Query(...),
        db: Session = Depends(get_db)
) -> Any:
    try:
        now_ts = int(time.time())
        result: List[Dict[str, Any]] = []
        forecasts = (
            db.query(ForecastDB)
            .filter(
                ForecastDB.symbol == symbol,
                ForecastDB.exchange == exchange,
                ForecastDB.interval == interval
            )
            .order_by(ForecastDB.forecast_time.desc())
            .limit(200)
            .all()
        )

        for fc in forecasts:
            interval_sec = get_interval_seconds(fc.interval)
            candle_close_time = fc.forecast_time
            candle_open_time = fc.forecast_time - interval_sec
            fc_time_str = datetime.datetime.fromtimestamp(candle_open_time, tz=datetime.timezone.utc).strftime(
                '%Y-%m-%d %H:%M:%S')

            if now_ts >= candle_close_time:
                actual_close = fetch_actual_close_from_exchange(fc.symbol, fc.interval, candle_open_time, fc.exchange)
                if actual_close is not None:
                    status, diff_percentage = calculate_status(fc.price, actual_close, fc.signal)
                else:
                    actual_close, status, diff_percentage = "—", "—", "—"
            else:
                actual_close, status, diff_percentage = "—", "ожидается", "—"

            result.append({
                "symbol": fc.symbol,
                "exchange": fc.exchange,
                "forecast_time": fc_time_str,
                "forecast_close": fc.price,
                "actual_close": actual_close,
                "diff_percentage": f"{diff_percentage}" if diff_percentage != "—" else "—",
                "status": status,
                "forecast_signal": fc.signal,
                "confidence": f"{fc.confidence:.2f}",
                "forecast_timestamp": fc.timestamp.replace(tzinfo=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            })

        result = sorted(result, key=lambda x: x["forecast_time"], reverse=True)
        return JSONResponse(result)
    except Exception as e:
        logging.exception("[forecast_cmp_data_api] Ошибка получения данных")
        return JSONResponse({"error": str(e)}, status_code=500)
