import datetime
import logging
import time
from typing import Dict, Any, List

from fastapi import Request, Depends, APIRouter
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dbmodels import ForecastDB
from app.services.predictions import fetch_actual_close_from_exchange, calculate_status
from app.state import templates
from app.utils.time import get_interval_seconds

router = APIRouter(prefix="/forecast_comparison_page", tags=["forecast_comparison_page"])


@router.get("", response_class=HTMLResponse)
async def forecast_cmp_page(request: Request, db: Session = Depends(get_db)) -> Any:
    """
    Формирует HTML-страницу со сравнением прогнозов.

    Для каждого сохранённого прогноза (ForecastDB) считается, что прогноз относится к свече,
    открывающейся в (forecast_time - интервал) и закрывающейся в forecast_time.
    Сравнение фактической цены проводится только после закрытия свечи.
    """
    try:
        now_ts = int(time.time())
        forecasts = (
            db.query(ForecastDB)
            .filter(ForecastDB.forecast_time <= now_ts)
            .order_by(ForecastDB.forecast_time.desc())
            .limit(200)
            .all()
        )

        result: List[Dict[str, Any]] = []
        for fc in forecasts:
            interval_sec = get_interval_seconds(fc.interval)
            # Поскольку forecast_time – время закрытия свечи,
            # время открытия = forecast_time - интервал
            candle_close_time = fc.forecast_time
            candle_open_time = fc.forecast_time - interval_sec
            fc_time_str = datetime.datetime.fromtimestamp(candle_open_time, tz=datetime.timezone.utc).strftime(
                '%Y-%m-%d %H:%M:%S')

            if now_ts >= candle_close_time:
                # Передаём candle_open_time, чтобы получить закрытие свечи,
                # соответствующей данному интервалу [candle_open_time, candle_close_time]
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
                "diff_percentage": diff_percentage,
                "status": status,
                "forecast_signal": fc.signal,
                "confidence": f"{fc.confidence:.2f}",
                "forecast_timestamp": fc.timestamp.replace(tzinfo=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            })

        result = sorted(result, key=lambda x: x["forecast_time"], reverse=True)
        return templates.TemplateResponse("forecast_comparison.html", {"request": request, "comparisons": result})
    except Exception as e:
        logging.error(f"[forecast_cmp_page] Ошибка: {e}")
        return templates.TemplateResponse("forecast_comparison.html",
                                          {"request": request, "comparisons": [], "error": str(e)})
