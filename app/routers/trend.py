import datetime
import logging
import time
from typing import Optional, Dict, Any

from fastapi import Request, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import HTMLResponse

from app.config import get_exchange_client, INTERVAL_MAPPING
from app.database import get_db
from app.dbmodels import TrendForecastDB
from app.state import templates
from app.utils.time import get_interval_seconds

router = APIRouter(prefix="/trend_comparison_page", tags=["trend_comparison_page"])


# ----------------------------
# Эндпоинты для получения прогнозов и сравнения с фактическими данными
# ----------------------------


def fetch_candle_by_start_time(symbol: str, interval: str, candle_start: int, exchange: str) -> Optional[
    Dict[str, Any]]:
    """
    Получает данные свечи, начинающейся во время candle_start (UNIX-время) по заданному интервалу.
    Возвращает словарь с ключами: "open", "high", "low", "close", "volume".
    """
    try:
        start_ms = candle_start * 1000
        interval_sec = get_interval_seconds(interval)
        end_ms = start_ms + (interval_sec * 1000) - 1
        client = get_exchange_client(exchange)
        if not client:
            return None
        ccxt_timeframe = INTERVAL_MAPPING.get(interval, "1h")
        candles = client.fetch_ohlcv(symbol, timeframe=ccxt_timeframe, since=start_ms, limit=2)
        if not candles:
            return None
        for c in candles:
            ts = c[0]
            if start_ms <= ts <= end_ms:
                return {
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5])
                }
        last = candles[-1]
        return {
            "open": float(last[1]),
            "high": float(last[2]),
            "low": float(last[3]),
            "close": float(last[4]),
            "volume": float(last[5])
        }
    except Exception as e:
        logging.error(f"Ошибка при получении свечи для {symbol} на {exchange}: {e}")
        return None


@router.get("", response_class=HTMLResponse)
async def trend_comparison_page(request: Request, db: Session = Depends(get_db)) -> Any:
    """
    Формирует HTML-страницу с сравнением прогнозированного тренда (TrendForecastDB) с фактическими данными свечи.
    """
    try:
        now_ts = int(time.time())
        comparisons = []
        trend_forecasts = (
            db.query(TrendForecastDB)
            .filter(TrendForecastDB.forecast_time <= now_ts)
            .order_by(TrendForecastDB.timestamp.desc())
            .limit(100)
            .all()
        )

        for tf in trend_forecasts:
            interval_sec = get_interval_seconds(tf.interval)
            candle_start = tf.forecast_time - interval_sec
            candle_time_str = datetime.datetime.fromtimestamp(candle_start, tz=datetime.timezone.utc).strftime(
                '%Y-%m-%d %H:%M:%S')

            actual_candle = fetch_candle_by_start_time(tf.symbol, tf.interval, candle_start, tf.exchange)
            if actual_candle:
                actual_open = actual_candle["open"]
                actual_close = actual_candle["close"]
                actual_trend = "uptrend" if actual_close > actual_open else "downtrend"
            else:
                actual_open, actual_close, actual_trend = "—", "—", "—"

            status = "accurate" if actual_trend != "—" and actual_trend == tf.trend else (
                "inaccurate" if actual_trend != "—" else "ожидается")
            comparisons.append({
                "symbol": tf.symbol,
                "exchange": tf.exchange,
                "interval": tf.interval,
                "candle_time": candle_time_str,
                "predicted_trend": tf.trend,
                "predicted_confidence": tf.confidence,
                "actual_open": actual_open,
                "actual_close": actual_close,
                "actual_trend": actual_trend,
                "status": status,
            })

        comparisons = sorted(comparisons, key=lambda x: x["candle_time"], reverse=True)
        return templates.TemplateResponse("trend_comparison.html", {"request": request, "comparisons": comparisons})
    except Exception as e:
        logging.exception("Ошибка при формировании страницы сравнения тренда")
        return templates.TemplateResponse("trend_comparison.html",
                                          {"request": request, "comparisons": [], "error": str(e)})
