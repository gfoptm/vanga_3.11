import datetime
import logging
import time
from typing import Optional, Dict, Any

from fastapi import Request, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import HTMLResponse

from app.database import get_db
from app.dbmodels import TrendForecastDB
from app.state import templates
from app.utils.time import get_interval_seconds

from fetch import (
    validate_exchange,
    validate_symbol,
    validate_interval,
    interval_to_timedelta,
    fetch_data_from_exchange,
)

router = APIRouter(prefix="/trend_comparison_page", tags=["trend_comparison_page"])


def fetch_candle_by_start_time(
    symbol: str,
    interval: str,
    candle_start: int,
    exchange: str
) -> Optional[Dict[str, Any]]:
    """
    Получает данные свечи, начинающейся во время candle_start (UNIX-секунды) по заданному интервалу,
    через fetch_data_from_exchange(limit=2). Возвращает словарь с ключами:
    "open", "high", "low", "close", "volume", или None при ошибке/отсутствии данных.
    """
    # 1) Валидация входных параметров
    exchange = validate_exchange(exchange)
    symbol = validate_symbol(symbol)
    interval = validate_interval(interval)

    # 2) Переводим время начала в миллисекунды и рассчитываем конец интервала
    start_ms = candle_start * 1000
    end_ms = start_ms + int(interval_to_timedelta(interval).total_seconds() * 1000) - 1

    try:
        # 3) Берём две свечи, чтобы гарантированно захватить нужную
        df = fetch_data_from_exchange(exchange, symbol, interval, limit=2)
        if df.empty:
            logging.warning(f"[fetch_candle_by_start_time] Нет данных OHLCV для {symbol}@{exchange} с {candle_start}")
            return None

        # 4) Фильтруем DataFrame по таймстампам
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)
        if mask.any():
            row = df.loc[mask].iloc[0]
        else:
            # fallback: первая (старейшая) свеча из результата
            row = df.iloc[0]

        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"])
        }

    except Exception as e:
        logging.error(f"[fetch_candle_by_start_time] Ошибка при получении свечи для "
                      f"{symbol}@{exchange}: {e}")
        return None


@router.get("", response_class=HTMLResponse)
async def trend_comparison_page(request: Request, db: Session = Depends(get_db)) -> Any:
    """
    Формирует HTML-страницу с сравнением прогнозированного тренда (TrendForecastDB)
    с фактическими данными свечи.
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
            candle_time_str = datetime.datetime.fromtimestamp(
                candle_start,
                tz=datetime.timezone.utc
            ).strftime('%Y-%m-%d %H:%M:%S')

            actual_candle = fetch_candle_by_start_time(
                tf.symbol, tf.interval, candle_start, tf.exchange
            )
            if actual_candle:
                actual_open = actual_candle["open"]
                actual_close = actual_candle["close"]
                actual_trend = "uptrend" if actual_close > actual_open else "downtrend"
            else:
                actual_open = actual_close = actual_trend = "—"

            status = (
                "accurate" if actual_trend != "—" and actual_trend == tf.trend else
                "inaccurate" if actual_trend != "—" else
                "ожидается"
            )

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

        comparisons.sort(key=lambda x: x["candle_time"], reverse=True)
        return templates.TemplateResponse(
            "trend_comparison.html",
            {"request": request, "comparisons": comparisons}
        )
    except Exception as e:
        logging.exception("Ошибка при формировании страницы сравнения тренда")
        return templates.TemplateResponse(
            "trend_comparison.html",
            {"request": request, "comparisons": [], "error": str(e)}
        )
