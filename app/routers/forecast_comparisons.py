import logging
from typing import Optional, Any

from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dbmodels import ForecastComparison, TrendForecastDB
from app.utils.time import get_interval_seconds

router = APIRouter(prefix="/forecast_comparisons", tags=["forecast_comparisons"])


@router.get("", response_class=JSONResponse)
def get_forecast_comparisons(
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: int = 100,
        db: Session = Depends(get_db)
) -> Any:
    try:
        q = db.query(ForecastComparison)
        if symbol:
            q = q.filter(ForecastComparison.symbol == symbol)
        if exchange:
            q = q.filter(ForecastComparison.exchange == exchange)
        comparisons = q.order_by(ForecastComparison.timestamp.desc()).limit(max(1, min(limit, 1000))).all()

        results = []
        for c in comparisons:
            # Если в модели ForecastComparison есть поле interval, используем его,
            # иначе полагаем, что прогноз проводится для интервала "1h".
            c_interval = getattr(c, "interval", "1h")
            interval_seconds = get_interval_seconds(c_interval)
            # В нашем сценарии ForecastComparison.forecast_time соответствует времени открытия свечи.
            # Прогноз тренда, как правило, вычисляется для закрытия той же свечи,
            # т.е. ожидаемое время закрытия = время открытия + длительность интервала.
            trend_forecast_time = c.forecast_time + interval_seconds

            trend_record = db.query(TrendForecastDB).filter(
                TrendForecastDB.symbol == c.symbol,
                TrendForecastDB.exchange == c.exchange,
                TrendForecastDB.interval == c_interval,
                TrendForecastDB.forecast_time == trend_forecast_time
            ).first()
            trend_value = trend_record.trend if trend_record else None
            trend_confidence = trend_record.confidence if trend_record else None

            results.append({
                "id": c.id,
                "symbol": c.symbol,
                "exchange": c.exchange,
                "interval": c_interval,
                "forecast_time": c.forecast_time,
                "forecast_close": c.forecast_close,
                "actual_close": c.actual_close,
                "diff_percentage": c.diff_percentage,
                "status": c.status,
                "timestamp": c.timestamp.isoformat(),
                "trend": trend_value,
                "trend_confidence": trend_confidence
            })
        return results
    except Exception as exc:
        logging.exception("[forecast_comparisons] Ошибка получения данных")
        return JSONResponse({"message": "Ошибка получения сравнений", "error": str(exc)}, status_code=500)
