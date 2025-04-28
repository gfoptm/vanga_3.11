from typing import Optional

from fastapi import Request, Query, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import HTMLResponse

from app.config import ALLOWED_SYMBOLS, ALLOWED_INTERVALS, ALLOWED_EXCHANGES
from app.database import get_db
from app.dbmodels import LLMForecast
from app.state import templates

router = APIRouter(prefix="/llm_forecasts_table", tags=["llm_forecasts_table"])


@router.get("", response_class=HTMLResponse)
def llm_forecasts_table(
        request: Request,
        symbol: Optional[str] = Query(None, description="Фильтр по символу"),
        interval: Optional[str] = Query(None, description="Фильтр по интервалу"),
        exchange: Optional[str] = Query(None, description="Фильтр по бирже"),
        limit: int = Query(200, gt=0, le=2000, description="Максимум записей"),
        db: Session = Depends(get_db),
):
    q = db.query(LLMForecast)
    if symbol:
        q = q.filter(LLMForecast.symbol == symbol)
    if interval:
        q = q.filter(LLMForecast.interval == interval)
    if exchange:
        q = q.filter(LLMForecast.exchange == exchange)

    forecasts = (
        q.order_by(LLMForecast.forecast_time.desc())
        .limit(limit)
        .all()
    )
    return templates.TemplateResponse(
        "llm_forecasts.html",
        {
            "request": request,
            "forecasts": forecasts,
            "filters": {"symbol": symbol, "interval": interval, "exchange": exchange, "limit": limit},
            "allowed_symbols": ALLOWED_SYMBOLS,
            "allowed_intervals": ALLOWED_INTERVALS,
            "allowed_exchanges": ALLOWED_EXCHANGES,
        }
    )
