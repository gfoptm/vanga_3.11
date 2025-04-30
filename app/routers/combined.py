from typing import Optional, Any

from fastapi import APIRouter, Request, Query, Depends
from sqlalchemy.orm import Session
from starlette.responses import HTMLResponse

from app.database import get_db
from app.dbmodels import CombinedForecast
from app.state import templates

router = APIRouter(prefix="/combined_forecasts_page", tags=["combined_forecasts_page"])


@router.get("", response_class=HTMLResponse)
def combined_forecasts_page(
        request: Request,
        symbol: Optional[str] = Query(None, description="Фильтр по символу"),
        exchange: Optional[str] = Query(None, description="Фильтр по бирже"),
        interval: Optional[str] = Query("1h", description="Тайм‑фрейм"),
        limit: int = Query(100, ge=1, le=1000),
        db: Session = Depends(get_db)
) -> Any:
    query = db.query(CombinedForecast)
    if symbol:
        query = query.filter(CombinedForecast.symbol == symbol)
    if exchange:
        query = query.filter(CombinedForecast.exchange == exchange)
    if interval:
        query = query.filter(CombinedForecast.interval == interval)
    query = query.order_by(CombinedForecast.forecast_time.desc())
    combined_data = query.limit(limit).all()

    return templates.TemplateResponse("combined_forecasts.html", {"request": request, "forecasts": combined_data})
