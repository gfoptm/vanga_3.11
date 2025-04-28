# app/routers/futures_forecasts.py
from typing import Any, List
from fastapi import APIRouter, Depends, Request, Query, BackgroundTasks
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dbmodels import FuturesForecastDB
from app.state import templates
from app.services.sde import compute_and_save_futures_forecast

router = APIRouter(
    prefix="/futures_forecasts",
    tags=["futures_forecasts"],
)


@router.get("/", response_class=HTMLResponse)
def futures_forecasts_page(
        request: Request,
        background_tasks: BackgroundTasks,
        symbol: str = Query(None),
        exchange: str = Query(None),
        interval: str = Query("1h"),
        db: Session = Depends(get_db),
) -> Any:
    # если пришли параметры — запустим пересчёт в фоне
    if symbol and exchange:
        background_tasks.add_task(
            compute_and_save_futures_forecast,
            symbol.upper(),
            interval.lower(),
            exchange.lower()
        )

    recs = (
        db.query(FuturesForecastDB)
        .order_by(FuturesForecastDB.timestamp.desc())
        .limit(100)
        .all()
    )
    symbols = sorted({r.symbol for r in recs})
    exchanges = sorted({r.exchange for r in recs})
    intervals = sorted({r.interval for r in recs})

    return templates.TemplateResponse(
        "futures_forecasts.html",
        {
            "request": request,
            "forecasts": recs,
            "symbols": symbols,
            "exchanges": exchanges,
            "intervals": intervals,
        },
    )
