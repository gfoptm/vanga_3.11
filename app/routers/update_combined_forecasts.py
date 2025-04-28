from typing import Optional, Any

from fastapi import Query, Depends, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.combined import update_and_store_combined_forecasts

router = APIRouter(prefix="/api/update_combined_forecasts", tags=["update_combined_forecasts"])


@router.post("", response_class=JSONResponse)
def update_combined_forecasts_endpoint(
        symbol: Optional[str] = Query(None, description="Фильтр по символу, напр. BTCUSDT"),
        exchange: Optional[str] = Query(None, description="Фильтр по бирже, напр. binance"),
        interval: Optional[str] = Query("1h", description="Тайм‑фрейм, напр. 1h"),
        include_old: bool = Query(False,
                                  description="Сохранять прошлые общие сравнения, если для текущего часа нет новых"),
        db: Session = Depends(get_db)
) -> Any:
    records = update_and_store_combined_forecasts(symbol, exchange, interval, db, include_old=include_old)
    return {"created_records": len(records)}
