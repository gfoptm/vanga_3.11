from typing import Any

from fastapi import Query, APIRouter
from starlette.responses import JSONResponse

from app.services.fundamental import compute_and_save_fundamental_prediction

router = APIRouter(prefix="/start_fundamental_prediction", tags=["start_fundamental_prediction"])


# Эндпоинт для запуска фундаментального прогнозирования
@router.post("", response_class=JSONResponse)
def start_fundamental_prediction(
        symbol: str = Query("BTCUSDT"),
        exchange: str = Query("binance")
) -> Any:
    result = compute_and_save_fundamental_prediction(symbol, exchange)
    if result.get("status") == "error":
        return JSONResponse({"message": result.get("message")}, status_code=500)
    return {"message": "Фундаментальный прогноз создан", "result": result}