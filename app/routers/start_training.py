from typing import Any

from fastapi import Form, APIRouter
from starlette.responses import JSONResponse

from app.config import window_size
from app.state import models
from fetch import fetch_data_from_exchange
from train import train_model_for_symbol

router = APIRouter(prefix="/start_training", tags=["start_training"])


@router.post("", response_class=JSONResponse)
def start_training(
        symbol: str = Form("BTCUSDT"),
        interval: str = Form("1h"),
        exchange: str = Form("binance"),
        use_genetics: bool = Form(False)
) -> Any:
    df = fetch_data_from_exchange(exchange, symbol, interval, limit=window_size + 150)
    if df.empty:
        return JSONResponse({"message": f"Нет данных для {symbol}@{interval} на {exchange}"})
    model = train_model_for_symbol(df, symbol, exchange, use_genetics)
    model_key = f"{symbol}_{exchange}"
    if model:
        models[model_key] = model
        return JSONResponse({"message": f"Модель для {symbol}@{interval} на {exchange} успешно обучена"})
    else:
        return JSONResponse({"message": f"Ошибка обучения для {symbol}@{interval} на {exchange}"})
