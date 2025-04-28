from typing import Any

import numpy as np
from fastapi import APIRouter, Query
from fastapi import HTTPException
from fastapi.responses import JSONResponse


from app.services.sde import estimate_parameters_mle
from app.utils.time import interval_to_timedelta
from fetch import fetch_data_from_exchange

router = APIRouter(prefix="/estimate_futures_params", tags=["estimate_futures_params"])


@router.post("", response_class=JSONResponse)
def estimate_futures_params(
        symbol: str = Query(...), interval: str = Query("1h"),
        exchange: str = Query("binance"), window: int = Query(60),
        epochs: int = Query(500), lr: float = Query(1e-3)
) -> Any:
    df = fetch_data_from_exchange(exchange, symbol, interval, limit=window + 1)
    if df.empty or len(df) < window + 1:
        raise HTTPException(404, "Недостаточно данных для оценки")
    prices = df["close"].values.astype(np.float32)
    dt_sec = interval_to_timedelta(interval).total_seconds()
    init = {
        "mu": 0., "kappa": 1., "theta": np.var(prices),
        "xi": 0.5, "rho": -0.5,
        "jump_intensity": 0.1, "jump_loc": 0., "jump_scale": 0.02
    }
    params = estimate_parameters_mle(prices, dt_sec, window, init, epochs, lr)
    return {
        "symbol": symbol, "exchange": exchange, "interval": interval,
        "estimated_params": params
    }
