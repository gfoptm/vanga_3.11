from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dbmodels import FuturesForecastDB
from app.services.sde import (
    estimate_parameters_mle,
    simulate_heston_jump_diffusion
)
from app.utils.time import interval_to_timedelta
from fetch import fetch_data_from_exchange

router = APIRouter(
    prefix="/futures_signal",
    tags=["futures_signal"],
)


@router.get("/", response_class=JSONResponse)
def get_futures_signal(
        symbol: str = Query(
            ...,
            description="Торговая пара, например BTCUSDT"
        ),
        interval: str = Query(
            "1h",
            description="Интервал свечей, например 1h, 15m"
        ),
        exchange: str = Query(
            "binance",
            description="Название биржи"
        ),
        window: int = Query(
            60,
            ge=1,
            description="Число исторических свечей для калибровки"
        ),
        scenarios: int = Query(
            2000,
            ge=1,
            description="Число сценариев для Монте-Карло"
        ),
        db: Session = Depends(get_db),
) -> Any:
    # 1) Забираем данные
    df = fetch_data_from_exchange(exchange, symbol, interval, limit=window + 1)
    if df.empty or len(df) < window + 1:
        raise HTTPException(status_code=404, detail="Недостаточно данных для прогноза")

    prices = df["close"].values.astype(np.float32)
    dt_sec = interval_to_timedelta(interval).total_seconds()

    # 2) Калибруем параметры
    init = {
        "mu": 0.0,
        "kappa": 1.0,
        "theta": float(np.var(prices)),
        "xi": 0.5,
        "rho": -0.5,
        "jump_intensity": 0.1,
        "jump_loc": 0.0,
        "jump_scale": 0.02,
    }
    params = estimate_parameters_mle(prices, dt_sec, window, init, epochs=500, lr=1e-3)

    # 3) Симулируем сценарии
    sim = simulate_heston_jump_diffusion(
        S0=tf.constant(prices[0]),
        v0=tf.constant(np.var(prices)),
        **{k: tf.constant(v) for k, v in params.items()},
        dt=tf.constant(dt_sec, tf.float32),
        steps=window,
        paths=scenarios
    ).numpy()
    ends = sim[:, -1]

    # 4) Считаем метрики риска и сигнал
    start = prices[0]
    deltas = ends - start
    prob_up = float((deltas > 0).mean())
    var95 = float(np.percentile(deltas, 5))
    es95 = float(deltas[deltas <= var95].mean())
    skew = float(((deltas - deltas.mean()) ** 3).mean() / deltas.std() ** 3)
    kurt = float(((deltas - deltas.mean()) ** 4).mean() / deltas.std() ** 4 - 3)

    up_thr = np.quantile(deltas, 0.4)
    down_thr = np.quantile(deltas, 0.6)
    avg = deltas.mean()
    if prob_up > 0.6 and avg > up_thr:
        signal = "BUY"
    elif (1 - prob_up) > 0.6 and avg < down_thr:
        signal = "SELL"
    else:
        signal = "WAIT"

    # 5) Сохраняем в БД
    record = FuturesForecastDB(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        params=params,
        prob_up=prob_up,
        var_95=var95,
        es_95=es95,
        skew=skew,
        kurtosis=kurt,
        signal=signal,
        confidence=prob_up
    )
    db.add(record)
    db.commit()

    # 6) Возвращаем результат
    return {
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "params": params,
        "probability_up": prob_up,
        "VaR_95": var95,
        "ES_95": es95,
        "skewness": skew,
        "kurtosis": kurt,
        "signal": signal,
        "confidence": prob_up
    }
