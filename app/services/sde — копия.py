import datetime
import logging
import time
from typing import Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from app.config import ALLOWED_SYMBOLS, ALLOWED_EXCHANGES, window_size
from app.database import SessionLocal
from app.dbmodels import FuturesForecastDB
from app.state import scheduler
from app.utils.time import get_interval_seconds
from fetch import fetch_data_from_exchange

tfd = tfp.distributions


# --- 1. Heston + Jump-Diffusion simulation with antithetic variates ---

def simulate_heston_jump_diffusion(
        S0: tf.Tensor, v0: tf.Tensor,
        mu: tf.Tensor, kappa: tf.Tensor, theta: tf.Tensor,
        xi: tf.Tensor, rho: tf.Tensor,
        jump_intensity: tf.Tensor, jump_loc: tf.Tensor, jump_scale: tf.Tensor,
        dt: tf.Tensor, steps: int, paths: int
) -> tf.Tensor:
    # Нормали [steps, paths, 2]
    normals = tf.random.normal([steps, paths, 2], dtype=tf.float32)
    Z1, Z2 = normals[..., 0], normals[..., 1]

    W1 = tf.sqrt(dt) * Z1
    W2 = rho * Z1 * tf.sqrt(dt) + tf.sqrt(1 - rho ** 2) * tf.sqrt(dt) * Z2

    # Антиизмеренные варианты
    W1 = tf.concat([W1, -W1], axis=1)
    W2 = tf.concat([W2, -W2], axis=1)
    n_paths = paths * 2

    # Прыжки
    jumps = tfd.LogNormal(loc=jump_loc, scale=jump_scale).sample([steps, n_paths])
    N_jumps = tfd.Poisson(rate=jump_intensity * dt).sample([steps, n_paths])

    S = tf.TensorArray(tf.float32, size=steps + 1)
    v = tf.TensorArray(tf.float32, size=steps + 1)
    S = S.write(0, tf.repeat(S0, n_paths))  # [n_paths]
    v = v.write(0, tf.repeat(v0, n_paths))

    def body(i, S_arr, v_arr):
        Si = S_arr.read(i)
        vi = v_arr.read(i)
        dv = kappa * (theta - vi) * dt + xi * tf.sqrt(tf.maximum(vi, 0.0)) * W2[i]
        vi1 = tf.maximum(vi + dv, 1e-6)
        drift = mu * Si * dt
        diffusion = tf.sqrt(tf.maximum(vi, 0.0)) * Si * W1[i]
        jump_term = Si * (tf.exp(jumps[i]) - 1.0) * tf.cast(N_jumps[i], tf.float32)
        Si1 = Si + drift + diffusion + jump_term
        return i + 1, S_arr.write(i + 1, Si1), v_arr.write(i + 1, vi1)

    _, S_final, _ = tf.while_loop(
        lambda i, *_: i < steps,
        body,
        (0, S, v),
        parallel_iterations=1
    )

    # S_final.stack() имеет форму [steps+1, n_paths]
    # транспонируем в [n_paths, steps+1]
    return tf.transpose(S_final.stack(), perm=[1, 0])


# --- 2. Калибровка параметров (гибрид MSE + NLL) ---
def estimate_parameters_mle(
    prices: np.ndarray, dt: float, steps: int,
    initial: dict, epochs: int, lr: float
) -> dict:
    # 1) создаём trainable-переменные
    mu      = tf.Variable(initial["mu"], dtype=tf.float32)
    kappa   = tf.Variable(initial["kappa"], dtype=tf.float32)
    theta   = tf.Variable(initial["theta"], dtype=tf.float32)
    xi      = tf.Variable(initial["xi"], dtype=tf.float32)
    rho     = tf.Variable(initial["rho"], dtype=tf.float32)
    lam     = tf.Variable(initial["jump_intensity"], dtype=tf.float32)
    jloc    = tf.Variable(initial["jump_loc"], dtype=tf.float32)
    jscale  = tf.Variable(initial["jump_scale"], dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 2) loss_fn без @tf.function
    def loss_fn():
        tf_prices = tf.constant(prices, dtype=tf.float32)
        last_price = tf_prices[-1]
        eps = tf.constant(1e-8, dtype=tf.float32)

        sim_paths = simulate_heston_jump_diffusion(
            S0=tf.constant(prices[0], dtype=tf.float32),
            v0=tf.constant(np.var(prices), dtype=tf.float32),
            mu=mu, kappa=kappa, theta=theta, xi=xi, rho=rho,
            jump_intensity=lam, jump_loc=jloc, jump_scale=jscale,
            dt=tf.constant(dt, tf.float32),
            steps=steps, paths=128
        )  # shape [n_paths, steps+1]

        sim_ends = sim_paths[:, -1]  # последний шаг

        # лог-MSE
        mse = tf.reduce_mean(tf.square(
            tf.math.log(sim_ends + eps) - tf.math.log(last_price + eps)
        ))

        # NLL
        dist = tfd.Normal(tf.reduce_mean(sim_ends), tf.math.reduce_std(sim_ends))
        nll = -tf.reduce_mean(dist.log_prob(last_price))

        return mse + nll

    # 3) сам цикл обучения
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, [mu, kappa, theta, xi, rho, lam, jloc, jscale])
        optimizer.apply_gradients(zip(grads, [mu, kappa, theta, xi, rho, lam, jloc, jscale]))

        if epoch % max(1, epochs // 5) == 0:
            logging.info(f"[Calibration] Epoch {epoch}/{epochs}: loss={loss:.6f}")

    return {
        "mu": float(mu.numpy()), "kappa": float(kappa.numpy()),
        "theta": float(theta.numpy()), "xi": float(xi.numpy()),
        "rho": float(rho.numpy()), "jump_intensity": float(lam.numpy()),
        "jump_loc": float(jloc.numpy()), "jump_scale": float(jscale.numpy()),
    }


# ----------------------------
# 1. Compute & save one futures forecast (исправлено)
# ----------------------------
def compute_and_save_futures_forecast(
    symbol: str,
    interval: str,
    exchange: str,
    window: int = window_size,
    scenarios: int = 2000,
) -> Dict[str, Any]:
    # 1) Скачиваем данные
    df = fetch_data_from_exchange(exchange, symbol, interval, limit=window + 1)
    if df.empty or len(df) < window + 1:
        return {"error": "insufficient data"}

    prices = df["close"].values.astype(np.float32)
    dt_sec = get_interval_seconds(interval)

    # 2) Начальные параметры
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

    # 3) Калибровка (если упадёт — останемся на init)
    try:
        params = estimate_parameters_mle(prices, dt_sec, window, init, epochs=500, lr=1e-3)
    except Exception as e:
        logging.error(f"[FuturesJob] Calibration failed for {symbol}@{interval}/{exchange}: {e}")
        params = init

    # 4) Симуляция и сбор метрик
    sim = simulate_heston_jump_diffusion(
        S0=tf.constant(prices[0], tf.float32),
        v0=tf.constant(np.var(prices), tf.float32),
        **{k: tf.constant(v, tf.float32) for k, v in params.items()},
        dt=tf.constant(dt_sec, tf.float32),
        steps=window,
        paths=scenarios
    ).numpy()
    ends = sim[:, -1]

    start = prices[0]
    deltas = ends - start
    prob_up = float((deltas > 0).mean())
    var95 = float(np.percentile(deltas, 5))
    es95 = float(deltas[deltas <= var95].mean())
    avg = deltas.mean()
    up_thr = np.quantile(deltas, 0.4)
    down_thr = np.quantile(deltas, 0.6)

    if prob_up > 0.6 and avg > up_thr:
        signal = "BUY"
    elif (1 - prob_up) > 0.6 and avg < down_thr:
        signal = "SELL"
    else:
        signal = "WAIT"

    # 5) Расчёт времени следующего прогноза
    now_ts = int(time.time())
    next_start = ((now_ts // dt_sec) + 1) * dt_sec
    next_dt = datetime.datetime.fromtimestamp(next_start, tz=datetime.timezone.utc)

    # 6) Сохраняем в БД (не дублируя)
    with SessionLocal() as db:
        existing = (
            db.query(FuturesForecastDB)
              .filter(
                  FuturesForecastDB.symbol   == symbol,
                  FuturesForecastDB.exchange == exchange,
                  FuturesForecastDB.interval == interval,
                  FuturesForecastDB.timestamp== next_dt
              )
              .first()
        )

        if not existing:
            rec = FuturesForecastDB(
                symbol     = symbol,
                exchange   = exchange,
                interval   = interval,
                timestamp  = next_dt,      # <--- здесь реальное поле модели
                params     = params,
                prob_up    = prob_up,
                var_95     = var95,
                es_95      = es95,
                signal     = signal,
                confidence = prob_up
            )
            db.add(rec)
            db.commit()
            logging.info(f"[FuturesJob] Saved forecast for {symbol}@{interval}/{exchange}: {signal}")
        else:
            logging.debug(f"[FuturesJob] Forecast already exists for {symbol}@{interval} at {next_dt}")

    return {
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "forecast_time": next_start,
        "signal": signal,
        "confidence": prob_up,
    }


# ----------------------------
# 2. Schedule one job for a single symbol/interval/exchange
# ----------------------------
def schedule_futures_forecast(symbol: str, interval: str, exchange: str) -> None:
    """
    Schedule compute_and_save_futures_forecast to run
    at the start of the next interval (minute 0 of each hour, etc.).
    """
    now_ts = int(time.time())
    interval_sec = get_interval_seconds(interval)
    next_start = ((now_ts // interval_sec) + 1) * interval_sec
    run_date = datetime.datetime.fromtimestamp(next_start, tz=datetime.timezone.utc)
    job_id = f"futures_{symbol}_{interval}_{exchange}_{next_start}"

    if scheduler.get_job(job_id):
        return

    scheduler.add_job(
        compute_and_save_futures_forecast,
        trigger="date",
        run_date=run_date,
        args=[symbol, interval, exchange],
        id=job_id
    )
    logging.info(f"[Scheduler] Scheduled futures forecast for {symbol}@{interval}/{exchange} at {run_date}")


# ----------------------------
# 3. Schedule all symbols/exchanges every minute
# ----------------------------
def schedule_all_futures_forecasts() -> None:
    for symbol in ALLOWED_SYMBOLS:
        for exchange in ALLOWED_EXCHANGES:
            schedule_futures_forecast(symbol, "1h", exchange)


# 4. Kick off the recurring scheduler
scheduler.add_job(
    schedule_all_futures_forecasts,
    trigger="interval",
    seconds=60,
    id="schedule_futures_all_job"
)
