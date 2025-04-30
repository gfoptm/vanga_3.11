import datetime
import logging
import time
from typing import Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import minimize
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
    normals = tf.random.normal([steps, paths, 2], dtype=tf.float32)
    Z1, Z2 = normals[..., 0], normals[..., 1]

    W1 = tf.sqrt(dt) * Z1
    W2 = rho * Z1 * tf.sqrt(dt) + tf.sqrt(1 - rho ** 2) * tf.sqrt(dt) * Z2

    W1 = tf.concat([W1, -W1], axis=1)
    W2 = tf.concat([W2, -W2], axis=1)
    n_paths = paths * 2

    jumps = tfd.LogNormal(loc=jump_loc, scale=jump_scale).sample([steps, n_paths])
    N_jumps = tfd.Poisson(rate=jump_intensity * dt).sample([steps, n_paths])

    S = tf.TensorArray(tf.float32, size=steps + 1)
    v = tf.TensorArray(tf.float32, size=steps + 1)
    S = S.write(0, tf.repeat(S0, n_paths))
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

    return tf.transpose(S_final.stack(), perm=[1, 0])

# --- 2. Калибровка параметров через Nelder-Mead ---
def estimate_parameters_mle(
    prices: np.ndarray,
    dt: float,
    steps: int,
    initial: dict,
    epochs: int,
    lr: float = None
) -> dict:
    # Используем дисперсию лог-доходностей вместо ценовой
    eps = 1e-8
    log_returns = np.diff(np.log(prices + eps))
    var0 = float(np.var(log_returns)) if log_returns.size > 0 else float(np.var(prices))
    last_price = float(prices[-1])
    dt_norm = dt

    it = {"n": 0}

    def objective(x: np.ndarray) -> float:
        mu_np, kappa_np, theta_np, xi_np, rho_np, lam_np, jloc_np, jscale_np = x
        try:
            sim = simulate_heston_jump_diffusion(
                S0=tf.constant(prices[0], tf.float32),
                v0=tf.constant(var0, tf.float32),
                mu=tf.constant(mu_np, tf.float32),
                kappa=tf.constant(kappa_np, tf.float32),
                theta=tf.constant(theta_np, tf.float32),
                xi=tf.constant(xi_np, tf.float32),
                rho=tf.constant(rho_np, tf.float32),
                jump_intensity=tf.constant(lam_np, tf.float32),
                jump_loc=tf.constant(jloc_np, tf.float32),
                jump_scale=tf.constant(jscale_np, tf.float32),
                dt=tf.constant(dt_norm, tf.float32),
                steps=steps,
                paths=128
            ).numpy()
        except Exception as e:
            logging.warning(f"[NM] Simulation error: {e}")
            return 1e8

        ends = sim[:, -1].astype(np.float64)
        if not np.all(np.isfinite(ends)):
            return 1e8

        mse = np.mean((np.log(ends + eps) - np.log(last_price + eps))**2)
        mu_end, sigma_end = float(np.mean(ends)), float(np.std(ends))
        if sigma_end <= eps:
            return 1e8

        logp = -0.5 * (((last_price - mu_end)**2 / sigma_end**2)
                       + np.log(2 * np.pi * sigma_end**2))
        nll = -logp
        loss = mse + nll
        return float(loss) if np.isfinite(loss) else 1e8

    def nm_callback(xk: np.ndarray):
        it["n"] += 1
        curr = objective(xk)
        logging.info(f"[NM] итерация {it['n']}/{epochs}, loss={curr:.6f}")

    x0 = np.array([
        initial["mu"], initial["kappa"], initial["theta"],
        initial["xi"], initial["rho"],
        initial["jump_intensity"], initial["jump_loc"], initial["jump_scale"]
    ], dtype=np.float64)

    res = minimize(
        objective, x0,
        method="Nelder-Mead",
        callback=nm_callback,
        options={"maxiter": epochs, "xatol": 1e-4, "fatol": 1e-4, "disp": True}
    )

    logging.info(
        f"[Calibration] Nelder–Mead завершился: success={res.success}, итераций={res.nit}, final loss={res.fun:.6f}"
    )

    if not res.success or not np.isfinite(res.fun):
        logging.warning("[Calibration] Неудачная оптимизация, возвращаем начальные параметры")
        return initial.copy()

    mu_o, kappa_o, theta_o, xi_o, rho_o, lam_o, jloc_o, jscale_o = res.x
    return {
        "mu": float(mu_o), "kappa": float(kappa_o), "theta": float(theta_o),
        "xi": float(xi_o), "rho": float(rho_o),
        "jump_intensity": float(lam_o), "jump_loc": float(jloc_o), "jump_scale": float(jscale_o)
    }

# ----------------------------
# 3. Compute & save one futures forecast (с исправлениями)
# ----------------------------
def compute_and_save_futures_forecast(
    symbol: str,
    interval: str,
    exchange: str,
    window: int = window_size,
    scenarios: int = 2000,
) -> Dict[str, Any]:
    df = fetch_data_from_exchange(exchange, symbol, interval, limit=window + 1)
    if df.empty or len(df) < window + 1:
        logging.error(f"[FuturesJob] Insufficient data for {symbol}@{interval}/{exchange}")
        return {"error": "insufficient data"}

    prices = df["close"].values.astype(np.float32)
    eps = 1e-8
    log_returns = np.diff(np.log(prices + eps))
    var0 = float(np.var(log_returns)) if log_returns.size > 0 else float(np.var(prices))
    dt_norm = 1.0

    init = {
        "mu": 0.0, "kappa": 1.0, "theta": var0,
        "xi": 0.5, "rho": -0.5,
        "jump_intensity": 0.1, "jump_loc": 0.0, "jump_scale": 0.02
    }

    try:
        params = estimate_parameters_mle(prices, dt_norm, window, init, epochs=100)
    except Exception as e:
        logging.error(f"[FuturesJob] Calibration failed for {symbol}@{interval}/{exchange}: {e}")
        params = init.copy()

    logging.info(f"[FuturesJob] Using params: {params}")
    sim = simulate_heston_jump_diffusion(
        S0=tf.constant(prices[0], tf.float32),
        v0=tf.constant(var0, tf.float32),
        **{k: tf.constant(v, tf.float32) for k, v in params.items()},
        dt=tf.constant(dt_norm, tf.float32),
        steps=window,
        paths=scenarios
    ).numpy()

    ends = sim[:, -1]
    finite = ends[np.isfinite(ends)]
    logging.info(f"[Debug] total_paths={len(ends)}, finite_paths={len(finite)}")
    if len(finite) == 0:
        logging.error(f"[FuturesJob] All simulation paths invalid for {symbol}@{interval}/{exchange}")
        return {"error": "all paths invalid"}

    returns = (finite - float(prices[0])) / (float(prices[0]) + eps)
    std = float(np.std(returns, ddof=0))
    if std < eps:
        logging.error(f"[FuturesJob] Zero volatility for {symbol}@{interval}/{exchange}")
        return {"error": "zero volatility"}

    prob_up = float((returns > 0).mean())
    var95 = float(np.percentile(returns, 5))
    es95 = float(np.mean(returns[returns <= var95]))
    avg = float(np.mean(returns))
    skew = float(np.mean(((returns - avg) / std) ** 3))
    kurtosis = float(np.mean(((returns - avg) / std) ** 4) - 3.0)

    up_thr, down_thr = np.quantile(returns, [0.6, 0.4])
    if prob_up > 0.6 and avg > up_thr:
        signal = "BUY"
    elif (1 - prob_up) > 0.6 and avg < down_thr:
        signal = "SELL"
    else:
        signal = "WAIT"

    now_ts = int(time.time())
    interval_sec = int(get_interval_seconds(interval))
    next_start = ((now_ts // interval_sec) + 1) * interval_sec
    next_dt = datetime.datetime.fromtimestamp(next_start, tz=datetime.timezone.utc)

    with SessionLocal() as db:
        existing = db.query(FuturesForecastDB).filter(
            FuturesForecastDB.symbol == symbol,
            FuturesForecastDB.exchange == exchange,
            FuturesForecastDB.interval == interval,
            FuturesForecastDB.timestamp == next_dt
        ).first()
        if not existing:
            rec = FuturesForecastDB(
                symbol=symbol, exchange=exchange, interval=interval,
                timestamp=next_dt, params=params,
                prob_up=prob_up, var_95=var95, es_95=es95,
                skew=skew, kurtosis=kurtosis, signal=signal,
                confidence=prob_up
            )
            db.add(rec)
            db.commit()
            logging.info(f"[FuturesJob] Saved forecast: {symbol}@{interval}/{exchange} -> {signal}")
        else:
            logging.debug(f"[FuturesJob] Forecast exists for {symbol}@{interval}@ {next_dt}")

    return {"symbol": symbol, "exchange": exchange, "interval": interval,
            "forecast_time": next_start, "signal": signal, "confidence": prob_up}


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
