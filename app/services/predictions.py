import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
import datetime
import time
import random

from typing import Optional, Dict, Any, List

from app.config import window_size, ALLOWED_EXCHANGES, ALLOWED_SYMBOLS
from app.database import SessionLocal
from app.dbmodels import (
    SignalDB,
    ForecastComparison,
    TrendForecastDB,
    ForecastDB,
    PredictionDB
)
from app.services.lyapunov import compute_lyapunov_exponent
from app.state import scheduler, models, device, last_forecast_times
from app.utils.time import align_forecast_time, get_interval_seconds
from fetch import (
    validate_exchange,
    validate_symbol,
    validate_interval,
    interval_to_timedelta,
    fetch_data_from_exchange,
)
from train import feature_engineering


def calculate_status(signal_price: float, actual_close: float, signal: str) -> (str, float):
    """
    Вычисляет процентную разницу между фактической ценой и ценой сигнала, а также возвращает статус.
    """
    diff = abs(actual_close - signal_price) / max(signal_price, 1e-9)
    wrong_direction = (signal == "buy" and actual_close < signal_price) or (
        signal == "sell" and actual_close > signal_price
    )
    status = "accurate" if (not wrong_direction and diff < 0.003) else "inaccurate"
    return status, round(diff * 100, 4)


def fetch_actual_close_from_exchange(
    symbol: str,
    interval: str,
    candle_start: int,
    exchange: str
) -> Optional[float]:
    """
    Получает фактическую цену закрытия свечи за счёт fetch_data_from_exchange(limit=2).
    """
    exchange = validate_exchange(exchange)
    symbol = validate_symbol(symbol)
    interval = validate_interval(interval)

    start_ms = candle_start * 1000
    end_ms = start_ms + int(interval_to_timedelta(interval).total_seconds() * 1000) - 1

    try:
        df = fetch_data_from_exchange(exchange, symbol, interval, limit=2)
        if df.empty:
            logging.warning(f"[fetch_actual_close] Нет данных для {symbol}@{exchange} в {candle_start}")
            return None

        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)
        if mask.any():
            return float(df.loc[mask, "close"].iloc[0])
        return float(df["close"].iloc[0])

    except Exception as e:
        logging.error(f"[fetch_actual_close] Ошибка: {e}")
        return None


def save_signal(
    symbol: str,
    interval: str,
    exchange: str,
    forecast_time: int,
    signal_data: Dict[str, Any]
) -> None:
    """
    Сохраняет сигнал в БД, если ещё не сохранён для данного времени.
    """
    aligned = align_forecast_time(forecast_time, interval)
    with SessionLocal() as db:
        exists = db.query(SignalDB).filter_by(
            symbol=symbol,
            interval=interval,
            exchange=exchange,
            forecast_time=aligned
        ).first()
        if not exists:
            db.add(SignalDB(
                symbol=symbol,
                interval=interval,
                exchange=exchange,
                forecast_time=aligned,
                signal=signal_data["signal"],
                confidence=signal_data["confidence"],
                price=signal_data["price"],
                volatility=signal_data["volatility"],
                atr=signal_data["atr"],
                volume=signal_data["volume"]
            ))
            db.commit()
            logging.info(f"[save_signal] Signal saved for {symbol}/{exchange}/{interval} @ {aligned}")


def schedule_signal(
    symbol: str,
    interval: str,
    exchange: str,
    forecast_time: int,
    signal_data: Dict[str, Any]
) -> None:
    """
    Планирует save_signal за 10 минут до открытия свечи.
    """
    aligned = align_forecast_time(forecast_time, interval)
    job_id = f"signal_{symbol}_{interval}_{exchange}_{aligned}"
    if scheduler.get_job(job_id):
        return

    run_ts = aligned - 600  # 10 минут раньше
    run_dt = datetime.datetime.fromtimestamp(run_ts, tz=datetime.timezone.utc)
    now = datetime.datetime.now(tz=datetime.timezone.utc)

    if run_dt <= now:
        save_signal(symbol, interval, exchange, aligned, signal_data)
    else:
        scheduler.add_job(
            save_signal,
            trigger="date",
            run_date=run_dt,
            args=[symbol, interval, exchange, aligned, signal_data],
            id=job_id
        )
        logging.info(f"[schedule_signal] Scheduled {job_id} at {run_dt}")


def compute_atr(df: pd.DataFrame, window: int = 14, base_price: float = None) -> float:
    """
    Вычисляет ATR по свечам.
    """
    base_price = base_price or df["close"].iloc[-1]
    df["tr"] = df.apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["close"]),
            abs(r["low"] - r["close"])
        ), axis=1
    )
    if len(df) >= window:
        atr = df["tr"].rolling(window).mean().iloc[-1]
    else:
        atr = base_price * 0.01
    return max(atr, base_price * 0.005)


def compute_and_save_prediction(
    symbol: str,
    interval: str,
    exchange: str
) -> Dict[str, Any]:
    """
    Основная функция: делает фиксированный и live прогноз, сохраняет в БД, планирует сигнал.
    """
    df = fetch_data_from_exchange(exchange, symbol, interval, limit=window_size + 150)
    if df.empty or len(df) < window_size + 1:
        return {"signal": "waiting", "confidence": 0.0, "forecast": None, "predict": None, "features": {}}

    df_closed = df.iloc[:-1].copy()
    live_candle = df.iloc[-1]
    now_ts = int(time.time())
    sec = int(interval_to_timedelta(interval).total_seconds())
    candle_close_time = ((now_ts // sec) + 1) * sec

    try:
        # 1) Lyapunov
        series = df_closed["close"].values
        lyapunov_exp = compute_lyapunov_exponent(series, m=3, tau=1)
        lyap_norm = 1 / (1 + abs(lyapunov_exp))
        chaotic = lyapunov_exp > 0.5

        # 2) Признаки
        base_price = df_closed["close"].iloc[-1]
        df_features = feature_engineering(df.copy())

        req_feats = [
            "log_return",
            "volatility",
            "momentum",
            "zscore_close",
            "volume_delta",
            "price_range",
            "ma_ratio"
        ]
        if len(df_features) < window_size or not set(req_feats).issubset(df_features.columns):
            logging.error(f"[compute_prediction] Недостаточно признаков для {symbol}")
            return {"signal": "error", "confidence": 0.0, "forecast": None, "predict": None, "features": {}}

        # 3) Фиксированный прогноз
        model_key = f"{symbol}_{exchange}"
        model = models[model_key]
        model.eval()

        X_fc = df_features.iloc[-window_size - 1:-1].values.astype(np.float32)
        X_fc_t = torch.from_numpy(X_fc).unsqueeze(0).to(device)
        with torch.no_grad():
            out_fc = model(X_fc_t)
            probs_fc = F.softmax(out_fc, dim=1).cpu().numpy()[0]

        raw_conf_fc = float(probs_fc.max())
        sig_fc = "buy" if probs_fc.argmax() == 1 else "sell"
        scaled_conf_fc = 0.0 if chaotic else max(0.2, raw_conf_fc) * lyap_norm

        curr_vol = df_features.iloc[-2]["volatility"]
        atr = compute_atr(df_closed, base_price=base_price)
        vol_scale = max(0.001, min(curr_vol, 0.02)) * lyap_norm
        factor = vol_scale * scaled_conf_fc

        forecast_close = base_price * (1 + factor if sig_fc == "buy" else 1 - factor)
        forecast_high = max(base_price, forecast_close) + atr * random.uniform(0.3, 0.7)
        forecast_low = min(base_price, forecast_close) - atr * random.uniform(0.3, 0.7)
        avg_vol = df_closed["volume"].tail(20).mean()
        pred_vol = round(avg_vol * (1 + random.uniform(-0.2, 0.2) + scaled_conf_fc * 0.1), 2)

        forecast_candle = {
            "time": candle_close_time,
            "open": round(base_price, 2),
            "close": round(forecast_close, 2),
            "high": round(forecast_high, 2),
            "low": round(forecast_low, 2),
            "volume": pred_vol
        }

        # Сохраняем ForecastDB
        with SessionLocal() as db:
            exists = db.query(ForecastDB).filter_by(
                symbol=symbol,
                interval=interval,
                exchange=exchange,
                forecast_time=candle_close_time
            ).first()
            if not exists:
                db.add(ForecastDB(
                    symbol=symbol,
                    interval=interval,
                    exchange=exchange,
                    forecast_time=candle_close_time,
                    signal=sig_fc,
                    confidence=scaled_conf_fc,
                    price=round(forecast_close, 2),
                    volatility=round(curr_vol, 6),
                    atr=round(atr, 2),
                    volume=pred_vol
                ))
                db.commit()
                last_forecast_times[f"{symbol}_{exchange}"] = candle_close_time

        # 4) Live прогноз
        X_live = df_features.iloc[-window_size:].values.astype(np.float32)
        X_live_t = torch.from_numpy(X_live).unsqueeze(0).to(device)
        with torch.no_grad():
            out_live = model(X_live_t)
            probs_live = F.softmax(out_live, dim=1).cpu().numpy()[0]

        raw_conf_lv = float(probs_live.max())
        sig_lv = "buy" if probs_live.argmax() == 1 else "sell"
        scaled_conf_lv = 0.0 if chaotic else max(0.2, raw_conf_lv) * lyap_norm

        bp_live = live_candle["close"]
        vol_live = df_features.iloc[-1]["volatility"]
        atr_live = compute_atr(df, base_price=bp_live)
        vs_live = max(0.001, min(vol_live, 0.02)) * lyap_norm

        lc = bp_live * (1 + vs_live * scaled_conf_lv) if sig_lv == "buy" else bp_live * (1 - vs_live * scaled_conf_lv)
        hh = max(bp_live, lc) + atr_live * random.uniform(0.3, 0.7)
        ll = min(bp_live, lc) - atr_live * random.uniform(0.3, 0.7)
        av = df["volume"].tail(20).mean()
        pv = round(av * (1 + random.uniform(-0.2, 0.2) + scaled_conf_lv * 0.1), 2)

        live_pred = {
            "time": int(live_candle["timestamp"] / 1000),
            "open": round(bp_live, 2),
            "close": round(lc, 2),
            "high": round(hh, 2),
            "low": round(ll, 2),
            "volume": pv
        }

        # Сохраняем PredictionDB
        aligned_live = (int(time.time()) // 60) * 60
        with SessionLocal() as db:
            exists_live = db.query(PredictionDB).filter_by(
                symbol=symbol,
                interval=interval,
                exchange=exchange,
                forecast_time=aligned_live
            ).first()
            if not exists_live:
                db.add(PredictionDB(
                    symbol=symbol,
                    interval=interval,
                    exchange=exchange,
                    forecast_time=aligned_live,
                    open=round(bp_live, 2),
                    close=round(lc, 2),
                    high=round(hh, 2),
                    low=round(ll, 2),
                    volume=pv
                ))
                db.commit()

        # Планируем сигнал
        schedule_signal(symbol, interval, exchange, candle_close_time, {
            "signal": sig_lv,
            "confidence": scaled_conf_lv,
            "price": round(bp_live, 2),
            "volatility": round(vol_live, 6),
            "atr": round(atr_live, 2),
            "volume": pv
        })

        # Формируем итоговый набор признаков
        feats = df_features.iloc[-1][req_feats].round(6).to_dict()
        feats.update({"lyapunov": lyapunov_exp, "lyap_norm": lyap_norm})

        return {
            "signal": sig_fc,
            "confidence": scaled_conf_fc,
            "forecast": forecast_candle,
            "predict": live_pred,
            "features": feats
        }

    except Exception as e:
        logging.error(f"[compute_prediction] Ошибка для {symbol}@{exchange}: {e}")
        return {"signal": "error", "confidence": 0.0, "forecast": None, "predict": None, "features": {}}


def calculate_status_with_trend(
    signal_price: float,
    actual_close: float,
    signal: str,
    trend: Optional[str] = None,
    trend_confidence: Optional[float] = None
) -> (str, float):
    diff = abs(actual_close - signal_price) / max(signal_price, 1e-9)
    wrong_direction = (signal == "buy" and actual_close < signal_price) or (
        signal == "sell" and actual_close > signal_price
    )
    status = "accurate" if (not wrong_direction and diff < 0.003) else "inaccurate"

    if trend is not None and trend_confidence is not None and trend_confidence > 0.7:
        if (signal == "buy" and trend != "uptrend") or (signal == "sell" and trend != "downtrend"):
            status = "inaccurate"
    return status, round(diff * 100, 4)


def compare_forecasts_job() -> None:
    now_ts = int(time.time())
    cutoff = now_ts - 60

    with SessionLocal() as db:
        signals = db.query(SignalDB).filter(SignalDB.forecast_time <= cutoff).all()
        groups: Dict[tuple, List[SignalDB]] = {}
        for s in signals:
            key = (s.symbol, s.interval, s.exchange, s.forecast_time)
            groups.setdefault(key, []).append(s)

        for (symbol, interval, exchange, f_time) in groups:
            candle_start = f_time - get_interval_seconds(interval)
            if db.query(ForecastComparison).filter_by(
                symbol=symbol,
                exchange=exchange,
                forecast_time=candle_start
            ).first():
                continue

            selected = max(groups[(symbol, interval, exchange, f_time)], key=lambda s: s.timestamp)

            trend_time = candle_start + get_interval_seconds(interval)
            trend_rec = db.query(TrendForecastDB).filter_by(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                forecast_time=trend_time
            ).first()
            trend, tconf = (trend_rec.trend, trend_rec.confidence) if trend_rec else (None, None)

            actual_close = fetch_actual_close_from_exchange(symbol, interval, candle_start, exchange)
            if actual_close is None:
                continue

            status, diff_pct = calculate_status_with_trend(
                selected.price, actual_close, selected.signal, trend, tconf
            )
            db.add(ForecastComparison(
                symbol=symbol,
                exchange=exchange,
                forecast_time=candle_start,
                forecast_close=selected.price,
                actual_close=actual_close,
                diff_percentage=diff_pct,
                status=status
            ))
        try:
            db.commit()
        except Exception:
            db.rollback()
            logging.exception("[compare_forecasts_job] Commit failed")


def compare_single_forecast_job(
    symbol: str,
    interval: str,
    exchange: str,
    forecast_time: int
) -> None:
    try:
        with SessionLocal() as db:
            fc = db.query(ForecastDB).filter_by(
                symbol=symbol,
                interval=interval,
                exchange=exchange,
                forecast_time=forecast_time
            ).first()
            if not fc:
                return
            if db.query(ForecastComparison).filter_by(
                symbol=symbol,
                exchange=exchange,
                forecast_time=forecast_time
            ).first():
                return

            actual = fetch_actual_close_from_exchange(symbol, interval, forecast_time, exchange)
            if actual is None:
                return

            status, diff_pct = calculate_status(fc.price, actual, fc.signal)
            db.add(ForecastComparison(
                symbol=symbol,
                exchange=exchange,
                forecast_time=forecast_time,
                forecast_close=fc.price,
                actual_close=actual,
                diff_percentage=diff_pct,
                status=status
            ))
            db.commit()
    except Exception:
        logging.exception(f"[compare_single_forecast_job] Error for {symbol}@{forecast_time}")


# Планирование задач
scheduler.add_job(
    lambda: [
        compute_and_save_prediction(sym, "1h", exch)
        for exch in ALLOWED_EXCHANGES
        for sym in ALLOWED_SYMBOLS
    ],
    trigger="interval",
    seconds=60,
    id="update_predictions_job"
)
scheduler.add_job(
    compare_forecasts_job,
    trigger="interval",
    minutes=1,
    id="compare_forecasts_job"
)
