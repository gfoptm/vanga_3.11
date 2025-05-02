import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
from app.config import get_exchange_client, INTERVAL_MAPPING, window_size, ALLOWED_EXCHANGES, ALLOWED_SYMBOLS
from app.database import SessionLocal
from app.dbmodels import SignalDB, ForecastComparison, TrendForecastDB, ForecastDB, PredictionDB

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
import datetime
import time
from typing import Optional, Dict, Any, List
import random

from train import feature_engineering


def calculate_status(signal_price: float, actual_close: float, signal: str) -> (str, float):
    """
    Вычисляет процентную разницу между фактической ценой и ценой сигнала, а также возвращает статус.
    """
    diff = abs(actual_close - signal_price) / max(signal_price, 1e-9)
    wrong_direction = (signal == "buy" and actual_close < signal_price) or (
            signal == "sell" and actual_close > signal_price)
    status = "accurate" if (not wrong_direction and diff < 0.003) else "inaccurate"
    return status, round(diff * 100, 4)


# Функции для работы с API биржи и сигналами

def fetch_actual_close_from_exchange(
    symbol: str,
    interval: str,
    candle_start: int,
    exchange: str
) -> Optional[float]:
    """
    Получает фактическую цену закрытия свечи, начинающейся в candle_start (сек):
    - Валидирует входные параметры.
    - Берёт через fetch_data_from_exchange 2 свечи, начиная с candle_start.
    - Ищет в полученном DataFrame строку с timestamp в [start_ms, end_ms].
    - Если не находит, возвращает первое значение close.
    """
    exchange = validate_exchange(exchange)
    symbol = validate_symbol(symbol)
    interval = validate_interval(interval)

    start_ms = candle_start * 1000
    end_ms = start_ms + int(interval_to_timedelta(interval).total_seconds() * 1000) - 1

    try:
        # limit=2, чтобы точно захватить нужную свечу и запасную
        df = fetch_data_from_exchange(exchange, symbol, interval, limit=2)
        if df.empty:
            logging.warning(f"[fetch_actual_close] Нет данных OHLCV для {symbol}@{exchange} с {candle_start}")
            return None

        # Фильтруем по таймстампам (в миллисекундах)
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)
        df_window = df.loc[mask]
        if not df_window.empty:
            return float(df_window.iloc[0]["close"])

        # fallback — берём первую строку (старейшую в ответе)
        return float(df.iloc[0]["close"])
    except Exception as e:
        logging.error(f"[fetch_actual_close] Ошибка: {e}")
        return None


def save_signal(symbol: str, interval: str, exchange: str, forecast_time: int, signal_data: Dict[str, Any]) -> None:
    """
    Сохраняет сигнал в БД, если сигнал для выровненного forecast_time ещё не записан.
    """
    aligned_time = align_forecast_time(forecast_time, interval)
    with SessionLocal() as db:
        existing_signal = db.query(SignalDB).filter(
            SignalDB.symbol == symbol,
            SignalDB.interval == interval,
            SignalDB.exchange == exchange,
            SignalDB.forecast_time == aligned_time
        ).first()
        if not existing_signal:
            new_signal = SignalDB(
                symbol=symbol,
                interval=interval,
                signal=signal_data["signal"],
                confidence=signal_data["confidence"],
                price=signal_data["price"],
                volatility=signal_data["volatility"],
                atr=signal_data["atr"],
                volume=signal_data["volume"],
                exchange=exchange,
                forecast_time=aligned_time
            )
            db.add(new_signal)
            db.commit()
            logging.info(
                f"[save_signal] Сигнал для {symbol} ({interval}, {exchange}) сохранён на время: {aligned_time}")


def schedule_signal(symbol: str, interval: str, exchange: str, forecast_time: int, signal_data: Dict[str, Any]) -> None:
    """
    Планирует сохранение сигнала за 10 минут до открытия свечи.
    Если время уже прошло, сигнал сохраняется сразу.
    """
    aligned_time = align_forecast_time(forecast_time, interval)
    job_id = f"signal_{symbol}_{interval}_{exchange}_{aligned_time}"
    if scheduler.get_job(job_id):
        return
    run_timestamp = aligned_time - 600  # 10 минут до начала свечи
    run_date = datetime.datetime.fromtimestamp(run_timestamp, tz=datetime.timezone.utc)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if run_date <= now:
        logging.info(f"[schedule_signal] Время {run_date} уже прошло, сохраняем сигнал сразу для {symbol}")
        save_signal(symbol, interval, exchange, aligned_time, signal_data)
    else:
        scheduler.add_job(
            save_signal,
            trigger="date",
            run_date=run_date,
            args=[symbol, interval, exchange, aligned_time, signal_data],
            id=job_id
        )
        logging.info(f"[schedule_signal] Сигнал для {symbol} запланирован на {run_date} (job_id={job_id}).")


def compute_atr(df: pd.DataFrame, window: int = 14, base_price: float = None) -> float:
    """
    Вычисляет средний истинный диапазон (ATR) по данным свечей.
    """
    base_price = base_price or df["close"].iloc[-1]
    # Вычисляем True Range для каждой свечи
    df["tr"] = df[["high", "low", "close"]].apply(
        lambda row: max(
            row["high"] - row["low"],
            abs(row["high"] - row["close"]),
            abs(row["low"] - row["close"])
        ),
        axis=1
    )
    if len(df) >= window:
        atr = df["tr"].rolling(window).mean().iloc[-1]
    else:
        atr = base_price * 0.01
    return max(atr, base_price * 0.005)


def compute_and_save_prediction(symbol: str, interval: str, exchange: str) -> Dict[str, Any]:
    """
    Вычисляет фиксированный (forecast) и динамический (live) прогнозы для заданной пары,
    сохраняет прогнозы в БД и планирует сигнал.
    Предсказания делаются через PyTorch‑модель.
    """
    # Получаем данные
    df = fetch_data_from_exchange(exchange, symbol, interval, limit=window_size + 150)
    if df.empty or len(df) < window_size + 1:
        return {"signal": "waiting", "confidence": 0.0, "forecast": None, "predict": None, "features": {}}

    # Разделяем закрытые свечи и текущую (live)
    df_closed = df.iloc[:-1].copy()
    live_candle = df.iloc[-1]

    # Расчет времени закрытия свечи
    now_ts = int(time.time())
    interval_sec = int(interval_to_timedelta(interval).total_seconds())
    candle_close_time = ((now_ts // interval_sec) + 1) * interval_sec

    try:
        # 1) Ляпунов
        series = df_closed['close'].values
        lyapunov_exp = compute_lyapunov_exponent(series, m=3, tau=1)
        lyap_norm = 1 / (1 + abs(lyapunov_exp))
        chaotic = lyapunov_exp > 0.5

        # 2) Базовая свеча и признаки
        base_price = df_closed.iloc[-1]['close']
        df_features = feature_engineering(df.copy())
        req_feats = ["log_return", "volatility", "momentum", "zscore_close", "volume_delta", "price_range", "ma_ratio"]
        if len(df_features) < window_size or not set(req_feats).issubset(df_features.columns):
            logging.error(f"[compute_prediction] Недостаточно признаков для {symbol}")
            return {"signal": "error", "confidence": 0.0, "forecast": None, "predict": None, "features": {}}

        model_key = f"{symbol}_{exchange}"
        model = models[model_key]
        model.eval()

        # 3) Фиксированный прогноз через PyTorch
        X_fc = df_features.iloc[-window_size - 1:-1].values.astype(np.float32)
        X_fc_t = torch.from_numpy(X_fc).unsqueeze(0).to(device)  # [1, window_size, feat_dim]
        with torch.no_grad():
            out_fc = model(X_fc_t)  # logits [1, num_classes]
            probs_fc = F.softmax(out_fc, dim=1).cpu().numpy()[0]
        raw_conf_fc = float(probs_fc.max())
        sig_fc = "buy" if probs_fc.argmax() == 1 else "sell"
        if chaotic:
            sig_fc, scaled_conf_fc = "waiting", 0.0
        else:
            scaled_conf_fc = max(0.2, raw_conf_fc) * lyap_norm

        # метрики и прогнозная свеча
        curr_vol = df_features.iloc[-2]["volatility"]
        atr = compute_atr(df_closed, window=14, base_price=base_price)
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
        key = f"{symbol}_{interval}_{exchange}"
        with SessionLocal() as db:
            exists = db.query(ForecastDB).filter_by(
                symbol=symbol, interval=interval, exchange=exchange, forecast_time=candle_close_time
            ).first()
            if not exists:
                rec = ForecastDB(
                    symbol=symbol, interval=interval, exchange=exchange,
                    signal=sig_fc, confidence=scaled_conf_fc, price=round(forecast_close, 2),
                    volatility=round(curr_vol, 6), atr=round(atr, 2), volume=pred_vol,
                    forecast_time=candle_close_time
                )
                db.add(rec)
                db.commit()
                last_forecast_times[key] = candle_close_time

        # 4) Динамический прогноз (live) через PyTorch
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
        atr_live = compute_atr(df, window=14, base_price=bp_live)
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
            ex = db.query(PredictionDB).filter_by(
                symbol=symbol, interval=interval, exchange=exchange, forecast_time=aligned_live
            ).first()
            if not ex:
                pred = PredictionDB(
                    symbol=symbol, interval=interval, exchange=exchange,
                    forecast_time=aligned_live,
                    open=round(bp_live, 2), close=round(lc, 2),
                    high=round(hh, 2), low=round(ll, 2), volume=pv
                )
                db.add(pred);
                db.commit()

        # Планируем сигнал
        schedule_signal(symbol, interval, exchange, candle_close_time, {
            "signal": sig_lv, "confidence": scaled_conf_lv,
            "price": round(bp_live, 2), "volatility": round(vol_live, 6),
            "atr": round(atr_live, 2), "volume": pv
        })

        # Формируем ответ
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
        logging.error(f"[compute_prediction] Ошибка для {symbol}: {e}")
        return {"signal": "error", "confidence": 0.0, "forecast": None, "predict": None, "features": {}}


def calculate_status_with_trend(signal_price: float, actual_close: float, signal: str,
                                trend: Optional[str] = None, trend_confidence: Optional[float] = None) -> (str, float):
    """
    Вычисляет процентное отклонение между фактической ценой и ценой сигнала,
    а также определяет статус с учетом базовой логики и информации о тренде.

    Если сигнал равен "buy", ожидается, что actual_close >= signal_price, а если "sell" — наоборот.
    Дополнительно, если прогноз тренда (trend) не соответствует направлению сигнала
    (например, сигнал "buy", а тренд не "uptrend") и уверенность в тренде (trend_confidence)
    выше 70%, итоговый статус устанавливается как "inaccurate".
    """
    diff = abs(actual_close - signal_price) / max(signal_price, 1e-9)
    wrong_direction = (signal == "buy" and actual_close < signal_price) or (
            signal == "sell" and actual_close > signal_price)
    base_status = "accurate" if (not wrong_direction and diff < 0.003) else "inaccurate"

    if trend is not None:
        # Если сигнал противоречит прогнозу тренда, и уверенность в тренде достаточно высока, статус считается некорректным.
        if (signal == "buy" and trend != "uptrend") or (signal == "sell" and trend != "downtrend"):
            if trend_confidence is not None and trend_confidence > 0.7:
                base_status = "inaccurate"
    return base_status, round(diff * 100, 4)


def compare_forecasts_job() -> None:
    """
    Периодическая задача: сравнение прогнозов с фактическими значениями.

    Сигналы в SignalDB сохраняются с использованием align_forecast_time,
    то есть, если сигнал получен в 5:50, он сохраняется с forecast_time = 6:00 (прогноз для свечи с 5:00 до 6:00).
    Для сравнения вычисляем начальное время свечи:
         candle_start = forecast_time - интервал
    и группируем сигналы по (symbol, interval, exchange, forecast_time).
    Из каждой группы выбирается самый последний сигнал, и производится сравнение с фактической ценой закрытия для свечи, начинающейся в candle_start.

    Теперь дополнительно извлекается прогноз тренда из TrendForecastDB для данной свечи,
    и при расчете статуса учитывается, соответствует ли прогноз тренда направлению сигнала.
    """
    now_ts = int(time.time())
    # Для гарантии, что свеча уже закрылась, используем буфер (например, 60 секунд)
    buffer = 60
    cutoff = now_ts - buffer

    with SessionLocal() as db:
        # Выбираем сигналы, для которых forecast_time уже наступило (т.е. свеча закрылась)
        signals = db.query(SignalDB).filter(SignalDB.forecast_time <= cutoff).all()

        # Группируем сигналы по forecast_time (так как они сохранены через align_forecast_time)
        groups: Dict[tuple, List[SignalDB]] = {}
        for s in signals:
            key = (s.symbol, s.interval, s.exchange, s.forecast_time)
            groups.setdefault(key, []).append(s)

        for (symbol, interval, exchange, f_time) in groups:
            # Вычисляем начальное время свечи как: candle_start = f_time - интервал
            interval_sec = get_interval_seconds(interval)
            candle_start = f_time - interval_sec

            # Если для данной свечи уже существует сравнение, пропускаем
            exists = db.query(ForecastComparison).filter_by(
                symbol=symbol,
                exchange=exchange,
                forecast_time=candle_start
            ).first()
            if exists:
                continue

            # Из группы сигналов выбираем самый последний сигнал по timestamp
            signal_list = groups[(symbol, interval, exchange, f_time)]
            selected_signal = max(signal_list, key=lambda s: s.timestamp)

            # Получаем прогноз тренда для данной свечи.
            # Предполагается, что в TrendForecastDB прогноз рассчитан для времени закрытия свечи,
            # т.е. trend_forecast_time = candle_start + длительность интервала.
            trend_forecast_time = candle_start + interval_sec
            trend_record = db.query(TrendForecastDB).filter(
                TrendForecastDB.symbol == symbol,
                TrendForecastDB.exchange == exchange,
                TrendForecastDB.interval == interval,
                TrendForecastDB.forecast_time == trend_forecast_time
            ).first()
            trend = trend_record.trend if trend_record else None
            trend_conf = trend_record.confidence if trend_record else None

            # Получаем фактическую цену закрытия свечи, начинающейся в candle_start
            actual_close = fetch_actual_close_from_exchange(symbol, interval, candle_start, exchange)
            if actual_close is None:
                logging.warning(
                    f"[compare_forecasts_job] Не удалось получить фактическое закрытие для {symbol} свечи, начинающейся в {candle_start}"
                )
                continue

            # Вычисляем статус с учетом тренда
            status, diff_pct = calculate_status_with_trend(
                selected_signal.price, actual_close, selected_signal.signal, trend, trend_conf
            )
            comparison = ForecastComparison(
                symbol=symbol,
                exchange=exchange,
                forecast_time=candle_start,  # время открытия свечи, например 5:00
                forecast_close=selected_signal.price,
                actual_close=actual_close,
                diff_percentage=diff_pct,
                status=status
            )
            db.add(comparison)
            try:
                db.commit()
                logging.info(
                    f"[compare_forecasts_job] {symbol} свеча, начинающаяся в {candle_start} → diff={diff_pct}% | status={status}"
                )
            except Exception as commit_exc:
                db.rollback()
                logging.exception(
                    f"[compare_forecasts_job] Ошибка коммита для {symbol} свечи, начинающейся в {candle_start}: {commit_exc}"
                )


def compare_single_forecast_job(symbol: str, interval: str, exchange: str, forecast_time: int) -> None:
    #Сравнивает один прогноз с фактическим значением.

    try:
        with SessionLocal() as db:
            forecast = db.query(ForecastDB).filter(
                ForecastDB.symbol == symbol,
                ForecastDB.interval == interval,
                ForecastDB.exchange == exchange,
                ForecastDB.forecast_time == forecast_time
            ).first()
            if not forecast:
                logging.warning(
                    f"[compare_single_forecast_job] Forecast не найден: {symbol} {interval} {exchange} @{forecast_time}")
                return
            if db.query(ForecastComparison).filter(
                    ForecastComparison.symbol == symbol,
                    ForecastComparison.exchange == exchange,
                    ForecastComparison.forecast_time == forecast_time
            ).first():
                logging.info(f"[compare_single_forecast_job] Уже сравнивалось: {symbol}@{forecast_time}")
                return
            actual_close = fetch_actual_close_from_exchange(symbol, interval, forecast_time, exchange)
            if actual_close is None:
                logging.warning(f"[compare_single_forecast_job] Нет фактического закрытия для {symbol}@{forecast_time}")
                return
            status, diff_pct = calculate_status(forecast.price, actual_close, forecast.signal)
            comparison = ForecastComparison(
                symbol=symbol,
                exchange=exchange,
                forecast_time=forecast_time,
                forecast_close=forecast.price,
                actual_close=actual_close,
                diff_percentage=diff_pct,
                status=status
            )
            db.add(comparison)
            db.commit()
            logging.info(f"[compare_single_forecast_job] {symbol}@{forecast_time}: diff={diff_pct}% | status={status}")
    except Exception as exc:
        logging.exception(f"[compare_single_forecast_job] Ошибка для {symbol}@{forecast_time}: {exc}")


scheduler.add_job(
    lambda: [compute_and_save_prediction(sym, "1h", exch) for exch in ALLOWED_EXCHANGES for sym in ALLOWED_SYMBOLS],
    trigger="interval",
    seconds=60,
    id="update_predictions_job"
)
scheduler.add_job(compare_forecasts_job, trigger="interval", minutes=1, id="compare_forecasts_job")
