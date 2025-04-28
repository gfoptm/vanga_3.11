
import pandas as pd

import logging
from app.config import get_exchange_client, INTERVAL_MAPPING, window_size, ALLOWED_SYMBOLS, ALLOWED_EXCHANGES
from app.database import SessionLocal
from app.dbmodels import SignalDB, ForecastComparison, TrendForecastDB, ForecastDB, PredictionDB

from app.services.lyapunov import compute_lyapunov_exponent
from app.state import scheduler
from app.utils.time import align_forecast_time, get_interval_seconds
from fetch import interval_to_timedelta, fetch_data_from_exchange
import datetime
import time
from typing import Optional, Dict, Any, List
import random

from train import feature_engineering


# ----------------------------
# 1. Расчёт технических индикаторов
# ----------------------------
def compute_indicators(df: pd.DataFrame, params: Dict[str, float]) -> Dict[str, Any]:
    """
    Вычисляет основные индикаторы (SMA, EMA, MACD, RSI) и волатильность.
    Возвращает сигналы и меры уверенности, а также дополнительные детали.
    """
    # SMA: скользящие средние
    df["sma_short"] = df["close"].rolling(window=5).mean()
    df["sma_long"] = df["close"].rolling(window=20).mean()
    last_sma_short = df["sma_short"].iloc[-1]
    last_sma_long = df["sma_long"].iloc[-1]
    sma_signal = 1 if last_sma_short > last_sma_long else 0
    conf_sma = round(abs(last_sma_short - last_sma_long) / (last_sma_long + 1e-9), 4)

    # EMA: экспоненциальные скользящие средние
    df["ema_short"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=20, adjust=False).mean()
    last_ema_short = df["ema_short"].iloc[-1]
    last_ema_long = df["ema_long"].iloc[-1]
    ema_signal = 1 if last_ema_short > last_ema_long else 0
    conf_ema = round(abs(last_ema_short - last_ema_long) / (last_ema_long + 1e-9), 4)

    # MACD
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
    last_macd = df["macd"].iloc[-1]
    last_signal_line = df["signal_line"].iloc[-1]
    macd_signal = 1 if last_macd > last_signal_line else 0
    conf_macd = round(abs(last_macd - last_signal_line) / (abs(last_signal_line) + 1e-9), 4)

    # RSI
    delta = df["close"].diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean().iloc[-1]
    avg_loss = loss.rolling(window=14, min_periods=14).mean().iloc[-1]
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
    rsi_signal = 1 if rsi > 50 else 0
    conf_rsi = round(abs(rsi - 50) / 50, 4)

    # Волатильность как отношение стандартного отклонения к среднему
    volatility = df["close"].std() / (df["close"].mean() + 1e-9)
    volatility_adjustment = 1 / (1 + volatility)

    details = {
        "sma_short": last_sma_short,
        "sma_long": last_sma_long,
        "conf_sma": conf_sma,
        "ema_short": last_ema_short,
        "ema_long": last_ema_long,
        "conf_ema": conf_ema,
        "macd": last_macd,
        "signal_line": last_signal_line,
        "conf_macd": conf_macd,
        "rsi": rsi,
        "conf_rsi": conf_rsi,
        "volatility": round(volatility, 4),
        "volatility_adjustment": round(volatility_adjustment, 4),
    }

    signals = {
        "sma_signal": sma_signal,
        "ema_signal": ema_signal,
        "macd_signal": macd_signal,
        "rsi_signal": rsi_signal,
    }
    confidences = {
        "conf_sma": conf_sma,
        "conf_ema": conf_ema,
        "conf_macd": conf_macd,
        "conf_rsi": conf_rsi,
    }
    return {"signals": signals, "confidences": confidences, "volatility_adjustment": volatility_adjustment,
            "details": details}


# ----------------------------
# 2. Прогноз тренда с использованием оптимизируемых параметров
# ----------------------------
def predict_trend(df: pd.DataFrame, params: Dict[str, float]) -> Dict[str, Any]:
    """
    Вычисляет прогноз тренда на основе рассчитанных индикаторов.

    Ожидается, что params содержит:
      - weight_sma, weight_ema, weight_macd, weight_rsi: веса для каждого индикатора.
      - threshold: порог для определения uptrend.
    """
    indicators = compute_indicators(df.copy(), params)
    signals = indicators["signals"]
    confidences = indicators["confidences"]
    vol_adjust = indicators["volatility_adjustment"]

    weight_sma = params.get("weight_sma")
    weight_ema = params.get("weight_ema")
    weight_macd = params.get("weight_macd")
    weight_rsi = params.get("weight_rsi")
    threshold = params.get("threshold")

    # Комбинированный бинарный сигнал
    combined_score = (weight_sma * signals["sma_signal"] +
                      weight_ema * signals["ema_signal"] +
                      weight_macd * signals["macd_signal"] +
                      weight_rsi * signals["rsi_signal"])

    final_trend = "uptrend" if combined_score > threshold else "downtrend"

    # Расчёт совокупной уверенности с корректировкой на волатильность
    combined_confidence = (weight_sma * confidences["conf_sma"] +
                           weight_ema * confidences["conf_ema"] +
                           weight_macd * confidences["conf_macd"] +
                           weight_rsi * confidences["conf_rsi"])
    combined_confidence *= vol_adjust

    # Фактор согласованности
    agreement_factor = abs(combined_score - threshold) * 2
    overall_confidence = round(combined_confidence * agreement_factor, 4)

    return {"trend": final_trend, "confidence": overall_confidence, "combined_score": combined_score,
            "details": indicators["details"]}


# ----------------------------
# 3. Генетическая оптимизация параметров
# ----------------------------
def optimize_parameters(symbol: str, interval: str, exchange: str, population_size: int = 20, generations: int = 10) -> \
        Dict[str, float]:
    """
    Оптимизирует набор параметров (веса и порог) с помощью генетического алгоритма на исторических данных.
    Возвращает набор параметров с наилучшей точностью прогнозов.
    """

    def create_candidate() -> Dict[str, float]:
        w_sma = random.random()
        w_ema = random.random()
        w_macd = random.random()
        w_rsi = random.random()
        total = w_sma + w_ema + w_macd + w_rsi
        return {
            "weight_sma": w_sma / total,
            "weight_ema": w_ema / total,
            "weight_macd": w_macd / total,
            "weight_rsi": w_rsi / total,
            "threshold": random.uniform(0.3, 0.7)
        }

    def evaluate_candidate(candidate: Dict[str, float], historical_data: pd.DataFrame) -> float:
        correct = 0
        total = 0
        for i in range(30, len(historical_data) - 1):
            window = historical_data.iloc[i - 29:i + 1].copy()  # последние 30 свечей
            pred = predict_trend(window, candidate)["trend"]
            current_close = window["close"].iloc[-1]
            next_close = historical_data.iloc[i + 1]["close"]
            actual = "uptrend" if next_close > current_close else "downtrend"
            if pred == actual:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0

    historical_data = fetch_data_from_exchange(exchange, symbol, interval, limit=200)
    if historical_data.empty or len(historical_data) < 50:
        raise ValueError("Недостаточно исторических данных для оптимизации параметров")

    population = [create_candidate() for _ in range(population_size)]
    best_candidate = None

    for gen in range(generations):
        scored_population = []
        for candidate in population:
            fitness = evaluate_candidate(candidate, historical_data)
            scored_population.append((candidate, fitness))
        scored_population.sort(key=lambda x: x[1], reverse=True)
        best_candidate, best_fitness = scored_population[0]
        print(f"Generation {gen + 1}: Best Fitness = {best_fitness:.4f}")

        if best_fitness >= 0.9:
            print("Достигнута целевая точность 90%.")
            return best_candidate

        survivors = [cand for cand, _ in scored_population[:population_size // 2]]
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = {}
            for key in parent1:
                child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
            if random.random() < 0.3:
                mutation_key = random.choice(list(child.keys()))
                if mutation_key.startswith("weight"):
                    child[mutation_key] += random.uniform(-0.1, 0.1)
                    child[mutation_key] = max(child[mutation_key], 0.01)
                if mutation_key == "threshold":
                    child[mutation_key] += random.uniform(-0.05, 0.05)
                    child[mutation_key] = min(max(child[mutation_key], 0), 1)
                total = child["weight_sma"] + child["weight_ema"] + child["weight_macd"] + child["weight_rsi"]
                child["weight_sma"] /= total
                child["weight_ema"] /= total
                child["weight_macd"] /= total
                child["weight_rsi"] /= total
            new_population.append(child)
        population = new_population
    return best_candidate


# ----------------------------
# 4. Вычисление и сохранение прогноза с использованием оптимизированных параметров
# ----------------------------
def compute_and_save_trend_prediction(symbol: str, interval: str, exchange: str) -> Dict[str, Any]:
    """
    Производит прогноз тренда с оптимизированными параметрами и корректировкой по показателю Ляпунова.
    Вместо резкого обнуления использует нормированный λ (lyap_norm) и порог на lyap_norm, чтобы не всегда получать "waiting".
    Сохраняет результат в БД.
    """
    # 1. Загрузка исторических данных
    df = fetch_data_from_exchange(exchange, symbol, interval, limit=30)
    if df.empty or len(df) < 20:
        return {"trend": "waiting", "confidence": 0.0, "details": {}}

    # 2. Генетическая оптимизация параметров и предсказание
    params = optimize_parameters(symbol, interval, exchange)
    prediction = predict_trend(df, params)
    orig_trend = prediction.get("trend", "waiting")
    orig_conf = float(prediction.get("confidence", 0.0))

    # 3. Расчёт экспоненты Ляпунова и нормировка
    series = df["close"].values
    lyapunov_exp = compute_lyapunov_exponent(series, m=2, tau=2)
    lyap_norm = 1.0 / (1.0 + abs(lyapunov_exp))  # в (0,1]

    # 4. Порог на нормированный λ (например, 0.2)
    NORM_THRESHOLD = 0.2

    # 5. Корректировка тренда и уверенности
    if lyap_norm < NORM_THRESHOLD:
        # слишком хаотично: делаем waiting, но сохраняем градацию по lyap_norm
        adj_trend = "waiting"
        adj_confidence = round(lyap_norm, 4)
    else:
        adj_trend = orig_trend
        adj_confidence = round(orig_conf * lyap_norm, 4)

    # 6. Расчёт времени прогноза — начало следующего интервала
    now_ts = int(time.time())
    interval_sec = get_interval_seconds(interval)
    forecast_time = ((now_ts // interval_sec) + 1) * interval_sec

    # 7. Сохранение в БД
    with SessionLocal() as db:
        existing = db.query(TrendForecastDB).filter_by(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            forecast_time=forecast_time
        ).first()

        if not existing:
            trend_forecast = TrendForecastDB(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                forecast_time=forecast_time,
                trend=adj_trend,
                confidence=adj_confidence
            )
            db.add(trend_forecast)
            db.commit()
            db.refresh(trend_forecast)
        else:
            trend_forecast = existing

    # 8. Формирование ответа с подробностями
    details = {
        **prediction.get("details", {}),
        "lyapunov_exp": lyapunov_exp,
        "lyap_norm": lyap_norm,
        "NORM_THRESHOLD": NORM_THRESHOLD,
        "orig_trend": orig_trend,
        "orig_conf": orig_conf,
        "adj_trend": adj_trend,
        "adj_confidence": adj_confidence,
    }

    return {
        "symbol": trend_forecast.symbol,
        "exchange": trend_forecast.exchange,
        "interval": trend_forecast.interval,
        "forecast_time": trend_forecast.forecast_time,
        "trend": trend_forecast.trend,
        "confidence": trend_forecast.confidence,
        "details": details
    }


# ----------------------------
# 5. Планирование прогнозов тренда (оптимизация + прогноз)
# ----------------------------
def schedule_trend_prediction(symbol: str, interval: str, exchange: str) -> None:
    """
    Планирует сохранение прогноза тренда за 10 минут до открытия свечи.
    Если время уже прошло, прогноз сохраняется сразу.
    """
    now_ts = int(time.time())
    interval_sec = get_interval_seconds(interval)
    forecast_time = ((now_ts // interval_sec) + 1) * interval_sec
    run_timestamp = forecast_time - interval_sec - 600
    run_date = datetime.datetime.fromtimestamp(run_timestamp, tz=datetime.timezone.utc)
    job_id = f"trend_{symbol}_{interval}_{exchange}_{forecast_time}"

    if run_date <= datetime.datetime.now(datetime.timezone.utc):
        logging.info(
            f"[schedule_trend_prediction] Время {run_date} уже прошло, сохраняем прогноз сразу для {symbol} на {exchange}.")
        compute_and_save_trend_prediction(symbol, interval, exchange)
        return

    if scheduler.get_job(job_id):
        return

    scheduler.add_job(
        compute_and_save_trend_prediction,
        trigger="date",
        run_date=run_date,
        args=[symbol, interval, exchange],
        id=job_id
    )
    logging.info(
        f"[schedule_trend_prediction] Прогноз для {symbol} на {exchange} запланирован на {run_date} (job_id={job_id}).")


def schedule_all_trend_predictions() -> None:
    """
    Планирует прогноз для всех разрешённых символов и бирж для указанного интервала (например, 1h).
    """
    for symbol in ALLOWED_SYMBOLS:
        for exchange in ALLOWED_EXCHANGES:
            schedule_trend_prediction(symbol, "1h", exchange)


# Запуск периодической задачи планирования прогнозов
scheduler.add_job(
    schedule_all_trend_predictions,
    trigger="interval",
    seconds=60,
    id="schedule_trend_predictions_job"
)
