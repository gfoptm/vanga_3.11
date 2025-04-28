import logging
from app.config import get_exchange_client, INTERVAL_MAPPING, window_size, ALLOWED_SYMBOLS, ALLOWED_EXCHANGES
from app.database import SessionLocal
from app.dbmodels import SignalDB, ForecastComparison, TrendForecastDB, ForecastDB, PredictionDB, FundamentalForecastDB

from app.services.lyapunov import compute_lyapunov_exponent
from app.state import scheduler

from fetch import interval_to_timedelta, fetch_data_from_exchange
import datetime

from typing import Optional, Dict, Any, List
import random


def compute_and_save_fundamental_prediction(symbol: str, exchange: str) -> Dict[str, Any]:
    """
    Вычисляет оптимизированный фундаментальный прогноз с учётом показателя Ляпунова.
    Используются данные за последние 14 дней для вычисления фундаментальных индикаторов,
    а также оценка хаотичности ряда на основе λ.
    """
    # 1. Загрузка данных за 14 дней
    df = fetch_data_from_exchange(exchange, symbol, "1d", limit=14)
    if df.empty or len(df) < 7:
        return {
            "status": "error",
            "message": f"Недостаточно данных для {symbol} на {exchange} для фундаментального анализа"
        }

    # Базовая цена
    base_price = df.iloc[-1]["close"]

    # 2. Вычисление фундаментальных метрик
    # 2.1 Моментум
    start_price = df.iloc[0]["close"]
    momentum = (base_price - start_price) / start_price
    momentum_score = (momentum + 0.10) / 0.20
    momentum_score = max(0.0, min(momentum_score, 1.0))

    # 2.2 Смещение к SMA
    sma_14 = df["close"].mean()
    bias = (base_price / sma_14) - 1.0
    sma_bias_score = (bias + 0.05) / 0.10
    sma_bias_score = max(0.0, min(sma_bias_score, 1.0))

    # 2.3 RSI
    delta = df["close"].diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
    avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_score = (rsi - 30.0) / 40.0
    rsi_score = max(0.0, min(rsi_score, 1.0))

    # 2.4 Волатильность
    volatility = df["close"].std() / sma_14
    volatility_score = 1.0 - min(volatility / 0.10, 1.0)

    # 2.5 Объём
    last_volume = df.iloc[-1]["volume"]
    avg_volume = df["volume"].mean()
    volume_ratio = last_volume / avg_volume if avg_volume != 0 else 1.0
    volume_score = 0.5 + (volume_ratio - 1.0) * 0.10
    volume_score = max(0.0, min(volume_score, 1.0))

    # 2.6 Случайный фактор
    random_score = random.random()

    # Оригинальный композитный score
    composite_orig = (
            0.25 * momentum_score +
            0.20 * sma_bias_score +
            0.20 * rsi_score +
            0.15 * volatility_score +
            0.10 * volume_score +
            0.10 * random_score
    )

    # 3. Оценка хаотичности: Ляпунов
    time_series = df["close"].values
    lyapunov_exp = compute_lyapunov_exponent(time_series, m=3, tau=1)
    lyap_norm = 1.0 / (1.0 + abs(lyapunov_exp))
    L_THRESHOLD = 0.5
    chaotic = lyapunov_exp > L_THRESHOLD

    # 4. Корректировка прогноза
    composite_adj = composite_orig * lyap_norm

    if chaotic:
        # при сильной хаотичности сохраняем небольшой сигнал и уверенность по lyap_norm
        signal = "waiting"
        confidence = round(lyap_norm, 4)
        # лёгкое отклонение на основе composite_adj и lyap_norm
        predicted_change = (composite_adj - 0.5) * lyap_norm * 0.02
        predicted_price = base_price * (1.0 + predicted_change)
    else:
        signal = "buy" if composite_adj > 0.5 else "sell"
        predicted_change = (composite_adj - 0.5) * 0.10
        predicted_price = base_price * (1.0 + predicted_change)
        confidence = round(abs(composite_adj - 0.5) * 2.0, 4)

    # 5. Время прогноза: начало следующего дня UTC
    now = datetime.datetime.utcnow()
    tomorrow = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)
    forecast_time = int(tomorrow.timestamp())

    # 6. Сохранение в БД
    with SessionLocal() as db:
        existing = db.query(FundamentalForecastDB).filter_by(
            symbol=symbol,
            exchange=exchange,
            forecast_time=forecast_time
        ).first()

        if not existing:
            fundamental_forecast = FundamentalForecastDB(
                symbol=symbol,
                exchange=exchange,
                forecast_time=forecast_time,
                signal=signal,
                confidence=confidence,
                price=round(predicted_price, 2)
            )
            db.add(fundamental_forecast)
            db.commit()
            db.refresh(fundamental_forecast)
        else:
            fundamental_forecast = existing

    # 7. Формируем ответ с деталями
    return {
        "symbol": symbol,
        "exchange": exchange,
        "forecast_time": forecast_time,
        "signal": fundamental_forecast.signal,
        "confidence": fundamental_forecast.confidence,
        "price": fundamental_forecast.price,
        "details": {
            "momentum_score": momentum_score,
            "sma_bias_score": sma_bias_score,
            "rsi_score": rsi_score,
            "volatility_score": volatility_score,
            "volume_score": volume_score,
            "random_score": random_score,
            "composite_orig": composite_orig,
            "lyapunov_exp": lyapunov_exp,
            "lyap_norm": lyap_norm,
            "chaotic": chaotic,
            "composite_adj": composite_adj
        }
    }


def update_fundamental_forecasts_job() -> None:
    """
    Периодическая задача, которая запускает фундаментальный анализ.
    Для каждого разрешённого символа и биржи вычисляется и сохраняется фундаментальный прогноз.
    """
    for symbol in ALLOWED_SYMBOLS:
        for exchange in ALLOWED_EXCHANGES:
            result = compute_and_save_fundamental_prediction(symbol, exchange)
            logging.info(
                f"[update_fundamental_forecasts_job] Фундаментальный прогноз для {symbol} на {exchange}: {result}")


# Добавление задачи планирования для фундаментального прогноза.
# Здесь настроено выполнение задачи каждые 24 часа; при необходимости интервал можно изменить.
scheduler.add_job(
    update_fundamental_forecasts_job,
    trigger="interval",
    hours=24,
    id="update_fundamental_forecasts_job"
)
