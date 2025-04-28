import logging

from sqlalchemy.orm import Session

from app.config import get_exchange_client, INTERVAL_MAPPING, window_size, ALLOWED_SYMBOLS, ALLOWED_EXCHANGES
from app.database import SessionLocal
from app.dbmodels import SignalDB, ForecastComparison, TrendForecastDB, ForecastDB, PredictionDB, CombinedForecast, \
    FundamentalForecastDB, NewsSentimentForecast, LLMForecast
from app.routers.trend import fetch_candle_by_start_time
from app.state import scheduler

from app.utils.time import align_forecast_time, get_interval_seconds

import datetime
import time
from typing import Optional, Dict, Any, List


# -----------------------
# Начало общего сравнения
# -----------------------
def update_and_store_combined_forecasts(
        symbol: Optional[str],
        exchange: Optional[str],
        interval: str,
        db: Session,
        include_old: bool = False
) -> List[Any]:
    """
    Обновляет и сохраняет агрегированные прогнозы, включая данные из прогнозов
    цены, фундаментальных, трендовых, новостных и LLM-прогнозов.
    Данные сохраняются только если для данного forecast_time ещё нет записи.
    """
    now_ts = int(time.time())
    results: List[Any] = []
    symbols_to_check = ALLOWED_SYMBOLS if symbol is None else [symbol]
    exchanges_to_check = ALLOWED_EXCHANGES if exchange is None else [exchange]

    for sym in symbols_to_check:
        for exch in exchanges_to_check:
            # Получаем последний прогноз из ForecastDB
            forecast = (db.query(ForecastDB)
                        .filter(
                ForecastDB.symbol == sym,
                ForecastDB.exchange == exch,
                ForecastDB.interval == interval,
                ForecastDB.forecast_time <= now_ts
            )
                        .order_by(ForecastDB.forecast_time.desc())
                        .first())
            if not forecast:
                if include_old:
                    forecast = (db.query(ForecastDB)
                                .filter(
                        ForecastDB.symbol == sym,
                        ForecastDB.exchange == exch,
                        ForecastDB.interval == interval
                    )
                                .order_by(ForecastDB.forecast_time.desc())
                                .first())
                    if not forecast:
                        continue
                else:
                    continue

            # Приводим forecast_time к int
            forecast_ts = int(forecast.forecast_time)

            # Пропускаем, если комбинированная запись уже существует
            if db.query(CombinedForecast).filter_by(
                    symbol=sym,
                    exchange=exch,
                    interval=interval,
                    forecast_time=forecast_ts
            ).first():
                continue

            # Сигнал из SignalDB
            signal = (db.query(SignalDB)
                      .filter(
                SignalDB.symbol == sym,
                SignalDB.exchange == exch,
                SignalDB.interval == interval,
                SignalDB.forecast_time == forecast_ts
            )
                      .order_by(SignalDB.timestamp.desc())
                      .first())
            if not signal:
                signal = (db.query(SignalDB)
                          .filter(
                    SignalDB.symbol == sym,
                    SignalDB.exchange == exch,
                    SignalDB.interval == interval
                )
                          .order_by(SignalDB.timestamp.desc())
                          .first())

            # Фундаментальный прогноз
            fundamental = (db.query(FundamentalForecastDB)
                           .filter(
                FundamentalForecastDB.symbol == sym,
                FundamentalForecastDB.exchange == exch
            )
                           .order_by(FundamentalForecastDB.forecast_time.desc())
                           .first())

            # Прогноз тренда
            trend = (db.query(TrendForecastDB)
                     .filter(
                TrendForecastDB.symbol == sym,
                TrendForecastDB.exchange == exch,
                TrendForecastDB.interval == interval,
                TrendForecastDB.forecast_time <= now_ts
            )
                     .order_by(TrendForecastDB.forecast_time.desc())
                     .first())

            # Время открытия свечи
            interval_sec = get_interval_seconds(interval)
            candle_start = forecast_ts - interval_sec
            candle_time_str = datetime.datetime.fromtimestamp(
                candle_start, tz=datetime.timezone.utc
            ).strftime('%Y-%m-%d %H:%M:%S')

            # Фактические данные свечи
            actual_open, actual_close = None, None
            actual_candle = fetch_candle_by_start_time(sym, interval, candle_start, exch)
            if actual_candle:
                actual_open = actual_candle.get("open")
                actual_close = actual_candle.get("close")

            # Собираем корреляции
            score_list: List[float] = []
            # 1. Цена
            if forecast and actual_close:
                err = abs(forecast.price - actual_close) / actual_close * 100
                score_list.append(max(0, 100 - err))
            # 2. Фундаментал
            if fundamental and actual_close:
                err = abs(fundamental.price - actual_close) / actual_close * 100
                score_list.append(max(0, 100 - err))
            # 3. Сигнал
            if signal and actual_open is not None and actual_close is not None:
                sig_corr = 100 if ((signal.signal == "buy" and actual_close >= actual_open) or
                                   (signal.signal == "sell" and actual_close <= actual_open)) else 0
                score_list.append(sig_corr)
            # 4. Тренд
            if trend and actual_open is not None and actual_close is not None:
                actual_tr = "uptrend" if actual_close > actual_open else "downtrend"
                tr_corr = 100 if trend.trend == actual_tr else 0
                score_list.append(tr_corr)
            # 5. Новости
            news_record = (db.query(NewsSentimentForecast)
                           .filter(
                NewsSentimentForecast.symbol == sym,
                NewsSentimentForecast.forecast_time == forecast_ts
            )
                           .first())
            if news_record:
                news_sig = "buy" if news_record.sentiment_score > 0 else "sell"
                news_corr = 100 if (forecast and news_sig == forecast.signal) else 0
                score_list.append(news_corr)
            # 6. LLM
            llm_record = (db.query(LLMForecast)
                          .filter(
                LLMForecast.symbol == sym,
                LLMForecast.exchange == exch,
                LLMForecast.interval == interval,
                LLMForecast.forecast_time == forecast_ts
            )
                          .order_by(LLMForecast.created_at.desc())
                          .first())
            llm_signal, llm_confidence = None, None
            if llm_record:
                llm_signal = llm_record.signal.lower()
                llm_confidence = llm_record.confidence
                if actual_open is not None and actual_close is not None:
                    llm_corr = 100 if ((llm_signal == "buy" and actual_close >= actual_open) or
                                       (llm_signal == "sell" and actual_close <= actual_open)) else 0
                    score_list.append(llm_corr)

            overall_corr = round(sum(score_list) / len(score_list), 2) if score_list else None

            # Создаём запись
            combined = CombinedForecast(
                symbol=sym,
                exchange=exch,
                interval=interval,
                forecast_price=forecast.price if forecast else None,
                forecast_signal=forecast.signal if forecast else None,
                signal=signal.signal if signal else None,
                fundamental_price=fundamental.price if fundamental else None,
                fundamental_signal=fundamental.signal if fundamental else None,
                trend=trend.trend if trend else None,
                trend_confidence=trend.confidence if trend else None,
                news_sentiment=news_record.sentiment_score if news_record else None,
                llm_signal=llm_signal,
                llm_confidence=llm_confidence,
                actual_open=actual_open,
                actual_close=actual_close,
                overall_correlation=overall_corr,
                candle_time=candle_time_str,
                forecast_time=forecast_ts
            )
            try:
                db.add(combined)
                db.commit()
                db.refresh(combined)
                results.append(combined)
            except Exception as e:
                db.rollback()
                logging.exception(f"Ошибка сохранения CombinedForecast для {sym}/{exch}: {e}")
    return results


# ============================================================================
# Новый scheduled job для сохранения агрегированных прогнозов каждый час
# ============================================================================
def scheduled_combined_forecasts_job():
    """
    Создаёт новую сессию БД, обновляет и сохраняет агрегированные прогнозы для всех разрешённых символов
    и бирж с таймфреймом "1h". Логирует количество созданных записей.
    """
    db = SessionLocal()
    try:
        results = update_and_store_combined_forecasts(None, None, "1h", db, include_old=False)
        logging.info(
            f"[scheduled_combined_forecasts_job] Создано {len(results)} новых записей агрегированных прогнозов.")
    except Exception as e:
        logging.exception(f"[scheduled_combined_forecasts_job] Ошибка при обновлении агрегированных прогнозов: {e}")
    finally:
        db.close()


# Планируем запуск задачи каждый час – в начале каждого часа (minute=0)
scheduler.add_job(
    scheduled_combined_forecasts_job,
    trigger="cron",
    minute=0,
    id="combined_forecasts_job"
)
