import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.config import openai_client, ALLOWED_EXCHANGES, \
    ALLOWED_SYMBOLS
from app.database import SessionLocal
from app.dbmodels import LLMForecast
from app.services.news import fetch_enhanced_news_sentiment
from app.services.predictions import compute_and_save_prediction
from app.services.trend import optimize_parameters, predict_trend
from app.state import scheduler
from fetch import fetch_data_from_exchange


def llm_meta_signal(meta_features: dict) -> dict:
    system_prompt = (
        "Вы — мета-модель для торговых сигналов."
        " На основании переданных признаков в формате JSON выдайте ТОЛЬКО JSON с двумя полями:\n"
        "  \"signal\": одна из строк \"BUY\", \"SELL\" или \"WAIT\",\n"
        "  \"confidence\": число от 0 до 1\n"
        "Больше никаких слов."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(meta_features, ensure_ascii=False)}
    ]
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    content = resp.choices[0].message.content.strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise HTTPException(502, f"Неверный ответ LLM, JSON не обнаружен: {content!r}")
    json_str = match.group()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise HTTPException(502, f"Ошибка разбора JSON от LLM: {e.msg}: {json_str!r}")


def llm_explanation(context: dict) -> str:
    system_prompt = (
        "Вы — помощник, объясняющий торговые решения кратко и по-русски."
        " На основании переданного контекста (JSON) напишите несколько предложений (3-4 предложения),"
        " объясните, почему было выбрано именно это действие."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)}
    ]
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return resp.choices[0].message.content.strip()


# Гибридный прогноз

def hybrid_prediction(symbol: str, interval: str, exchange: str) -> dict:
    base = compute_and_save_prediction(symbol, interval, exchange)
    if base.get("signal") == "error":
        raise HTTPException(500, "Сбой базового прогноза")

    params = optimize_parameters(symbol, interval, exchange)
    tech = predict_trend(fetch_data_from_exchange(exchange, symbol, interval, limit=30), params)
    sentiment = fetch_enhanced_news_sentiment(symbol)

    # исходные признаки
    meta_features = {
        "lstm_signal": base["signal"],
        "lstm_confidence": base["confidence"],
        "tech_trend": tech["trend"],
        "tech_confidence": tech["confidence"],
        "news_sentiment": sentiment,
        "lyap_norm": base["features"].get("lyap_norm"),
    }

    # вызываем LLM
    llm_out = llm_meta_signal(meta_features)
    explanation = llm_explanation({**meta_features, **llm_out})

    return {
        "base_prediction": base,
        "tech_prediction": tech,
        "meta_features": meta_features,
        "llm_output": llm_out,
        "explanation": explanation,
        "meta_signal": llm_out.get("signal"),
        "meta_confidence": llm_out.get("confidence"),
    }


def _process_one_combo(args):
    exchange, symbol, interval, next_hour, now_ts = args
    try:
        logging.info(f"[LLM] start {exchange}/{symbol}/{interval}")
        result = hybrid_prediction(symbol, interval, exchange)

        full_meta = {
            **result["meta_features"],
            **result["llm_output"],
            "base_prediction": result["base_prediction"],
            "tech_prediction": result["tech_prediction"],
            "explanation": result["explanation"],
        }

        # отдельная сессия для каждого воркера
        session = SessionLocal()
        llm = LLMForecast(
            symbol=symbol,
            interval=interval,
            exchange=exchange,
            forecast_time=next_hour,
            signal=result["meta_signal"],
            confidence=result["meta_confidence"],
            meta_features=full_meta,
            created_at=now_ts
        )
        session.add(llm)
        session.commit()
        session.close()

        logging.info(f"[LLM] saved id={llm.id} for {exchange}/{symbol}/{interval}")
    except HTTPException as he:
        logging.warning(f"[LLM] skipped {exchange}/{symbol}/{interval} — {he.detail}")
    except SQLAlchemyError as sae:
        logging.error(f"[LLM] DB error for {exchange}/{symbol}/{interval}: {sae}", exc_info=True)
    except Exception as e:
        logging.error(f"[LLM] unexpected error for {exchange}/{symbol}/{interval}: {e}", exc_info=True)


def save_all_llm_forecasts():
    now_ts = int(time.time())
    next_hour = ((now_ts // 3600) + 1) * 3600

    # Генерируем только для 1‑часового интервала
    interval = "1h"
    # собираем все задачи
    combos = [
        (exchange, symbol, interval, next_hour, now_ts)
        for exchange in ALLOWED_EXCHANGES
        for symbol in ALLOWED_SYMBOLS
        # for interval in ALLOWED_INTERVALS
    ]
    logging.info(f"[LLM] launching {len(combos)} tasks in ThreadPoolExecutor")

    # параллельный запуск
    with ThreadPoolExecutor(max_workers=5) as exe:
        futures = [exe.submit(_process_one_combo, combo) for combo in combos]
        for f in as_completed(futures):
            # результат функции ничего не возвращает, но исключения здесь не всплывут
            pass


# регистрация в планировщике — каждые 10 минут
scheduler.add_job(
    save_all_llm_forecasts,
    trigger="cron",
    minute=0,
    id="schedule_llm_forecasts_job",
    max_instances=1,
    coalesce=True,
)
