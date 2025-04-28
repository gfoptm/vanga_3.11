import math

import logging
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import feedparser
from sqlalchemy.orm import Session

from app.config import get_exchange_client, INTERVAL_MAPPING, window_size, SOURCE_RELIABILITY, NEWS_API_CONFIGS, \
    CRYPTO_RSS_FEEDS, ALLOWED_SYMBOLS
from app.database import SessionLocal
from app.dbmodels import SignalDB, ForecastComparison, TrendForecastDB, ForecastDB, PredictionDB, NewsSentimentForecast

import datetime
import time

from app.state import scheduler


# === Функция для получения новостной тональности с учётом веса по времени и надёжности ===
def fetch_enhanced_news_sentiment(symbol: str) -> float:
    """
    Усиленный сбор новостных данных для заданного символа и вычисление средней тональности.
      - Объединяет заголовок и описание (если есть);
      - Взвешивает оценку по времени публикации (рецентность);
      - Учитывает рейтинг надёжности источника.
    """

    analyzer = SentimentIntensityAnalyzer()
    total_weighted_score = 0.0
    total_weight = 0.0
    now_ts = time.time()

    def process_article(article: dict, source_name: str):
        title = article.get("title", "")
        description = article.get("description", "") or article.get("content", "")
        text = f"{title}. {description}" if description else title
        if not text:
            return None
        sentiment = analyzer.polarity_scores(text)["compound"]

        # Получение времени публикации (ожидается ISO-формат)
        published_at = article.get("publishedAt") or article.get("published") or ""
        article_ts = None
        try:
            if published_at:
                dt = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                article_ts = dt.timestamp()
        except Exception as e:
            logging.debug(f"Ошибка при парсинге даты публикации: {e}")

        # Вес по рецентности — экспоненциальное уменьшение каждые 6 часов
        recency_weight = math.exp(-((now_ts - article_ts) / (6 * 3600))) if article_ts else 0.5

        # Коэффициент надёжности (значения заданы в SOURCE_RELIABILITY)
        reliability = SOURCE_RELIABILITY.get(source_name.lower(), 0.8)
        weight = recency_weight * reliability
        return sentiment, weight

    # Обработка новостей из API (NEWS_API_CONFIGS) и RSS-лент (CRYPTO_RSS_FEEDS)
    for config in NEWS_API_CONFIGS:
        for key in config["api_keys"]:
            try:
                params = config["params_base"].copy()
                if config["name"] == "NewsAPI":
                    params.update({"q": symbol, "apiKey": key})
                elif config["name"] == "GNews":
                    params.update({"q": symbol, "token": key})
                logging.info(f"[fetch_enhanced_news_sentiment] Пробуем {config['name']} с ключом {key[:5]}...")
                response = requests.get(config["url"], params=params, timeout=10)
                if response.status_code != 200:
                    logging.warning(f"[{config['name']}] Ошибка {response.status_code}")
                    continue

                data = response.json()
                articles = data.get("articles", [])
                for article in articles:
                    source = ""
                    if config["name"] == "NewsAPI":
                        source = article.get("source", {}).get("name", "")
                    elif config["name"] == "GNews":
                        source = article.get("source", "")
                    result = process_article(article, source)
                    if result:
                        sentiment, weight = result
                        total_weighted_score += sentiment * weight
                        total_weight += weight

                if total_weight > 0:
                    break
            except Exception as e:
                logging.error(f"[{config['name']}] Ошибка: {e}")
                continue
        if total_weight > 0:
            break

    # Если API не вернуло результатов, пробуем получать новости из RSS-лент
    if total_weight == 0:
        logging.info(f"[fetch_enhanced_news_sentiment] Используем RSS для {symbol}")
        for source, rss_url in CRYPTO_RSS_FEEDS.items():
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries:
                    article = {
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", ""),
                    }
                    if "published_parsed" in entry and entry.published_parsed:
                        dt = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        article["publishedAt"] = dt.isoformat()
                    result = process_article(article, source)
                    if result:
                        sentiment, weight = result
                        total_weighted_score += sentiment * weight
                        total_weight += weight
            except Exception as e:
                logging.error(f"[RSS {source}] Ошибка: {e}")
                continue

    if total_weight == 0:
        logging.info(f"[fetch_enhanced_news_sentiment] Новостей не найдено по теме '{symbol}'")
        return 0.0

    avg_sentiment = total_weighted_score / total_weight
    logging.info(f"[fetch_enhanced_news_sentiment] {symbol} enhanced sentiment: {avg_sentiment:.3f}")
    return avg_sentiment


# ---------------------------------------------------------------------
# Функция сохранения (расчёта) новостного прогноза в БД
# ---------------------------------------------------------------------
def compute_and_save_news_forecast(symbol: str, forecast_time: int) -> None:
    """
    Вычисляет тональность новостей для symbol, затем сохраняет (или обновляет) прогноз
    для указанного forecast_time в таблице NewsSentimentForecast.
    """
    sentiment = fetch_enhanced_news_sentiment(symbol)
    db: Session = SessionLocal()
    try:
        forecast = (
            db.query(NewsSentimentForecast)
            .filter(
                NewsSentimentForecast.symbol == symbol,
                NewsSentimentForecast.forecast_time == forecast_time
            )
            .first()
        )
        if forecast:
            forecast.sentiment_score = sentiment
        else:
            forecast = NewsSentimentForecast(
                symbol=symbol,
                forecast_time=forecast_time,
                sentiment_score=sentiment
            )
            db.add(forecast)
        db.commit()
        logging.info(
            f"[compute_and_save_news_forecast] Прогноз для {symbol} на {forecast_time} сохранён со значением {sentiment:.3f}")
    except Exception as e:
        db.rollback()
        logging.error(f"[compute_and_save_news_forecast] Ошибка при сохранении прогноза для {symbol}: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------
# Планирование (scheduler) для сохранения новостного прогноза
# ---------------------------------------------------------------------
def schedule_news_forecast(symbol: str) -> None:
    """
    Для заданного символа рассчитывает ближайший прогноз (на следующий час) и планирует задачу
    по сохранению прогноза за 10 минут до начала прогноза.
    Если время для запуска уже прошло – выполняет сохранение сразу.
    """
    now_ts = int(time.time())
    interval_sec = 3600  # прогноз на ближайший час
    forecast_time = ((now_ts // interval_sec) + 1) * interval_sec
    run_timestamp = forecast_time - 600  # запуск за 10 минут до forecast_time
    run_date = datetime.datetime.fromtimestamp(run_timestamp, tz=datetime.timezone.utc)
    job_id = f"news_{symbol}_{forecast_time}"

    # Если время запуска уже прошло – сохраняем прогноз немедленно
    if run_date <= datetime.datetime.now(datetime.timezone.utc):
        logging.info(
            f"[schedule_news_forecast] Время запуска {run_date} уже прошло для {symbol}. Сохраняем прогноз сразу.")
        compute_and_save_news_forecast(symbol, forecast_time)
        return

    # Если задача с таким идентификатором уже запланирована, ничего не делаем
    if scheduler.get_job(job_id):
        return

    scheduler.add_job(
        compute_and_save_news_forecast,
        trigger="date",
        run_date=run_date,
        args=[symbol, forecast_time],
        id=job_id
    )
    logging.info(f"[schedule_news_forecast] Прогноз для {symbol} запланирован на {run_date} (job_id={job_id}).")


def schedule_all_news_forecasts() -> None:
    """
    Для всех разрешённых символов (ALLOWED_SYMBOLS) планирует сохранение новостного прогноза.
    """
    for symbol in ALLOWED_SYMBOLS:
        schedule_news_forecast(symbol)


# Добавляем периодическую задачу, которая каждые 10 минут обновляет (перепланирует) задания для новостного прогноза
scheduler.add_job(
    schedule_all_news_forecasts,
    trigger="interval",
    minutes=10,
    id="schedule_news_forecasts_job"
)
