import logging
import os
import warnings
from typing import Any

import nltk
nltk.download('vader_lexicon')

import uvicorn
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_SYMBOLS, ALLOWED_EXCHANGES, window_size
from app.state import device, models, scheduler, templates
from fetch import fetch_data_from_exchange
from train import train_model_for_symbol, feature_engineering
from model import build_lstm_model

# Роутеры
from app.routers.combined import router as combined_router
from app.routers.trend import router as trend_router
from app.routers.fundamental import router as fundamental_router
from app.routers.news import router as news_router
from app.routers.llm import router as llm_router
from app.routers.logs import router as logs_router
from app.routers.predictions import router as predictions_router
from app.routers.training import router as training_router
from app.routers.api_signals import router as api_signals_router
from app.routers.signals import router as signals_router
from app.routers.live_predictions import router as live_predictions_router
from app.routers.latest_prediction import router as latest_prediction_router
from app.routers.candles import router as candles_router
from app.routers.forecasts import router as forecasts_router
from app.routers.forecast_comparisons import router as forecast_comparison_router
from app.routers.schedule_forecast_comparison import router as schedule_forecast_comparison_router
from app.routers.forecast_comparison_page import router as forecast_comparison_page_router
from app.routers.forecast_comparison_page_data import router as forecast_comparison_page_data_router
from app.routers.api_combined_forecasts import router as combined_forecasts_router
from app.routers.update_combined_forecasts import router as update_combined_forecasts_router
from app.routers.llm_dashboard import router as llm_dashboard_router
from app.routers.llm_forecasts import router as llm_forecasts_router
from app.routers.live_logs import router as live_logs_router
from app.routers.start_news_forecast import router as start_news_forecast_router
from app.routers.news_forecasts import router as news_forecasts_router
from app.routers.start_training import router as start_training_router
from app.routers.trend_forecasts import router as trend_forecasts_router
from app.routers.start_fundamental_prediction import router as start_fundamental_prediction_router
from app.routers.futures_forecasts import router as futures_forecasts_router
from app.routers.futures_signal import router as futures_signal_router
from app.routers.futures_simulate import router as futures_simulate_router
from app.routers.estimate_futures_params import router as estimate_futures_params_router

# Настройка логирования и предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Утилиты для загрузки/тренировки моделей
def _model_file_name(symbol: str, exchange: str) -> str:
    return f"lstm_{symbol}_{exchange}_model.pth"

def _load_or_train(symbol: str, exchange: str) -> None:
    key = f"{symbol}_{exchange}"
    model_file = _model_file_name(symbol, exchange)

    # 1) Попытка загрузить сохранённую модель
    if os.path.exists(model_file):
        try:
            ckpt = torch.load(model_file, map_location=device)
            cfg = ckpt["config"]
            state = ckpt["state_dict"]

            # Нам нужно знать input_size, поэтому повторно считаем фичи для df
            df = fetch_data_from_exchange(exchange, symbol, "1h", limit=window_size + 150)
            if df.empty:
                logging.warning(f"[Startup] No data for {key}, cannot infer input_size, retraining.")
                raise ValueError("Empty data for feature inference")

            df_feat = feature_engineering(df.copy())
            input_size = df_feat.shape[1]

            model = build_lstm_model(
                num_layers=cfg["num_layers"],
                units1=cfg["units1"],
                units2=cfg["units2"],
                dropout_rate=cfg["dropout_rate"],
                input_size=input_size,
                use_genetics=False
            ).to(device)
            model.load_state_dict(state)
            model.eval()
            models[key] = model
            logging.info(f"[Startup] Loaded model {key}")
            return

        except Exception as e:
            logging.warning(f"[Startup] Failed to load {model_file} for {key}: {e}")

    # 2) Если загрузка не удалась — собираем данные и тренируем новую модель
    df = fetch_data_from_exchange(exchange, symbol, "1h", limit=window_size + 150)
    if df.empty:
        logging.warning(f"[Startup] No data for {key}, skipping training.")
        return

    model = train_model_for_symbol(df, symbol, exchange, use_genetics=True)
    if model:
        models[key] = model
        try:
            # Сохраняем конфиг и состояние модели
            torch.save({
                "config": {
                    "num_layers": model.num_layers,
                    "units1": model.units1,
                    "units2": model.units2,
                    "dropout_rate": model.dropout_rate,
                    "input_size": model.input_size
                },
                "state_dict": model.state_dict()
            }, model_file)
            logging.info(f"[Startup] Trained and saved model {key}")
        except Exception as e:
            logging.error(f"[Startup] Failed to save {model_file}: {e}")

# Инициализация FastAPI
app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory="app/static"),  # <-- здесь ваш фактический путь
    name="static",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Регистрация роутеров
routers = [
    combined_router, trend_router, fundamental_router, news_router, llm_router,
    logs_router, predictions_router, training_router, api_signals_router,
    signals_router, live_predictions_router, latest_prediction_router,
    candles_router, forecasts_router, forecast_comparison_router,
    schedule_forecast_comparison_router, forecast_comparison_page_router,
    forecast_comparison_page_data_router, combined_forecasts_router,
    update_combined_forecasts_router, llm_dashboard_router, llm_forecasts_router,
    live_logs_router, start_news_forecast_router, news_forecasts_router,
    start_training_router, trend_forecasts_router,
    start_fundamental_prediction_router, futures_forecasts_router,
    futures_signal_router, futures_simulate_router,
    estimate_futures_params_router
]
for r in routers:
    app.include_router(r)

@app.on_event("startup")
def on_startup():
    logging.info("[Startup] Initializing models for all symbols/exchanges...")
    for sym in ALLOWED_SYMBOLS:
        for exch in ALLOWED_EXCHANGES:
            _load_or_train(sym, exch)

    # Запускаем периодическое обновление
    scheduler.add_job(update_models_job, 'interval', hours=3, replace_existing=True)
    scheduler.start()
    logging.info("[Startup] Scheduler started.")

def update_models_job() -> None:
    logging.info("[update_models_job] Retraining all models...")
    for sym in ALLOWED_SYMBOLS:
        for exch in ALLOWED_EXCHANGES:
            _load_or_train(sym, exch)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    return templates.TemplateResponse("chart.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
