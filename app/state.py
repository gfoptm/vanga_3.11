import datetime
import os
from typing import Dict
from pathlib import Path
import torch
from apscheduler.schedulers.background import BackgroundScheduler
from starlette.templating import Jinja2Templates
from torch import nn

# Глобальные переменные
models: Dict[str, nn.Module] = {}
last_forecast_times: Dict[str, int] = {}  # ключ: f"{symbol}_{interval}_{exchange}" -> forecast_time
scheduler = BackgroundScheduler(timezone=datetime.timezone.utc)


DEFAULT_DEVICE = "cuda:0"
device_str = os.getenv("DEVICE", DEFAULT_DEVICE)
# если cuda недоступна и не указан "cpu" — падаем на CPU
if device_str != "cpu" and not torch.cuda.is_available():
    print(f"⚠️ CUDA unavailable, falling back to CPU")
    device_str = "cpu"
device = torch.device(device_str)
print(f"Using device: {device}")  # для логов при старте

# Абсолютный путь до папки app
BASE_DIR = Path(__file__).resolve().parent

# Папка с шаблонами
TEMPLATE_DIR = BASE_DIR / "templates"
if not TEMPLATE_DIR.is_dir():
    raise RuntimeError(f"Templates directory not found: {TEMPLATE_DIR}")

# Инициализируем Jinja2Templates только один раз и только здесь
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Регистрируем ваш фильтр
templates.env.filters["datetimeformat"] = lambda ts: (
    datetime.datetime
    .fromtimestamp(ts, datetime.timezone.utc)
    .strftime("%Y-%m-%d %H:%M UTC")
)

# (не печатаем templates.directory — он не существует)
print(f"[Startup] Templates directory = {TEMPLATE_DIR}")
