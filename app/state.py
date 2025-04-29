import datetime
import os
from typing import Dict

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

# Получаем абсолютный путь к текущему файлу
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Шаблоны лежат в папке app/templates относительно этого модуля
templates_dir = os.path.join(BASE_DIR, "app", "templates")

# Инициализируем Jinja2Templates с полным путём
templates = Jinja2Templates(directory=templates_dir)

# Регистрируем фильтр для преобразования UNIX-времени в строку
templates.env.filters["datetimeformat"] = lambda ts: (
    datetime.datetime
    .fromtimestamp(ts, datetime.timezone.utc)
    .strftime("%Y-%m-%d %H:%M UTC")
)
