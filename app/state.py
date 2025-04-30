import datetime
import os
from typing import Dict
from pathlib import Path
import torch
from apscheduler.schedulers.background import BackgroundScheduler
# Лучше брать из fastapi, а не из starlette — тогда ниже не будет конфликтов
from fastapi.templating import Jinja2Templates
from torch import nn

# Глобальные переменные
models: Dict[str, nn.Module] = {}
last_forecast_times: Dict[str, int] = {}
scheduler = BackgroundScheduler(timezone=datetime.timezone.utc)

DEFAULT_DEVICE = "cuda:0"
device_str = os.getenv("DEVICE", DEFAULT_DEVICE)
if device_str != "cpu" and not torch.cuda.is_available():
    print("⚠️ CUDA unavailable, falling back to CPU")
    device_str = "cpu"
device = torch.device(device_str)
print(f"[Startup] Using device: {device}")

# Абсолютный путь до папки app
BASE_DIR = Path(__file__).resolve().parent

# Папка с шаблонами
TEMPLATE_DIR = BASE_DIR / "templates"
if not TEMPLATE_DIR.is_dir():
    raise RuntimeError(f"Templates directory not found: {TEMPLATE_DIR}")

# Единственный экземпляр Jinja2Templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Гарантированно ищем только в этой папке
templates.env.loader.searchpath = [str(TEMPLATE_DIR)]

# Фильтр для UNIX-времени
templates.env.filters["datetimeformat"] = lambda ts: (
    datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    .strftime("%Y-%m-%d %H:%M UTC")
)

print(f"[Startup] Jinja searchpath = {templates.env.loader.searchpath}")
