from apscheduler.schedulers.background import BackgroundScheduler
import torch
from typing import Optional, Dict, Any, List
import datetime

from starlette.templating import Jinja2Templates
from torch import nn

# Глобальные переменные
models: Dict[str, nn.Module] = {}
last_forecast_times: Dict[str, int] = {}  # ключ: f"{symbol}_{interval}_{exchange}" -> forecast_time
scheduler = BackgroundScheduler(timezone=datetime.timezone.utc)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

templates = Jinja2Templates(directory="templates")
# регистрируем фильтр для преобразования UNIX‑времени в строку
templates.env.filters["datetimeformat"] = lambda ts: (
    datetime.datetime
    .fromtimestamp(ts, datetime.timezone.utc)
    .strftime("%Y-%m-%d %H:%M UTC")
)
