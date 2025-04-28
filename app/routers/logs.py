import os
from typing import Any

from fastapi import Request, APIRouter
from starlette.responses import HTMLResponse

from app.state import templates

router = APIRouter(prefix="/logs", tags=["Logs"])


@router.get("", response_class=HTMLResponse)
async def logs_view(request: Request) -> Any:
    if not os.path.exists("training.log"):
        return templates.TemplateResponse("logs.html", {"request": request, "log_lines": ["Лог файл не найден."]})
    with open("training.log", "r") as f:
        lines = f.readlines()[-300:]
    return templates.TemplateResponse("logs.html", {"request": request, "log_lines": lines})
