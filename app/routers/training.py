from typing import Any

from fastapi import Request, APIRouter
from starlette.responses import HTMLResponse

from app.state import templates

router = APIRouter(prefix="/start_training_page", tags=["start_training_page"])


@router.get("", response_class=HTMLResponse)
def start_training_page(request: Request) -> Any:
    return templates.TemplateResponse("start_training.html", {"request": request})
