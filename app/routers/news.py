from fastapi import Request, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import HTMLResponse

from app.database import get_db
from app.dbmodels import NewsSentimentForecast
from app.state import templates

router = APIRouter(prefix="/news_forecasts_html", tags=["news_forecasts_html"])


@router.get("", response_class=HTMLResponse)
def news_forecasts_html(request: Request, db: Session = Depends(get_db)):
    """
    Отдаёт HTML-страницу с таблицей, содержащей данные новостных прогнозов,
    и пояснением, как интерпретировать данные.
    """
    forecasts = db.query(NewsSentimentForecast).order_by(NewsSentimentForecast.timestamp.desc()).all()
    return templates.TemplateResponse("news_forecasts.html", {"request": request, "forecasts": forecasts})
