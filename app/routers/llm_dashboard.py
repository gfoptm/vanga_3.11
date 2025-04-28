from fastapi import Request, Query, HTTPException, APIRouter
from starlette.responses import HTMLResponse

from app.config import ALLOWED_SYMBOLS, ALLOWED_INTERVALS, ALLOWED_EXCHANGES
from app.services.llm import hybrid_prediction
from app.state import templates

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_class=HTMLResponse)
def dashboard(
        request: Request,
        symbol: str = Query(default="BTCUSDT", description="Торговая пара"),
        interval: str = Query(default="1h", description="Тайм‑фрейм"),
        exchange: str = Query(default="binance", description="Биржа")
):
    try:
        result = hybrid_prediction(symbol, interval, exchange)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка при формировании гибридного прогноза: {e}")

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "symbol": symbol,
        "interval": interval,
        "exchange": exchange,
        "allowed_symbols": ALLOWED_SYMBOLS,
        "allowed_intervals": ALLOWED_INTERVALS,
        "allowed_exchanges": ALLOWED_EXCHANGES,
        "explanation": result["explanation"],
        "signal": result["meta_signal"],
        "confidence": result["meta_confidence"],
        "base_prediction": result["base_prediction"],
        "tech_prediction": result["tech_prediction"],
    })
