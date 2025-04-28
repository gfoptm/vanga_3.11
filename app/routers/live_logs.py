import asyncio

from fastapi import APIRouter
from starlette.responses import StreamingResponse

router = APIRouter(prefix="/live_logs", tags=["live_logs"])


@router.get("")
async def live_logs():
    async def event_generator():
        with open("training.log", "r") as f:
            # Переместим указатель в конец файла
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                yield f"data: {line.rstrip()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
