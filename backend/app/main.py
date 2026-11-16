from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import CORS_ORIGINS
from backend.app.database import init_db
from backend.app.routes.runs import router as runs_router
from backend.app.schemas import QueueStatus
from backend.app.workers.local_worker import LocalWorker, local_worker


def create_app(worker: Optional[LocalWorker] = None) -> FastAPI:
    run_queue = worker if worker is not None else local_worker

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        init_db()
        run_queue.start()
        try:
            yield
        finally:
            run_queue.stop()

    application = FastAPI(
        title="Adaptive Compute TTS API",
        description="Local backend for submitting and monitoring TTS runs.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.run_queue = run_queue
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(runs_router)

    @application.get("/health", tags=["Health"])
    def health_check() -> dict[str, str]:
        return {"status": "healthy"}

    @application.get("/queue", response_model=QueueStatus, tags=["Queue"])
    def queue_status(request: Request) -> QueueStatus:
        return request.app.state.run_queue.snapshot()

    return application


app = create_app()
