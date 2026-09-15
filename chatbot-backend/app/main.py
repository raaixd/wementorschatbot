from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from . import db
from .knowledge import load_knowledge
from .pipeline import run_pipeline
from .schemas import ChatRequest, ChatResponse, ClearRequest, ClearResponse

logger = logging.getLogger("wementors.chatbot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_rate_buckets: Dict[str, List[float]] = defaultdict(list)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    load_knowledge()
    logger.info("WeMentors chatbot started with %s entries", len(load_knowledge().entries))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG chatbot API for the WeMentors website.",
        lifespan=lifespan,
    )

    origins = settings.allowed_origins
    if settings.environment == "development" and "*" not in origins:
        origins = origins + ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if request.url.path in {"/chat", "/chat/clear"}:
            client = request.client.host if request.client else "unknown"
            now = time.time()
            window = settings.rate_limit_window_seconds
            bucket = [stamp for stamp in _rate_buckets[client] if now - stamp < window]
            if len(bucket) >= settings.rate_limit_requests:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please wait a moment and try again."},
                )
            bucket.append(now)
            _rate_buckets[client] = bucket
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": "Please send a valid chat message (1 to 2000 characters)."},
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong while answering. Please try again."},
        )

    @app.get("/health")
    def health_check() -> Dict[str, object]:
        kb = load_knowledge()
        return {
            "status": "healthy",
            "service": "wementors-chatbot",
            "knowledge_entries": len(kb.entries),
            "database": "ok" if db.db_ok() else "error",
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        request_id = uuid.uuid4().hex[:8]
        try:
            reply, session_id, intent, sources = run_pipeline(
                request.message,
                request.history,
                request.session_id,
            )
            logger.info("chat %s intent=%s session=%s", request_id, intent, session_id[:8])
            return ChatResponse(reply=reply, session_id=session_id, intent=intent, sources=sources)
        except HTTPException:
            raise
        except Exception:
            logger.exception("chat failed %s", request_id)
            raise HTTPException(status_code=500, detail="Something went wrong while answering. Please try again.")

    @app.post("/chat/clear", response_model=ClearResponse)
    def clear_chat(request: ClearRequest) -> ClearResponse:
        db.ensure_session(request.session_id)
        db.clear_session(request.session_id)
        return ClearResponse(ok=True, session_id=request.session_id)

    return app


app = create_app()
