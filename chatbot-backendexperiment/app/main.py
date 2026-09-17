"""
WeMentors Chatbot API.

Endpoints
---------
GET  /health              - liveness/readiness check
POST /chat                - main chat endpoint (RAG pipeline)
POST /chat/clear          - clear a session's stored conversation history
POST /feedback            - optional thumbs up/down on a reply
GET  /admin/analytics     - simple aggregate stats, protected by ADMIN_API_KEY

See README.md for setup and chatbot-backend/.env.example for configuration.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response, FileResponse


class UTF8JSONResponse(JSONResponse):
    """JSON responses that declare their charset.

    Starlette's default JSONResponse sends `Content-Type: application/json`
    with no charset. The body is already UTF-8, but clients that assume
    ISO-8859-1 in that situation — notably PowerShell's Invoke-RestMethod
    on Windows PowerShell 5.1 — decode it wrongly, which is why em dashes
    and apostrophes appeared as "a EUR TM"-style mojibake. Declaring the
    charset explicitly fixes it at the source; the knowledge base keeps
    its real punctuation."""

    media_type = "application/json; charset=utf-8"
from pydantic import BaseModel, Field

from . import config, database, llm
from .conversation import ConversationEngine
from .knowledge import KnowledgeBaseError, load_entries
from .ratelimit import RateLimiter

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger("wementors.api")

# Session ids must look like a generated token (the frontend uses
# crypto.randomUUID()). Length floor of 8 prevents guessable values.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,100}")

app = FastAPI(
    title="WeMentors Chatbot API",
    version="2.1.0",
    description="RAG-powered FAQ chatbot backend for the WeMentors website.",
    default_response_class=UTF8JSONResponse,
)

# Allowed origins for the website and local development servers
_cors_origins = list(dict.fromkeys(["null", "http://127.0.0.1:5500", "http://localhost:5500", *config.ALLOWED_ORIGINS]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-Id", "X-Admin-Key"],
)

# --- knowledge base / conversation engine ------------------------------
# Loaded once at startup. If the knowledge base is broken, the app still
# starts (so /health works and ops can see the problem) but /chat returns
# a safe error instead of crashing.
_engine: Optional[ConversationEngine] = None
_kb_load_error: Optional[str] = None

# --- rate limiter ---------------------------------------------------------
# Logic lives in app/ratelimit.py so it can be unit-tested without FastAPI
# installed. See that module's docstring for the single-instance caveat.
_rate_limiter = RateLimiter(
    max_requests=config.RATE_LIMIT_MAX_REQUESTS,
    window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
)


def _ensure_engine() -> Optional[ConversationEngine]:
    global _engine, _kb_load_error
    if _engine is None and _kb_load_error is None:
        database.init_db()
        if config.LLM_PROVIDER_CONFIG_ERROR:
            logger.warning("LLM configuration: %s", config.LLM_PROVIDER_CONFIG_ERROR)
        try:
            entries = load_entries()
            _engine = ConversationEngine(entries)
            logger.info("Knowledge base loaded: %d entries", len(entries))
        except KnowledgeBaseError as exc:
            _kb_load_error = str(exc)
            logger.error("Failed to load knowledge base: %s", exc)
    return _engine


@app.on_event("startup")
def on_startup() -> None:
    _ensure_engine()


# --- schemas ---------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=100)
    # Kept for backwards compatibility with the previous API shape; the
    # server is now the source of truth for history via the database, so
    # this is accepted but not required.
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    suggestions: Optional[List[str]] = None


class ClearChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=100)


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., max_length=100)
    message_id: Optional[int] = None
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = Field(default=None, max_length=500)


# --- helpers ---------------------------------------------------------------
def _client_key(request: Request) -> str:
    if request.client:
        return request.client.host
    return "unknown"


def _check_rate_limit(request: Request) -> None:
    if not config.RATE_LIMIT_ENABLED:
        return
    if not _rate_limiter.allow(_client_key(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again shortly.",
        )


def _resolve_session_id(session_id: Optional[str], header_session_id: Optional[str]) -> str:
    """Accept the client's session id only if it looks like a generated
    token, otherwise mint a fresh one.

    Session ids are unauthenticated bearer values: whoever presents one
    gets that conversation's follow-up context. That is acceptable here
    because the frontend generates a random UUID per browser tab and no
    sensitive data is stored against it — but the charset/length check
    below stops trivially-guessable values (e.g. "1", "abc") from being
    used at all, which would otherwise let unrelated visitors collide into
    a shared conversation. See the README's Security Review for the
    residual risk."""
    candidate = session_id or header_session_id
    if candidate and _SESSION_ID_RE.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())


# --- routes ---------------------------------------------------------------
@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def serve_frontend_root():
    try:
        candidates = [
            Path(__file__).resolve().parent.parent / "index.html",
            Path("api/index.html"),
            Path("index.html"),
            config.PROJECT_ROOT / "index.html",
            Path(__file__).resolve().parents[2] / "index.html",
        ]
        for c in candidates:
            if c and c.exists():
                return HTMLResponse(content=c.read_bytes().decode("utf-8", errors="replace"), status_code=200)
        return HTMLResponse("<h1>WeMentors Chatbot</h1><p>Website frontend loaded successfully.</p>", status_code=200)
    except Exception as exc:
        return HTMLResponse(f"<h1>Error</h1><pre>{exc}</pre>", status_code=500)


@app.get("/wm_brand_logo.svg", include_in_schema=False)
def serve_logo():
    try:
        candidates = [
            config.PROJECT_ROOT / "wm_brand_logo.svg",
            Path("wm_brand_logo.svg"),
            Path(__file__).resolve().parents[2] / "wm_brand_logo.svg",
            Path("api/wm_brand_logo.svg"),
        ]
        for c in candidates:
            if c and c.exists():
                return HTMLResponse(content=c.read_bytes().decode("utf-8", errors="replace"), media_type="image/svg+xml")
    except Exception:
        pass
    return HTMLResponse("", status_code=404)


@app.get("/wm_brand_logo.png", include_in_schema=False)
@app.get("/download.png", include_in_schema=False)
def serve_logo_png():
    try:
        candidates = [
            config.PROJECT_ROOT / "wm_brand_logo.png",
            config.PROJECT_ROOT / "download.png",
            Path("wm_brand_logo.png"),
            Path("download.png"),
            Path(__file__).resolve().parent.parent / "wm_brand_logo.png",
            Path(__file__).resolve().parents[2] / "wm_brand_logo.png",
            Path("api/wm_brand_logo.png"),
        ]
        for c in candidates:
            if c and c.exists():
                return Response(content=c.read_bytes(), media_type="image/png")
    except Exception:
        pass
    return Response(b"", status_code=404)


@app.get("/health")
def health_check() -> Dict[str, object]:
    engine = _ensure_engine()
    active = llm.get_active_provider()
    is_llm_active = not isinstance(active, llm.NullProvider)
    payload: Dict[str, object] = {
        "status": "healthy" if engine is not None else "degraded",
        "service": "wementors-chatbot",
        "backend_running": True,
        "knowledge_base_loaded": engine is not None,
        "knowledge_base_entries": len(engine.entries) if engine else 0,
        "llm_enabled": is_llm_active,
        "llm_provider": active.name,
        "llm_provider_configured": config.LLM_PROVIDER,
        "llm_api_key_present": bool(config.GROQ_API_KEY if config.LLM_PROVIDER == "groq" else (config.ANTHROPIC_API_KEY if config.LLM_PROVIDER == "anthropic" else False)),
        "llm_model": (config.GROQ_MODEL if active.name == "groq" else (config.ANTHROPIC_MODEL if active.name == "anthropic" else None)),
        "response_mode": "ai" if is_llm_active else "deterministic_knowledge_base",
    }
    if config.LLM_PROVIDER_CONFIG_ERROR:
        payload["llm_config_warning"] = config.LLM_PROVIDER_CONFIG_ERROR
    return payload


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: Request,
    body: ChatRequest,
    x_session_id: Optional[str] = Header(default=None),
) -> ChatResponse:
    _check_rate_limit(request)

    session_id = _resolve_session_id(body.session_id, x_session_id)
    database.ensure_session(session_id)

    message = body.message.strip()
    if not message:
        return ChatResponse(reply="Please enter a question so I can help you.", session_id=session_id, intent="empty")

    engine = _ensure_engine()
    if engine is None:
        database.log_error("chat", _kb_load_error or "engine not initialized")
        return ChatResponse(
            reply="The WeMentors knowledge base is currently unavailable. Please contact the WeMentors team directly for assistance.",
            session_id=session_id,
            intent="system_error",
        )

    database.log_message(session_id, "user", message)

    t_start = time.perf_counter()
    try:
        result = engine.handle_message(session_id, message)
    except Exception as exc:  # noqa: BLE001 - must never leak internals to the visitor
        logger.exception("Unhandled error while generating a reply")
        database.log_error("chat.handle_message", repr(exc))
        return ChatResponse(
            reply="Something went wrong on my end. Please try again, or contact the WeMentors team directly.",
            session_id=session_id,
            intent="system_error",
        )

    duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "[TIMING] duration_ms=%.1f intent=%s matched_ids=%s chars=%d",
        duration_ms,
        result.intent,
        result.matched_entry_ids,
        len(result.reply),
    )

    database.log_message(
        session_id,
        "assistant",
        result.reply,
        intent=result.intent,
        matched_entry_ids=",".join(result.matched_entry_ids) if result.matched_entry_ids else None,
        confidence=result.confidence,
    )

    return ChatResponse(
        reply=result.reply,
        session_id=session_id,
        intent=result.intent,
        suggestions=result.suggestions,
    )


@app.post("/chat/clear")
def clear_chat(
    request: Request,
    body: ClearChatRequest,
    x_session_id: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    _check_rate_limit(request)
    resolved = _resolve_session_id(body.session_id, x_session_id)
    database.clear_session(resolved)
    return {"status": "cleared", "session_id": resolved}


@app.post("/feedback")
def submit_feedback(request: Request, body: FeedbackRequest) -> Dict[str, str]:
    _check_rate_limit(request)
    database.record_feedback(body.session_id, body.message_id, body.rating, body.comment)
    return {"status": "received"}


@app.get("/admin/analytics")
def admin_analytics(x_admin_key: Optional[str] = Header(default=None)) -> Dict[str, object]:
    if not config.ADMIN_API_KEY or x_admin_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=404, detail="Not found")
    return database.get_analytics_summary()


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_path = config.PROJECT_ROOT / "index.html"
    if not index_path.exists():
        index_path = config.PROJECT_ROOT.parent / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>WeMentors Chatbot API</h1>", status_code=200)


@app.get("/{filename:path}")
def serve_static(filename: str):
    allowed_files = {"wm_brand_logo.png", "wm_brand_logo.svg", "download.png", "favicon.ico"}
    if filename in allowed_files:
        fpath = config.PROJECT_ROOT / filename
        if not fpath.exists():
            fpath = config.PROJECT_ROOT.parent / filename
        if fpath.exists():
            return FileResponse(str(fpath))
    raise HTTPException(status_code=404, detail="Not found")


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception) -> UTF8JSONResponse:
    # Deliberately a sync `def`: this handler writes to SQLite, which is a
    # blocking call. Starlette runs sync exception handlers in a threadpool,
    # so the event loop is never blocked. Declaring it `async def` would
    # stall the loop for every other in-flight request.
    logger.exception("Unhandled exception on %s", request.url.path)
    database.log_error(str(request.url.path), repr(exc))
    return UTF8JSONResponse(status_code=500, content={"error": "An unexpected error occurred. Please try again."})
