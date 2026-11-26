"""Vercel entry point: exposes the FastAPI app in backend/ as a serverless function."""

import os
import sys
from pathlib import Path
from urllib.parse import parse_qs

from starlette.types import ASGIApp, Receive, Scope, Send

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

# On Vercel only /tmp is writable.
os.environ.setdefault("DATABASE_PATH", "/tmp/chatbot.db")
# Vercel sets X-Real-IP, so the rate limiter can key on the real visitor.
os.environ.setdefault("TRUST_PROXY_HEADERS", "true")

from app.main import app as fastapi_app  # noqa: E402


class VercelQueryPathMiddleware:
    """Restores the original URL path from the __path query parameter added by vercel.json."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            query = scope.get("query_string", b"").decode("utf-8")
            if "__path=" in query:
                values = parse_qs(query).get("__path")
                if values:
                    scope["path"] = "/" + values[0].lstrip("/")
        await self.app(scope, receive, send)


app = VercelQueryPathMiddleware(fastapi_app)
