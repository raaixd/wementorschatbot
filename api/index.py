import os
import sys
from pathlib import Path
from urllib.parse import parse_qs
from starlette.types import ASGIApp, Receive, Scope, Send

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
backend_dir = root_dir / "chatbot-backendexperiment"

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# On Vercel serverless functions, SQLite must live in /tmp
if "DATABASE_PATH" not in os.environ:
    os.environ["DATABASE_PATH"] = "/tmp/chatbot.db"

# Point knowledge base to local packaged copy
local_kb = current_dir / "wementors_kb.json"
if local_kb.exists():
    os.environ["KNOWLEDGE_FILE"] = str(local_kb)
elif "KNOWLEDGE_FILE" not in os.environ:
    kb_path = root_dir / "knowledge" / "wementors_kb.json"
    if kb_path.exists():
        os.environ["KNOWLEDGE_FILE"] = str(kb_path)

from app.main import app as fastapi_app


class VercelQueryPathMiddleware:
    """Restores the exact original URL path from the __path query parameter added by Vercel rewrites."""
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            qs = scope.get("query_string", b"").decode("utf-8")
            if "__path=" in qs:
                params = parse_qs(qs)
                if "__path" in params and params["__path"]:
                    target_path = params["__path"][0]
                    if not target_path.startswith("/"):
                        target_path = "/" + target_path
                    while target_path.startswith("//"):
                        target_path = target_path[1:]
                    scope["path"] = target_path
        await self.app(scope, receive, send)


app = VercelQueryPathMiddleware(fastapi_app)
