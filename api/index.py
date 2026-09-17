import os
import sys
from pathlib import Path

# Add project root and backend folder to sys.path so imports resolve seamlessly on Vercel
root_dir = Path(__file__).resolve().parents[1]
backend_dir = root_dir / "chatbot-backendexperiment"

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# On Vercel serverless environments, the filesystem is read-only except for /tmp
if "DATABASE_PATH" not in os.environ:
    os.environ["DATABASE_PATH"] = "/tmp/chatbot.db"

# Point knowledge base to project root location if not configured
if "KNOWLEDGE_BASE_PATH" not in os.environ:
    kb_path = root_dir / "knowledge" / "wementors_kb.json"
    if kb_path.exists():
        os.environ["KNOWLEDGE_BASE_PATH"] = str(kb_path)

from app.main import app
