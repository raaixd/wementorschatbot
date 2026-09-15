from pathlib import Path
from typing import List

from dotenv import load_dotenv
import os

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(BACKEND_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env")


def _csv(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    app_name = "WeMentors Chatbot API"
    app_version = "2.0.0"
    environment = os.getenv("ENVIRONMENT", "development")

    allowed_origins = _csv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8080,http://localhost:8080",
    )
    knowledge_path = Path(
        os.getenv("KNOWLEDGE_PATH", PROJECT_ROOT / "knowledge" / "wementors_kb.json")
    )
    database_path = Path(os.getenv("DATABASE_PATH", BACKEND_ROOT / "data" / "chatbot.db"))

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "12"))

    max_message_length = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
    history_limit = int(os.getenv("HISTORY_LIMIT", "16"))
    retrieve_top_k = int(os.getenv("RETRIEVE_TOP_K", "4"))
    min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", "4.0"))

    rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "40"))
    rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


settings = Settings()
