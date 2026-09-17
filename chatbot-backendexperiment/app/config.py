"""
Centralized configuration for the WeMentors chatbot backend.

Everything here is read from environment variables (optionally loaded from
a local .env file via python-dotenv) so nothing sensitive is hardcoded.
See .env.example for the full list of supported variables.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

try:
    from dotenv import load_dotenv

    # Load a .env file if one exists in the backend or project root (never committed to git).
    if not os.getenv("SKIP_DOTENV"):
        if (BACKEND_ROOT / ".env").exists():
            load_dotenv(BACKEND_ROOT / ".env")
        elif (PROJECT_ROOT / ".env").exists():
            load_dotenv(PROJECT_ROOT / ".env")
        else:
            load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt,
    # but degrade gracefully rather than crash if it's somehow missing.
    pass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_path(name: str, default: Path) -> Path:
    """Resolve a path setting to an absolute path, portably.

    Handles three things that bit us in practice:

    * Windows-style separators. A .env written on Windows may say
      ``..\\knowledge\\wementors_kb.json``; on Linux that is one filename
      containing backslashes, not a path, so separators are normalised.
    * Ambiguous roots. Existing .env files were written assuming the
      backend directory as the base ("..\\knowledge\\..."), while others
      assume the project root ("knowledge/..."). Both are tried and the
      one that actually resolves to an existing file wins.
    * Current working directory. Relative values are never resolved
      against the shell's CWD, so the backend behaves identically whether
      it is started from the project root or from chatbot-backend/.
    """
    value = os.getenv(name)
    if not value:
        return default

    raw = value.strip().replace("\\", "/")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    candidates = [(PROJECT_ROOT / path), (BACKEND_ROOT / path)]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    # Not created yet (e.g. the SQLite file on first run), so existence
    # can't disambiguate. Resolve against the backend directory, since
    # .env lives there and its paths are written relative to it — unless
    # the value already starts with the backend directory name, which
    # means it is project-root relative (as in .env.example).
    if path.parts and path.parts[0] in {BACKEND_ROOT.name, "chatbot-backend", "chatbot-backendexperiment"}:
        return (BACKEND_ROOT / Path(*path.parts[1:])).resolve()
    return (BACKEND_ROOT / path).resolve()


# --- Knowledge base -------------------------------------------------------
KNOWLEDGE_FILE = _get_path(
    "KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge" / "wementors_kb.json"
)

# --- Database ---------------------------------------------------------------
# SQLite is used for local development. Point DATABASE_PATH somewhere else
# (or swap the connection helper in database.py) for a different backend.
DATABASE_PATH = _get_path("DATABASE_PATH", BACKEND_ROOT / "data" / "chatbot.db")

# --- CORS ---------------------------------------------------------------
# In production, set ALLOWED_ORIGINS to a comma-separated list of the
# real website origin(s), e.g. "https://wementors.co,https://www.wementors.co"
ALLOWED_ORIGINS = _get_list(
    "ALLOWED_ORIGINS",
    [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
)

# --- Rate limiting --------------------------------------------------------
RATE_LIMIT_ENABLED = _get_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_MAX_REQUESTS = _get_int("RATE_LIMIT_MAX_REQUESTS", 20)
RATE_LIMIT_WINDOW_SECONDS = _get_int("RATE_LIMIT_WINDOW_SECONDS", 60)

# --- Conversation management ---------------------------------------------
# How many recent messages the engine reads back when resolving a
# follow-up ("the first one", "its fees"). Keeps context bounded.
MAX_HISTORY_MESSAGES = _get_int("MAX_HISTORY_MESSAGES", 20)

# --- Retrieval ---------------------------------------------------------
RETRIEVAL_TOP_K = _get_int("RETRIEVAL_TOP_K", 3)
RETRIEVAL_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.12"))

# --- Optional LLM answer generation ---------------------------------------
# When no provider key is set, the chatbot falls back to deterministic
# template answers built from the knowledge base (no external call, works
# fully offline). Groq is checked first, then Anthropic.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")


_VALID_PROVIDERS = ("none", "groq", "anthropic")

_PROVIDER_KEYS = {"groq": GROQ_API_KEY, "anthropic": ANTHROPIC_API_KEY}


def _select_provider() -> tuple[str, str]:
    """Resolve the active provider and any configuration problem found.

    Returns (provider, config_error). A provider is only selected when its
    API key is actually present: asking for a provider without its key is
    a misconfiguration, and silently reporting the LLM as "enabled" while
    every request falls back to templates hides the problem from whoever
    deployed it. In that case we return "none" plus a message, so startup
    logging and /health can state the truth.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()

    if explicit:
        if explicit not in _VALID_PROVIDERS:
            return "none", (
                f"LLM_PROVIDER={explicit!r} is not recognised "
                f"(expected one of: {', '.join(_VALID_PROVIDERS)}). "
                "Falling back to knowledge-base answers."
            )
        if explicit == "none":
            return "none", ""
        if not _PROVIDER_KEYS[explicit]:
            return "none", (
                f"LLM_PROVIDER={explicit!r} was requested but {explicit.upper()}_API_KEY "
                "is empty. Falling back to knowledge-base answers."
            )
        return explicit, ""

    if GROQ_API_KEY:
        return "groq", ""
    if ANTHROPIC_API_KEY:
        return "anthropic", ""
    return "none", ""


LLM_PROVIDER, LLM_PROVIDER_CONFIG_ERROR = _select_provider()
LLM_ENABLED = LLM_PROVIDER != "none"
LLM_TIMEOUT_SECONDS = _get_int("LLM_TIMEOUT_SECONDS", 4)
LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 1000)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# How many prior turns are replayed to the model so follow-ups resolve.
LLM_HISTORY_TURNS = _get_int("LLM_HISTORY_TURNS", 6)

# --- Admin / analytics ----------------------------------------------------
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()

# --- Logging ---------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
