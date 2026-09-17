"""
Lightweight SQLite persistence layer.

Why SQLite: the project runs on a single local/small server, has no
concurrent-write scale requirements, and SQLite needs zero extra
infrastructure (no separate DB server, no extra deployment step) while
still giving us real persistence across backend restarts. If the project
ever needs multi-instance deployment, swap get_connection() for a
different driver — every other module only talks to this file.

Tables
------
sessions   - one row per browser chat session
messages   - every user/assistant message, linked to a session
feedback   - optional thumbs up/down on a specific assistant message
error_logs - server-side errors, for debugging without exposing internals
             to the visitor
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    intent TEXT,
    matched_entry_ids TEXT,
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_id INTEGER,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback (session_id);

CREATE TABLE IF NOT EXISTS error_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_error_logs_created ON error_logs (created_at);
"""

_EMAIL_REDACT_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_REDACT_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]*)?\(?\d{3,5}\)?[-.\s]*\d{3,5}[-.\s]*\d{3,5}")


def _redact_pii(text: str) -> str:
    """Mask personal email addresses and phone numbers before writing to logs."""
    if not text:
        return text
    redacted = _EMAIL_REDACT_RE.sub("[EMAIL REDACTED]", text)
    redacted = _PHONE_REDACT_RE.sub("[PHONE REDACTED]", redacted)
    return redacted


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create the database file and tables if they don't exist yet."""
    config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
        # Ensure demo_leads table is removed if it previously existed
        conn.execute("DROP TABLE IF EXISTS demo_leads")
        conn.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a short-lived connection. SQLite connections are cheap; we
    open/close per request instead of holding one open for the process
    lifetime, which keeps behavior predictable across reloads and avoids
    locking issues with the --reload dev server."""
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def ensure_session(session_id: str) -> None:
    now = _now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, created_at, last_active_at)
            VALUES (?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET last_active_at = excluded.last_active_at
            """,
            (session_id, now, now),
        )
        conn.commit()


def log_message(
    session_id: str,
    role: str,
    content: str,
    intent: Optional[str] = None,
    matched_entry_ids: Optional[str] = None,
    confidence: Optional[float] = None,
) -> int:
    safe_content = _redact_pii(content)
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (session_id, role, content, intent, matched_entry_ids, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, safe_content, intent, matched_entry_ids, confidence, _now()),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_recent_messages(session_id: str, limit: int) -> list[sqlite3.Row]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, intent, matched_entry_ids, confidence, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return list(reversed(rows))


def clear_session(session_id: str) -> None:
    """Remove stored messages for a session (used by Clear Chat)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()


def record_feedback(session_id: str, message_id: Optional[int], rating: str, comment: Optional[str]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (session_id, message_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, message_id, rating, comment, _now()),
        )
        conn.commit()


def log_error(context: str, message: str) -> None:
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO error_logs (context, message, created_at) VALUES (?, ?, ?)",
                (context, message[:2000], _now()),
            )
            conn.commit()
    except sqlite3.Error:
        # Logging must never itself crash the request.
        pass


def get_analytics_summary(days: int = 30) -> dict:
    """Simple aggregate used by the admin analytics endpoint."""
    with get_connection() as conn:
        total_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
        total_messages = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE role = 'user'"
        ).fetchone()["c"]
        top_intents = conn.execute(
            """
            SELECT intent, COUNT(*) AS c
            FROM messages
            WHERE role = 'user' AND intent IS NOT NULL
            GROUP BY intent
            ORDER BY c DESC
            LIMIT 10
            """
        ).fetchall()
        low_confidence_count = conn.execute(
            """
            SELECT COUNT(*) AS c FROM messages
            WHERE role = 'assistant' AND confidence IS NOT NULL AND confidence < ?
            """,
            (config.RETRIEVAL_CONFIDENCE_THRESHOLD,),
        ).fetchone()["c"]
        return {
            "total_sessions": total_sessions,
            "total_user_messages": total_messages,
            "top_intents": [dict(row) for row in top_intents],
            "low_confidence_replies": low_confidence_count,
        }
