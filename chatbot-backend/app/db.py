from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .config import settings

_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: Optional[Path] = None) -> None:
    db_path = path or settings.database_path
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                state_json TEXT NOT NULL DEFAULT '{}',
                message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                event_type TEXT NOT NULL,
                intent TEXT,
                detail TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, created_at);
            """
        )
        conn.commit()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    with _lock:
        conn = _connect(settings.database_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def ensure_session(session_id: str) -> None:
    now = utc_now()
    with get_conn() as conn:
        existing = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if existing:
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            return
        conn.execute(
            "INSERT INTO sessions (id, created_at, updated_at, state_json) VALUES (?, ?, ?, '{}')",
            (session_id, now, now),
        )


def load_state(session_id: str) -> Dict:
    with get_conn() as conn:
        row = conn.execute("SELECT state_json FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["state_json"] or "{}")
        except json.JSONDecodeError:
            return {}


def save_state(session_id: str, state: Dict) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET state_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(state), utc_now(), session_id),
        )


def add_message(session_id: str, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, utc_now()),
        )
        conn.execute(
            "UPDATE sessions SET message_count = message_count + 1, updated_at = ? WHERE id = ?",
            (utc_now(), session_id),
        )


def recent_messages(session_id: str, limit: int) -> List[Dict[str, str]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def clear_session(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute(
            "UPDATE sessions SET state_json = '{}', message_count = 0, updated_at = ? WHERE id = ?",
            (utc_now(), session_id),
        )


def log_event(session_id: Optional[str], event_type: str, intent: str = "", detail: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO events (session_id, event_type, intent, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, event_type, intent, detail[:180], utc_now()),
        )


def db_ok() -> bool:
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except sqlite3.Error:
        return False
