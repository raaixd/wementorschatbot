"""Turn thumbs-down feedback into candidate evaluation cases.

Reads the chatbot SQLite database, finds every assistant reply a visitor rated
"down", rebuilds the user turns that led to it, and writes candidate cases in
the evaluation/questions/dataset.json shape.

Candidates are deliberately NOT ready to run: `expected_intent` and
`expected_keywords` are left empty and `needs_review` is true. A person decides
what the right answer is (and fixes the knowledge base if it has a gap) before
a case is copied into dataset.json.

    python scripts/export_feedback_cases.py --db path/to/chatbot.db \
        --out evaluation/cache/feedback_candidates.json

The output holds real visitor text, so the default location is gitignored.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "evaluation" / "cache" / "feedback_candidates.json"
CATEGORY = "category_k_user_feedback"


def build_candidates(conn: sqlite3.Connection, include_empty_comments: bool = True) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    ratings = conn.execute(
        "SELECT id, session_id, message_id, comment FROM feedback WHERE rating = 'down' ORDER BY id"
    ).fetchall()

    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for fb in ratings:
        if not include_empty_comments and not (fb["comment"] or "").strip():
            continue
        rated = _rated_message(conn, fb["session_id"], fb["message_id"])
        if rated is None or (fb["session_id"], rated["id"]) in seen:
            continue
        seen.add((fb["session_id"], rated["id"]))

        turns = [
            row["content"]
            for row in conn.execute(
                "SELECT content FROM messages WHERE session_id = ? AND id < ? AND role = 'user' ORDER BY id",
                (fb["session_id"], rated["id"]),
            )
        ]
        if not turns:
            continue
        cases.append(
            {
                "id": f"feedback_{fb['id']:04d}",
                "category": CATEGORY,
                "turns": turns,
                "expected_intent": None,
                "expected_keywords": [],
                "forbidden_keywords": [],
                "requires_honest_fallback": False,
                "demo_safe": True,
                "description": "Visitor rated this reply thumbs-down. Label expected_* before use.",
                "needs_review": True,
                "observed": {
                    "bot_reply": rated["content"],
                    "intent": rated["intent"],
                    "matched_entry_ids": rated["matched_entry_ids"],
                    "confidence": rated["confidence"],
                    "visitor_comment": fb["comment"],
                },
            }
        )
    return cases


def _rated_message(conn: sqlite3.Connection, session_id: str, message_id: int | None):
    """The assistant message the rating refers to, falling back to the
    session's latest assistant reply when the client sent no message id."""
    if message_id is not None:
        row = conn.execute(
            "SELECT * FROM messages WHERE id = ? AND session_id = ? AND role = 'assistant'",
            (message_id, session_id),
        ).fetchone()
        if row:
            return row
    return conn.execute(
        "SELECT * FROM messages WHERE session_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", required=True, type=Path, help="path to chatbot.db")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only-with-comment", action="store_true", help="skip ratings without a written comment")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        cases = build_candidates(conn, include_empty_comments=not args.only_with_comment)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(cases)} candidate case(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
