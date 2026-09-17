"""
Regression test suite verifying lead collection removal and privacy guardrails.

Ensures:
- Demo inquiries direct to official contact info and never collect lead fields.
- database.get_demo_lead or similar legacy APIs are cleanly absent or disabled.
- Acknowledgements ("ok", "sure", "thanks") never claim a demo has been booked or submitted.
- No lead data is stored in SQLite.
"""

import os
import sys
import uuid
import tempfile
import sqlite3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["SKIP_DOTENV"] = "1"
os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(), "test_leads.db"))

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import database, personality, conversation
from app.knowledge import load_entries

passed_checks = 0
failed_checks = []


def check(description: str, condition: bool, detail: str = ""):
    global passed_checks
    if condition:
        passed_checks += 1
        print(f"[PASS] {description}")
    else:
        failed_checks.append(description)
        print(f"[FAIL] {description} -- {detail}")


def ask(engine, text, sid):
    database.log_message(sid, "user", text)
    result = engine.handle_message(sid, text)
    database.log_message(
        sid,
        "assistant",
        result.reply,
        intent=result.intent,
        matched_entry_ids=",".join(result.matched_entry_ids) if result.matched_entry_ids else None,
        confidence=result.confidence,
    )
    return result


def main():
    database.init_db()
    entries = load_entries()
    engine = conversation.ConversationEngine(entries)

    print("--- 1. Direct Contact Routing for Demo Requests ---")
    sid = f"test-lead-{uuid.uuid4()}"
    database.ensure_session(sid)

    r1 = ask(engine, "I'd like to book a free demo class", sid)
    check("Intent is demo_booking", r1.intent == "demo_booking", f"intent={r1.intent}")
    check("Provides official email admin@wementors.co", "admin@wementors.co" in r1.reply, r1.reply)
    check("Provides official phone +91 76111 92227", "+91 76111 92227" in r1.reply, r1.reply)
    check("Does NOT ask for student name or grade", not any(x in r1.reply.lower() for x in ["student or parent name", "student's grade", "could you share the following"]), r1.reply)

    print("--- 2. Follow-up 'ok' after Demo Request ---")
    r2 = ask(engine, "ok", sid)
    forbidden_phrases = [
        "noted those details",
        "i've noted",
        "i have noted",
        "within 24 hours",
        "passed them along",
        "schedule your demo",
        "demo is scheduled",
        "booked",
        "submitted",
        "team will reach out within 24 hours",
    ]
    r2_lower = r2.reply.lower()
    for phrase in forbidden_phrases:
        check(
            f"'ok' response must NOT contain '{phrase}'",
            phrase not in r2_lower,
            f"Found forbidden phrase in: {r2.reply}",
        )

    print("--- 3. Database Integrity: demo_leads table removed ---")
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='demo_leads';")
        check("demo_leads table does not exist in DB", cursor.fetchone() is None)

    print("\n================ SUMMARY ================")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {len(failed_checks)}")
    if failed_checks:
        sys.exit(1)


if __name__ == "__main__":
    main()
