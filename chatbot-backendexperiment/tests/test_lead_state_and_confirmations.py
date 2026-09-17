"""
Regression test suite verifying demo flow state machine, false-confirmation prevention,
and structured lead handling (Section 5 and Section 12 requirements).

Covers:
- Demo inquiries provide official contact channels (email/phone).
- User replies only "ok" after being asked for details -> no false confirmation, friendly prompt.
- User provides only a name -> acknowledged, asks for missing fields.
- User provides name and grade -> acknowledged, asks for missing fields.
- User provides incomplete details in multiple messages -> state accumulates.
- User provides all required details -> summary presented, asks for confirmation.
- User changes a previously provided detail -> detail updated.
- User refuses to provide a detail / cancels -> cancelled gracefully.
- Backend submission succeeds -> confirmation stated only after successful submission.
- Backend submission fails -> failure stated honestly without claiming submission.
- User asks whether the demo is booked before submission is confirmed -> clarifies not yet booked.
- Database integrity: demo_leads table exists and stores structured fields.
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

from app import database, personality, conversation, leads
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
    sid1 = f"test-demo-init-{uuid.uuid4()}"
    database.ensure_session(sid1)

    r1 = ask(engine, "I'd like to book a free demo class", sid1)
    check("Intent is demo_booking", r1.intent == "demo_booking", f"intent={r1.intent}")
    check("Provides official email admin@wementors.co", "admin@wementors.co" in r1.reply, r1.reply)
    check("Provides official phone +91 76111 92227", "+91 76111 92227" in r1.reply, r1.reply)

    print("\n--- 2. Follow-up 'ok' after Demo Request (False Confirmation Prevention) ---")
    r2 = ask(engine, "ok", sid1)
    forbidden_phrases = [
        "noted those details",
        "i've noted",
        "i have noted",
        "within 24 hours",
        "passed them along",
        "schedule your demo",
        "demo is scheduled",
        "your demo has been scheduled",
        "booked",
        "your information has been submitted",
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
    check("'ok' responds naturally inviting details", "ready" in r2_lower or "send the details" in r2_lower or "share" in r2_lower, r2.reply)

    print("\n--- 3. User Asks whether Demo is Booked before Submission is Confirmed ---")
    r3 = ask(engine, "Is my demo booked?", sid1)
    check("Explains demo is not yet booked/submitted", "not yet" in r3.reply.lower() or "not been submitted" in r3.reply.lower(), r3.reply)

    print("\n--- 4. User Provides Only Name ---")
    sid2 = f"test-demo-steps-{uuid.uuid4()}"
    database.ensure_session(sid2)
    ask(engine, "I want to enroll in a demo", sid2)
    r_name = ask(engine, "My name is Priya Sharma", sid2)
    check("Acknowledges name Priya", "Priya" in r_name.reply or "name" in r_name.reply.lower(), r_name.reply)
    check("Asks for missing details (grade, subject, contact)", "grade" in r_name.reply.lower(), r_name.reply)

    print("\n--- 5. User Provides Incomplete Details Across Multiple Turns ---")
    r_grade = ask(engine, "She is in Grade 7", sid2)
    check("Noted Grade 7", "7" in r_grade.reply, r_grade.reply)
    check("Prompts for subject or contact", "subject" in r_grade.reply.lower() or "contact" in r_grade.reply.lower() or "email" in r_grade.reply.lower(), r_grade.reply)

    r_subj = ask(engine, "Interested in Maths", sid2)
    check("Noted Maths", "Maths" in r_subj.reply or "math" in r_subj.reply.lower(), r_subj.reply)

    print("\n--- 6. User Completes Required Details -> Confirmation Stage ---")
    r_contact = ask(engine, "My email is priya.sharma@example.com and time is 5 PM", sid2)
    check("Enters confirming stage", "submit" in r_contact.reply.lower() or "confirm" in r_contact.reply.lower(), r_contact.reply)
    check("Summary contains Priya", "Priya" in r_contact.reply, r_contact.reply)
    check("Summary contains Grade 7", "7" in r_contact.reply, r_contact.reply)
    check("Summary contains Maths", "Maths" in r_contact.reply or "Math" in r_contact.reply, r_contact.reply)
    check("Summary contains email", "priya.sharma@example.com" in r_contact.reply, r_contact.reply)

    print("\n--- 7. User Changes a Previously Provided Detail ---")
    r_change = ask(engine, "Actually make the subject Science instead of Maths", sid2)
    check("Updates subject to Science", "Science" in r_change.reply, r_change.reply)
    check("Re-confirms submission readiness", "submit" in r_change.reply.lower(), r_change.reply)

    print("\n--- 8. User Confirms Submission (Backend Success) ---")
    r_submit = ask(engine, "Yes, please submit it", sid2)
    check("States demo request submitted successfully", "submitted" in r_submit.reply.lower() or "received" in r_submit.reply.lower(), r_submit.reply)
    check("Mentions contact value for confirmation", "priya.sharma@example.com" in r_submit.reply, r_submit.reply)

    # Verify in DB
    lead_row = database.get_demo_lead(sid2)
    check("Lead stored in DB with status confirmed", lead_row is not None and lead_row.get("submission_status") == "confirmed", str(lead_row))
    check("Lead name stored correctly", lead_row.get("name") == "Priya Sharma", str(lead_row))

    print("\n--- 9. Backend Submission Failure Handling ---")
    sid3 = f"test-demo-fail-{uuid.uuid4()}"
    database.ensure_session(sid3)
    # Directly test process_demo_flow with failure override
    mock_lead = leads.DemoLead(
        name="Rohan",
        grade="Grade 9",
        subject="Maths",
        contact_method="phone",
        contact_value="+91 99999 88888",
        preferred_time="Morning",
        stage="confirming",
        submission_status="pending",
    )
    fail_reply, fail_intent, _ = leads.process_demo_flow(sid3, "yes", mock_lead, backend_submit_override=False)
    check("Submission failure states honest error message", "issue" in fail_reply.lower() or "problem" in fail_reply.lower(), fail_reply)
    check("Submission failure does NOT claim booking succeeded", "successfully" not in fail_reply.lower(), fail_reply)

    print("\n--- 10. User Cancels / Refuses Detail ---")
    sid4 = f"test-demo-cancel-{uuid.uuid4()}"
    database.ensure_session(sid4)
    ask(engine, "I want a demo", sid4)
    r_cancel = ask(engine, "Actually never mind, cancel that", sid4)
    check("Cancellation handled gracefully", "cancelled" in r_cancel.reply.lower() or "no problem" in r_cancel.reply.lower(), r_cancel.reply)
    check("Provides direct team contact on cancel", "admin@wementors.co" in r_cancel.reply, r_cancel.reply)

    print("\n--- 11. Database Integrity: demo_leads table exists & stores structured fields ---")
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='demo_leads';")
        check("demo_leads table exists in DB", cursor.fetchone() is not None)

    print("\n================ SUMMARY ================")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {len(failed_checks)}")
    if failed_checks:
        for f in failed_checks:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL DEMO-FLOW AND FALSE CONFIRMATION TESTS PASSED!")


if __name__ == "__main__":
    main()
