"""
Regression test suite verifying that the demo booking flow is guidance-only.

The chatbot does NOT have demo_transaction_capability. Therefore:
- Demo intents provide official contact channels and website form guidance.
- The chatbot NEVER collects personal information through chat.
- The chatbot NEVER claims to note, save, submit, or confirm a demo booking.
- "yes" after demo context does NOT trigger a fake submission.
- No lead state is created in the database during normal demo conversations.
- Cancellation is handled gracefully.
- The "is my demo booked?" question gets an honest answer.
- The process_demo_flow capability gate works correctly.
"""

import os
import sys
import uuid
import tempfile

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

    # Verify the capability gate is disabled
    print("--- 0. Capability Gate ---")
    check("DEMO_TRANSACTION_ENABLED is False", leads.DEMO_TRANSACTION_ENABLED is False)

    print("\n--- 1. Direct Contact Routing for Demo Requests ---")
    sid1 = f"test-demo-init-{uuid.uuid4()}"
    database.ensure_session(sid1)

    r1 = ask(engine, "I'd like to book a free demo class", sid1)
    check("Intent is demo_booking", r1.intent == "demo_booking", f"intent={r1.intent}")
    check("Provides official email admin@wementors.co", "admin@wementors.co" in r1.reply, r1.reply)
    check("Provides official phone +91 76111 92227", "+91 76111 92227" in r1.reply, r1.reply)
    check("Does NOT ask for name/grade/phone", "what's your name" not in r1.reply.lower() and "student's grade" not in r1.reply.lower(), r1.reply)

    print("\n--- 2. Follow-up 'ok' after Demo Request (No Fake Confirmation) ---")
    r2 = ask(engine, "ok", sid1)
    forbidden_phrases = [
        "noted those details", "i've noted", "i have noted",
        "within 24 hours", "passed them along", "schedule your demo",
        "demo is scheduled", "your demo has been scheduled", "booked",
        "your information has been submitted", "submitted",
        "team will reach out within 24 hours",
    ]
    r2_lower = r2.reply.lower()
    for phrase in forbidden_phrases:
        check(f"'ok' response must NOT contain '{phrase}'", phrase not in r2_lower, f"Found in: {r2.reply}")

    print("\n--- 3. User Asks whether Demo is Booked ---")
    sid3 = f"test-demo-booked-{uuid.uuid4()}"
    database.ensure_session(sid3)
    ask(engine, "I want a demo", sid3)
    r3 = ask(engine, "Is my demo booked?", sid3)
    r3_lower = r3.reply.lower()
    check("Does NOT claim demo is booked", "your demo is booked" not in r3_lower and "demo is confirmed" not in r3_lower, r3.reply)
    check("Mentions Book Free Demo form", "book free demo" in r3_lower, r3.reply)

    print("\n--- 4. Providing Name after Demo Intent (No Fake Lead) ---")
    sid4 = f"test-name-after-demo-{uuid.uuid4()}"
    database.ensure_session(sid4)
    ask(engine, "book demo", sid4)
    r_name = ask(engine, "Priya", sid4)
    r_name_lower = r_name.reply.lower()
    check("Does NOT claim 'I noted your name'", "noted your name" not in r_name_lower and "saved your name" not in r_name_lower, r_name.reply)
    check("Does NOT ask for grade/subject/phone as next field", "student's grade" not in r_name_lower and "preferred subject" not in r_name_lower and "phone number" not in r_name_lower, r_name.reply)
    check("Guides to Book Free Demo", "book free demo" in r_name_lower, r_name.reply)
    lead_row = database.get_demo_lead(sid4)
    check("No lead created in database", lead_row is None or lead_row.get("name") is None, str(lead_row))

    print("\n--- 5. Grade/Subject/Phone do NOT Create Fake Leads ---")
    sid5 = f"test-fields-nolead-{uuid.uuid4()}"
    database.ensure_session(sid5)
    ask(engine, "book demo", sid5)

    r_grade = ask(engine, "Grade 7", sid5)
    check("Grade 7 does NOT get 'noted'", "noted" not in r_grade.reply.lower() and "saved" not in r_grade.reply.lower(), r_grade.reply)

    r_subj = ask(engine, "Maths", sid5)
    check("Maths does NOT get 'noted'", "noted" not in r_subj.reply.lower() and "saved" not in r_subj.reply.lower(), r_subj.reply)

    r_phone = ask(engine, "8085947527", sid5)
    check("Phone does NOT get 'noted'", "noted" not in r_phone.reply.lower() and "saved" not in r_phone.reply.lower(), r_phone.reply)

    lead_row5 = database.get_demo_lead(sid5)
    check("No lead state persisted after fields", lead_row5 is None or lead_row5.get("stage") not in ("collecting", "confirming", "submitted"), str(lead_row5))

    print("\n--- 6. 'Yes' after Demo Context Does NOT Trigger Fake Submission ---")
    r_yes = ask(engine, "yes", sid5)
    r_yes_lower = r_yes.reply.lower()
    check("'yes' does NOT claim submission", "submitted" not in r_yes_lower and "received" not in r_yes_lower and "confirmed" not in r_yes_lower, r_yes.reply)
    check("'yes' does NOT claim team will contact", "team will contact" not in r_yes_lower and "team will reach out" not in r_yes_lower, r_yes.reply)

    print("\n--- 7. 'Would you like me to submit?' is Never Generated ---")
    sid7 = f"test-no-submit-ask-{uuid.uuid4()}"
    database.ensure_session(sid7)
    r7a = ask(engine, "book demo", sid7)
    r7b = ask(engine, "My name is Raaid, Grade 7, Maths, 9876543210", sid7)
    check("Combined fields do NOT trigger 'submit' prompt", "would you like me to submit" not in r7b.reply.lower() and "shall i submit" not in r7b.reply.lower(), r7b.reply)
    check("Combined fields do NOT claim lead saved", "noted" not in r7b.reply.lower() and "saved" not in r7b.reply.lower(), r7b.reply)

    print("\n--- 8. Cancellation Handled Gracefully ---")
    sid8 = f"test-demo-cancel-{uuid.uuid4()}"
    database.ensure_session(sid8)
    ask(engine, "I want a demo", sid8)
    r_cancel = ask(engine, "never mind", sid8)
    check("Cancellation is graceful", "no problem" in r_cancel.reply.lower() or "feel free" in r_cancel.reply.lower(), r_cancel.reply)

    print("\n--- 9. process_demo_flow Capability Gate Works ---")
    sid9 = f"test-gate-{uuid.uuid4()}"
    mock_lead = leads.DemoLead(
        name="Rohan", grade="Grade 9", subject="Maths",
        contact_method="phone", contact_value="+91 99999 88888",
        preferred_time="Morning", stage="confirming", submission_status="pending",
    )
    # Even with a complete lead in confirming stage, saying "yes" must NOT submit
    reply, intent, updated_lead = leads.process_demo_flow(sid9, "yes", mock_lead)
    check("Capability gate blocks submission", "submitted" not in reply.lower() and "successfully" not in reply.lower(), reply)
    check("Updated lead is reset (idle)", updated_lead.stage == "idle", f"stage={updated_lead.stage}")

    # Cancellation through gate
    reply_c, intent_c, _ = leads.process_demo_flow(sid9, "cancel", mock_lead)
    check("Cancellation through gate works", intent_c == "demo_cancelled", f"intent={intent_c}")

    # "Is it booked?" through gate
    reply_b, intent_b, _ = leads.process_demo_flow(sid9, "Is my demo booked?", mock_lead)
    check("Booked query through gate redirects", "book free demo" in reply_b.lower(), reply_b)

    print("\n--- 10. Database Integrity: demo_leads table exists ---")
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
        print("ALL DEMO-FLOW AND CAPABILITY GATE TESTS PASSED!")


if __name__ == "__main__":
    main()
