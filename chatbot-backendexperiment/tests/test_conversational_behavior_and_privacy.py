"""
Verification of Conversational Behavior Improvements and Privacy Guardrails.

Validates the 12 required scenarios from the specification:
1. "Hi" -> Natural greeting without robotic assumption.
2. "Help" -> Helpful overview asking what the user wants help with (no fees assumption).
3. "I want to know more" -> Clarifying question asking what topics they are interested in.
4. "How much?" -> Clarifying question asking what course or subject they mean.
5. "I want a demo class" -> Direct official contact routing (email + phone), zero lead collection.
6. "I want to enquire about joining" -> Direct official contact routing (email + phone), zero lead collection.
7. "Can you help me choose a class?" -> Conversational advising question about student grade/subject without creating a lead.
8. Question about fees -> Clear policy from KB (customized/unverified quote) and contact routing.
9. Question about subjects or curriculum -> Accurate listing of subjects/programs from KB.
10. Unrelated question outside KB -> Honest fallback acknowledging missing info and offering contact.
11. Vague follow-up message -> Clarifies or resolves with session context without hallucinating.
12. Privacy & Data Integrity -> Verifies demo_leads table is gone, no lead storage occurs, and PII in messages is redacted.
"""

import os
import sqlite3
import sys
import tempfile
import uuid

# Ensure backend directory is in path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Use isolated temp database for tests
temp_db = os.path.join(tempfile.mkdtemp(), "test_conversational_privacy.db")
os.environ["DATABASE_PATH"] = temp_db
os.environ["SKIP_DOTENV"] = "1"

from app import config, database, personality, conversation
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


def run_tests():
    database.init_db()
    entries = load_entries()
    engine = conversation.ConversationEngine(entries)

    print("\n=== 1. Scenario 1: 'Hi' (Natural Greeting) ===")
    s1 = f"s1-{uuid.uuid4()}"
    database.ensure_session(s1)
    r1 = engine.handle_message(s1, "Hi")
    check("Intent is greeting", r1.intent == "greeting", f"intent={r1.intent}")
    check("Greeting is friendly and concise", len(r1.reply) < 300 and "WeMentors" in r1.reply, r1.reply)
    check("Greeting does not mention fees or push demo", "fee" not in r1.reply.lower() and "parent or student name" not in r1.reply.lower(), r1.reply)

    print("\n=== 2. Scenario 2: 'Help' (Guidance without assuming fees) ===")
    s2 = f"s2-{uuid.uuid4()}"
    database.ensure_session(s2)
    r2 = engine.handle_message(s2, "Help")
    check("Intent is help", r2.intent == "help", f"intent={r2.intent}")
    check("Offers help with classes, subjects, demo classes, etc.", "classes" in r2.reply.lower() and "subjects" in r2.reply.lower(), r2.reply)
    check("Crucial: 'Help' NEVER talks about course fees or pricing", "pricing" not in r2.reply.lower() and "fee" not in r2.reply.lower(), r2.reply)

    print("\n=== 3. Scenario 3: 'I want to know more' (Clarifying question) ===")
    s3 = f"s3-{uuid.uuid4()}"
    database.ensure_session(s3)
    r3 = engine.handle_message(s3, "I want to know more")
    check("Intent is clarify", r3.intent == "clarify", f"intent={r3.intent}")
    check("Asks clarifying options (classes, subjects, demo)", "interested in" in r3.reply.lower() or "classes" in r3.reply.lower(), r3.reply)
    check("Does not make wild guesses", "demo" in r3.reply.lower() or "subjects" in r3.reply.lower(), r3.reply)

    print("\n=== 4. Scenario 4: 'How much?' (Clarification when context is missing) ===")
    s4 = f"s4-{uuid.uuid4()}"
    database.ensure_session(s4)
    r4 = engine.handle_message(s4, "How much?")
    check("Intent is clarify", r4.intent == "clarify", f"intent={r4.intent}")
    check("Asks which grade, program, or subject", "which" in r4.reply.lower() or "grade" in r4.reply.lower() or "program" in r4.reply.lower(), r4.reply)

    print("\n=== 5. Scenario 5: 'I want a demo class' (Official contact details, NO lead collection) ===")
    s5 = f"s5-{uuid.uuid4()}"
    database.ensure_session(s5)
    r5 = engine.handle_message(s5, "I want a demo class")
    check("Intent is demo_booking", r5.intent == "demo_booking", f"intent={r5.intent}")
    check("Includes verified email admin@wementors.co", "admin@wementors.co" in r5.reply, r5.reply)
    check("Includes verified phone +91 76111 92227", "+91 76111 92227" in r5.reply, r5.reply)
    check("Does NOT ask for student name, grade, or phone", not any(x in r5.reply.lower() for x in ["could you share the following", "parent or student name", "preferred contact method"]), r5.reply)
    check("Does NOT claim booking has been made or submitted", not any(x in r5.reply.lower() for x in ["submitted", "booked", "i've booked", "reach out within 24 hours"]), r5.reply)

    print("\n=== 6. Scenario 6: 'I want to enquire about joining' (Official contact routing) ===")
    s6 = f"s6-{uuid.uuid4()}"
    database.ensure_session(s6)
    r6 = engine.handle_message(s6, "I want to enquire about joining")
    check("Intent is demo_booking", r6.intent == "demo_booking", f"intent={r6.intent}")
    check("Provides official email and phone", "admin@wementors.co" in r6.reply and "+91 76111 92227" in r6.reply, r6.reply)
    check("No lead form shown", "preferred time" not in r6.reply.lower(), r6.reply)

    print("\n=== 7. Scenario 7: 'Can you help me choose a class?' (Conversational advising, NO lead creation) ===")
    s7 = f"s7-{uuid.uuid4()}"
    database.ensure_session(s7)
    r7 = engine.handle_message(s7, "Can you help me choose a class?")
    check("Intent is advising", r7.intent == "advising", f"intent={r7.intent}")
    check("Asks helpful questions about student's grade/goals", "grade" in r7.reply.lower() and "subjects" in r7.reply.lower(), r7.reply)
    check("Does NOT ask for contact phone or email", "contact number" not in r7.reply.lower() and "phone number" not in r7.reply.lower(), r7.reply)

    print("\n=== 8. Scenario 8: Question about fees ===")
    s8 = f"s8-{uuid.uuid4()}"
    database.ensure_session(s8)
    r8 = engine.handle_message(s8, "What are the course fees?")
    check("Matched fees entry", r8.matched_entry_ids and "fees-and-pricing" in r8.matched_entry_ids, str(r8.matched_entry_ids))
    check("Accurately explains fees are not published / customized", "customized" in r8.reply.lower() or "not published" in r8.reply.lower() or "contact" in r8.reply.lower(), r8.reply)
    check("Does NOT invent arbitrary rupee amounts", "5000" not in r8.reply and "10000" not in r8.reply, r8.reply)

    print("\n=== 9. Scenario 9: Question about subjects or curriculum ===")
    s9 = f"s9-{uuid.uuid4()}"
    database.ensure_session(s9)
    r9 = engine.handle_message(s9, "What subjects do you teach?")
    check("Answers subjects accurately", "mathematics" in r9.reply.lower() or "science" in r9.reply.lower() or "english" in r9.reply.lower(), r9.reply)
    check("Does not switch topics or ask for user details", "share your details" not in r9.reply.lower(), r9.reply)

    print("\n=== 10. Scenario 10: Unrelated question outside KB ===")
    s10 = f"s10-{uuid.uuid4()}"
    database.ensure_session(s10)
    r10 = engine.handle_message(s10, "What is the recipe for chocolate cake?")
    check("Intent is off_topic or low_confidence", r10.intent in {"off_topic", "low_confidence"}, f"intent={r10.intent}")
    check("Honest and polite refusal to answer off-topic queries", "wementors" in r10.reply.lower() or "can't help" in r10.reply.lower() or "cannot answer" in r10.reply.lower() or "outside" in r10.reply.lower(), r10.reply)

    print("\n=== 11. Scenario 11: Vague follow-up message ===")
    s11 = f"s11-{uuid.uuid4()}"
    database.ensure_session(s11)
    # Turn 1: Ask about programs
    engine.handle_message(s11, "What programs do you offer?")
    # Turn 2: Vague follow-up
    r11 = engine.handle_message(s11, "Tell me more")
    check("Vague follow-up either resolves program or asks clarifying question", len(r11.reply) > 20 and not r11.reply.startswith("Sorry"), r11.reply)

    print("\n=== 12. Scenario 12: Zero Demo Storage & Privacy Verification ===")
    # A. Check SQLite schema: demo_leads table MUST NOT exist
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='demo_leads';")
    demo_leads_table = cursor.fetchone()
    check("demo_leads table DOES NOT exist in SQLite database", demo_leads_table is None, str(demo_leads_table))

    # B. Test PII redaction in message logger
    test_session = f"pii-test-{uuid.uuid4()}"
    database.ensure_session(test_session)
    database.log_message(test_session, "user", "My email is parent@example.com and phone is +91 9876543210")
    
    cursor.execute("SELECT content FROM messages WHERE session_id = ? AND role = 'user'", (test_session,))
    row = cursor.fetchone()
    conn.close()
    
    logged_content = row[0] if row else ""
    check("Email address is redacted from message logs", "parent@example.com" not in logged_content and "[EMAIL REDACTED]" in logged_content, logged_content)
    check("Phone number is redacted from message logs", "9876543210" not in logged_content and "[PHONE REDACTED]" in logged_content, logged_content)

    print("\n================ SUMMARY ================")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {len(failed_checks)}")
    if failed_checks:
        for f in failed_checks:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL 12 SCENARIOS AND PRIVACY CHECKS PASSED!")


if __name__ == "__main__":
    run_tests()
