"""
Comprehensive Regression Test Suite for:
1. Eliminating False Demo Bookings & Fake Transactional Behavior (Parts 1-8, 17)
2. Natural Language Understanding & Answer Quality (Parts 9-15, 18)
"""

import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path so app imports cleanly
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import database, personality
from app.conversation import ConversationEngine
from app.knowledge import load_entries

failures = []
passed_count = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed_count
    if condition:
        passed_count += 1
        print(f"[PASS] {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"[FAIL] {name} -> {detail}")


def ask(engine, sid, text):
    database.log_message(sid, "user", text, source="text_input")
    res = engine.handle_message(sid, text, source="text_input", role="user")
    database.log_message(
        sid,
        "assistant",
        res.reply,
        intent=res.intent,
        matched_entry_ids=",".join(res.matched_entry_ids) if res.matched_entry_ids else None,
        confidence=res.confidence,
        source="assistant_response",
    )
    return res


def run_tests():
    database.init_db()
    entries = load_entries()
    engine = ConversationEngine(entries)

    print("\n" + "=" * 80)
    print("RUNNING FALSE DEMO BOOKING ELIMINATION & NATURAL LANGUAGE TESTS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: The Core Broken Behavior: "book demo" -> "name"
    # -------------------------------------------------------------------------
    print("\n--- 1. 'book demo' -> 'name' Must Never Enter a Fake Form ---")
    sid = f"test-demo-name-{uuid.uuid4()}"
    database.ensure_session(sid)

    r1 = ask(engine, sid, "book demo")
    check("1a. 'book demo' gives booking instructions", "book free demo" in r1.reply.lower() or "button" in r1.reply.lower(), r1.reply)
    check("1a. Intent is demo_booking", r1.intent == "demo_booking", r1.intent)

    r2 = ask(engine, sid, "name")
    r2_lower = r2.reply.lower()
    check("1b. 'name' does NOT treat 'name' as a person's name", "name: name" not in r2_lower and "noted your name: name" not in r2_lower, r2.reply)
    check("1b. 'name' does NOT claim it saved name", "saved your name" not in r2_lower and "i've noted" not in r2_lower, r2.reply)
    check("1b. 'name' does NOT ask for next booking field", "student's grade" not in r2_lower and "preferred time" not in r2_lower, r2.reply)
    check("1b. 'name' does NOT claim request is being prepared", "being prepared" not in r2_lower and "submit that for you" not in r2_lower, r2.reply)
    check("1b. 'name' clarifies how name is used in booking", "book free demo form" in r2_lower or "clarify" in r2_lower or "student's or parent's name" in r2_lower, r2.reply)

    lead_row = database.get_demo_lead(sid)
    check("1c. Database demo lead name is NOT 'Name'", not lead_row or lead_row.get("name") not in ("Name", "name"), str(lead_row))

    # -------------------------------------------------------------------------
    # PART 2: Standalone Names & Fields Without Active Workflow
    # -------------------------------------------------------------------------
    print("\n--- 2. Standalone Names & Fields Must Not Create Fake Leads ---")
    sid_raaid = f"test-raaid-{uuid.uuid4()}"
    database.ensure_session(sid_raaid)

    r_raaid = ask(engine, sid_raaid, "Raaid")
    r_raaid_lower = r_raaid.reply.lower()
    check("2a. Standalone 'Raaid' greets naturally", "hello" in r_raaid_lower or "how can i help" in r_raaid_lower, r_raaid.reply)
    check("2a. Standalone 'Raaid' does NOT claim to note or save", "noted your name" not in r_raaid_lower and "saved your name" not in r_raaid_lower, r_raaid.reply)
    check("2a. Standalone 'Raaid' does NOT enter booking workflow", "student's grade" not in r_raaid_lower and "preferred time" not in r_raaid_lower, r_raaid.reply)
    lead_raaid = database.get_demo_lead(sid_raaid)
    check("2a. Standalone 'Raaid' created NO lead in database", lead_raaid is None or lead_raaid.get("name") is None, str(lead_raaid))

    # Test "book demo" -> "Raaid"
    sid_demo_raaid = f"test-demo-raaid-{uuid.uuid4()}"
    database.ensure_session(sid_demo_raaid)
    ask(engine, sid_demo_raaid, "book demo")
    r_dr = ask(engine, sid_demo_raaid, "Raaid")
    r_dr_lower = r_dr.reply.lower()
    check("2b. 'book demo' -> 'Raaid' addresses Raaid", "raaid" in r_dr_lower, r_dr.reply)
    check("2b. 'book demo' -> 'Raaid' guides to website button", "book free demo" in r_dr_lower or "top-right" in r_dr_lower, r_dr.reply)
    check("2b. 'book demo' -> 'Raaid' does NOT claim 'I noted your name'", "noted your name" not in r_dr_lower and "saved your name" not in r_dr_lower, r_dr.reply)

    # Test form field queries: "email", "phone", "grade", "time" after demo
    for field in ("email", "phone", "grade", "time"):
        sid_f = f"test-field-{field}-{uuid.uuid4()}"
        database.ensure_session(sid_f)
        ask(engine, sid_f, "How do I book a demo?")
        r_f = ask(engine, sid_f, field)
        r_f_lower = r_f.reply.lower()
        check(f"2c. Field '{field}' after demo guides to form", "book free demo" in r_f_lower or "form" in r_f_lower, r_f.reply)
        check(f"2c. Field '{field}' does not claim to save '{field}'", f"noted your {field}" not in r_f_lower, r_f.reply)

    # -------------------------------------------------------------------------
    # PART 3: Transactional Demo Requests (Booking Guide, Not Booking Agent)
    # -------------------------------------------------------------------------
    print("\n--- 3. Transactional Demo Requests Distinct from Instructions ---")
    transaction_phrases = [
        "Book a demo for me.",
        "Can you book it?",
        "Register me for a demo.",
        "Submit my demo request.",
        "I want you to book it.",
        "Can you sign me up?",
    ]
    for phrase in transaction_phrases:
        sid_t = f"test-tx-{uuid.uuid4()}"
        database.ensure_session(sid_t)
        res = ask(engine, sid_t, phrase)
        res_lower = res.reply.lower()
        check(f"3. '{phrase}' intent is demo_transaction_request", res.intent == "demo_transaction_request", res.intent)
        check(f"3. '{phrase}' does not pretend to execute booking", "i can't submit" in res_lower or "cannot submit" in res_lower or "directly from the chat" in res_lower, res.reply)
        check(f"3. '{phrase}' directs to Book Free Demo button", "book free demo" in res_lower and ("top-right" in res_lower or "button" in res_lower), res.reply)

    # -------------------------------------------------------------------------
    # PART 4: Natural Language Equivalents Matrix (Part 18)
    # -------------------------------------------------------------------------
    print("\n--- 4. Natural Language Phrasing Matrix ---")

    # WeMentors General Info
    wementors_queries = [
        "What is WeMentors?",
        "Tell me about WeMentors.",
        "What do you guys do?",
        "I want to know more about WeMentors.",
        "What's this academy about?",
        "Can you explain what WeMentors is?",
    ]
    for q in wementors_queries:
        sid_w = f"test-wm-{uuid.uuid4()}"
        database.ensure_session(sid_w)
        res = ask(engine, sid_w, q)
        check(f"4a. WeMentors query recognized: '{q}'", res.intent in ("general_info", "capability") or "wementors" in res.reply.lower(), f"{res.intent} -> {res.reply[:60]}")
        check(f"4a. Mentions personal mentoring: '{q}'", "mentor" in res.reply.lower() or "academic" in res.reply.lower(), res.reply[:60])

    # Personalized Mentoring
    mentoring_queries = [
        "What is personalized mentoring?",
        "What does personalized mentoring mean?",
        "How does personal mentoring work?",
        "Do students get individual attention?",
        "Is it one-to-one?",
    ]
    for q in mentoring_queries:
        sid_m = f"test-ment-{uuid.uuid4()}"
        database.ensure_session(sid_m)
        res = ask(engine, sid_m, q)
        check(f"4b. Mentoring query recognized: '{q}'", res.intent in ("personalized_mentoring_general", "mentoring_approach", "one_on_one_general") or "mentor" in res.reply.lower(), f"{res.intent} -> {res.reply[:60]}")
        check(f"4b. Does not dump full Confident Speaker: '{q}'", "confident speaker program is an 8-week" not in res.reply.lower(), res.reply[:60])

    # Confident Speaker
    cs_queries = [
        ("Tell me about Confident Speaker.", "confident_speaker"),
        ("What's the format?", "confident_speaker_format"),
        ("What skills does it cover?", "confident_speaker_scope"),
        ("Who is it for?", "confident_speaker_audience"),
    ]
    for q, expected_intent in cs_queries:
        sid_cs = f"test-cs-{uuid.uuid4()}"
        database.ensure_session(sid_cs)
        # Set context first
        ask(engine, sid_cs, "Tell me about Confident Speaker.")
        res = ask(engine, sid_cs, q)
        check(f"4c. Confident speaker sub-field: '{q}'", res.intent == expected_intent or "speaker" in res.reply.lower() or "confident" in res.reply.lower(), f"{res.intent} -> {res.reply[:60]}")

    # Foundation Years
    fy_queries = [
        ("Tell me about Foundation Years.", "foundation_years_overview"),
        ("What grades is it for?", "foundation_grades"),
        ("What subjects do you teach?", "foundation_subjects"),
        ("How does it work?", "foundation_approach"),
        ("How can I enroll?", "foundation_enrollment"),
    ]
    for q, expected_intent in fy_queries:
        sid_fy = f"test-fy-{uuid.uuid4()}"
        database.ensure_session(sid_fy)
        ask(engine, sid_fy, "Tell me about Foundation Years.")
        res = ask(engine, sid_fy, q)
        check(f"4d. Foundation Years sub-field: '{q}'", res.intent == expected_intent or "foundation" in res.reply.lower(), f"{res.intent} -> {res.reply[:60]}")

    # Middle School
    ms_queries = [
        ("Tell me about Middle School.", "middle_school_overview"),
        ("What grades?", "middle_school_grades"),
        ("What subjects?", "middle_school_all_subjects"),
        ("Do you have doubt solving?", "middle_school_doubt_solving"),
        ("How does the program work?", "middle_school_overview"),
    ]
    for q, expected_intent in ms_queries:
        sid_ms = f"test-ms-{uuid.uuid4()}"
        database.ensure_session(sid_ms)
        ask(engine, sid_ms, "Tell me about Middle School.")
        res = ask(engine, sid_ms, q)
        check(f"4e. Middle School sub-field: '{q}'", res.intent == expected_intent or "middle school" in res.reply.lower(), f"{res.intent} -> {res.reply[:60]}")

    # Demo queries
    demo_queries = [
        ("Is the demo free?", "demo_information"),
        ("How do I book a demo?", "demo_booking"),
        ("I want a demo.", "demo_booking"),
        ("Book a demo.", "demo_booking"),
        ("Where do I book?", "demo_booking"),
    ]
    for q, expected_intent in demo_queries:
        sid_d = f"test-d-{uuid.uuid4()}"
        database.ensure_session(sid_d)
        res = ask(engine, sid_d, q)
        check(f"4f. Demo query: '{q}'", res.intent == expected_intent or "demo" in res.intent, f"{res.intent} -> {res.reply[:60]}")

    # -------------------------------------------------------------------------
    # PART 5: Response Length Scaling (Narrow vs Broad)
    # -------------------------------------------------------------------------
    print("\n--- 5. Response Length Scaling: Narrow vs Broad ---")
    sid_scale = f"test-scale-{uuid.uuid4()}"
    database.ensure_session(sid_scale)

    # Narrow question: What grades is Foundation Years for?
    r_narrow = ask(engine, sid_scale, "What grades is Foundation Years for?")
    check("5a. Narrow grades question is concise", len(r_narrow.reply) < 150, f"chars={len(r_narrow.reply)}: {r_narrow.reply}")
    check("5a. Narrow grades mentions Grades 3–5", "grades 3–5" in r_narrow.reply.lower() or "grades 3-5" in r_narrow.reply.lower(), r_narrow.reply)
    check("5a. Narrow grades does NOT dump subjects/labs", "doubt-solving" not in r_narrow.reply.lower() and "practical labs" not in r_narrow.reply.lower(), r_narrow.reply)

    # Narrow question: What grades is Middle School for?
    r_narrow_ms = ask(engine, sid_scale, "What grades is Middle School for?")
    check("5b. Narrow middle school grades is concise", len(r_narrow_ms.reply) < 150, f"chars={len(r_narrow_ms.reply)}: {r_narrow_ms.reply}")
    check("5b. Narrow middle school grades mentions Grades 6–8", "grades 6–8" in r_narrow_ms.reply.lower() or "grades 6-8" in r_narrow_ms.reply.lower(), r_narrow_ms.reply)

    # Broad question: Tell me about Foundation Years
    r_broad = ask(engine, sid_scale, "Tell me about Foundation Years.")
    check("5c. Broad question gives comprehensive overview", len(r_broad.reply) > 200, f"chars={len(r_broad.reply)}")
    check("5c. Broad question mentions Grades 3–5 and foundation focus", "grades 3–5" in r_broad.reply.lower() or "grades 3-5" in r_broad.reply.lower(), r_broad.reply)

    # -------------------------------------------------------------------------
    # PART 6: Context Priority: Current Explicit Question > Prior Context
    # -------------------------------------------------------------------------
    print("\n--- 6. Context Priority: Current Question Overrides Past Topic ---")
    sid_prio = f"test-prio-{uuid.uuid4()}"
    database.ensure_session(sid_prio)

    # Turn 1: Discuss Confident Speaker
    ask(engine, sid_prio, "Tell me about Confident Speaker.")
    # Turn 2: Ask general mentoring concept question
    r_switch = ask(engine, sid_prio, "What is personalized mentoring?")
    check("6a. General mentoring question not overridden by Confident Speaker", "personalized mentoring means" in r_switch.reply.lower() or "individual attention" in r_switch.reply.lower(), r_switch.reply)
    check("6a. Does NOT repeat Confident Speaker overview", "8-week public speaking" not in r_switch.reply.lower() and "confident speaker program is an 8-week" not in r_switch.reply.lower(), r_switch.reply)

    # Turn 3: Ask program-specific mentoring question
    r_prog_ment = ask(engine, sid_prio, "How is personalized mentoring used in Confident Speaker?")
    check("6b. Explicit program mentoring question is program-specific", "confident speaker" in r_prog_ment.reply.lower() or "speech" in r_prog_ment.reply.lower() or "mentor" in r_prog_ment.reply.lower(), r_prog_ment.reply)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed_count} passed, {len(failures)} failed")
    print("=" * 80)
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nALL FALSE DEMO BOOKING & NATURAL LANGUAGE TESTS PASSED (100% SUCCESS)!")


if __name__ == "__main__":
    run_tests()
