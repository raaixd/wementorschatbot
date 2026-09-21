"""
Comprehensive Test Suite for Natural Language Inference, Program Recognition,
Structured Context Memory, Reference Resolution, and Varied Follow-ups.

Verifies Section 11 flows and architecture improvements for WeMentors Chatbot.
"""

import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path so app imports cleanly
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import config, database, personality
from app.conversation import ConversationEngine, normalize_query
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


def run_tests():
    database.init_db()
    entries = load_entries()
    engine = ConversationEngine(entries)

    print("\n" + "=" * 80)
    print("RUNNING NATURAL INFERENCE, PROGRAM RECOGNITION & MEMORY REGRESSION TESTS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Foundation Years and Middle School Recognition & Typo Tolerance
    # -------------------------------------------------------------------------
    print("\n--- 1. Foundation and Middle School Typo Tolerance & Recognition ---")
    
    # Standalone 'foundation'
    sid = f"test-fnd-{uuid.uuid4()}"
    res = engine.handle_message(sid, "foundation")
    check("1a. 'foundation' recognized", res.intent in ("foundation_years_clarification", "foundation_years_overview"), res.intent)
    check("1a. 'foundation' mentions Foundation Years & Grades 3–5", "foundation" in res.reply.lower() and "3" in res.reply and "5" in res.reply, res.reply)
    check("1a. 'foundation' does not trigger generic fallback menu", "I can help with WeMentors’ academic programs (Grades 3–10)" not in res.reply, res.reply)

    # Standalone 'found' (typo for Foundation)
    sid = f"test-found-{uuid.uuid4()}"
    res = engine.handle_message(sid, "found")
    check("1b. 'found' recognized as Foundation Years", res.intent in ("foundation_years_clarification", "foundation_years_overview"), res.intent)
    check("1b. 'found' clarifies Foundation Years for Grades 3–5", "foundation" in res.reply.lower() and "grades 3" in res.reply.lower(), res.reply)
    check("1b. 'found' not a generic fallback", "not in my verified records" not in res.reply and "What would you like to know?" not in res.reply, res.reply)

    # Standalone 'foundating' (typo)
    sid = f"test-foundating-{uuid.uuid4()}"
    res = engine.handle_message(sid, "foundating")
    check("1c. 'foundating' recognized as Foundation Years", res.intent in ("foundation_years_clarification", "foundation_years_overview"), res.intent)
    check("1c. 'foundating' clarifies Foundation Years", "foundation" in res.reply.lower(), res.reply)

    # Standalone 'doundation' (typo)
    sid = f"test-doundation-{uuid.uuid4()}"
    res = engine.handle_message(sid, "doundation")
    check("1d. 'doundation' recognized as Foundation Years", res.intent in ("foundation_years_clarification", "foundation_years_overview"), res.intent)
    check("1d. 'doundation' clarifies Foundation Years", "foundation" in res.reply.lower(), res.reply)

    # Standalone 'middle'
    sid = f"test-mid-{uuid.uuid4()}"
    res = engine.handle_message(sid, "middle")
    check("1e. 'middle' recognized as Middle School", res.intent == "middle_school_overview", res.intent)
    check("1e. 'middle' covers Grades 6–8 & all subjects", "grades 6" in res.reply.lower() and "all subjects" in res.reply.lower(), res.reply)

    # 'tell me about middle school'
    sid = f"test-mid-ov-{uuid.uuid4()}"
    res = engine.handle_message(sid, "tell me about middle school")
    check("1f. 'tell me about middle school' matches overview", res.intent == "middle_school_overview", res.intent)
    reply_l = res.reply.lower()
    check("1f. Mentions Grades 6–8", "grades 6" in reply_l and "8" in reply_l, res.reply)
    check("1f. Mentions all subjects", "all subjects" in reply_l, res.reply)
    check("1f. Mentions doubt-solving", "doubt-solving" in reply_l or "doubt" in reply_l, res.reply)
    check("1f. Mentions practical labs", "practical labs" in reply_l, res.reply)
    check("1f. Mentions concept-based learning", "concept-based learning" in reply_l, res.reply)
    check("1f. Mentions practical problem sets", "practical problem sets" in reply_l, res.reply)
    check("1f. Mentions fortnightly doubt clinics", "fortnightly doubt clinics" in reply_l, res.reply)
    check("1f. Mentions progress dashboard access", "progress dashboard" in reply_l, res.reply)

    # -------------------------------------------------------------------------
    # 2. Middle School Field-Level Queries
    # -------------------------------------------------------------------------
    print("\n--- 2. Middle School Field-Level Recognition ---")

    # Does middle school include all subjects?
    sid = f"test-sub-{uuid.uuid4()}"
    res = engine.handle_message(sid, "Does middle school include all subjects?")
    check("2a. All subjects field recognized", res.intent == "middle_school_subjects", res.intent)
    check("2a. Mentions all core subjects and doubt-solving", "all core subjects" in res.reply.lower(), res.reply)

    # What are doubt clinics?
    sid = f"test-dc-{uuid.uuid4()}"
    res = engine.handle_message(sid, "What are doubt clinics?")
    check("2b. Doubt clinics field recognized", res.intent in ("doubt_clinics", "middle_school_doubt_clinics"), res.intent)
    check("2b. Explains fortnightly doubt clinics specifically", "fortnightly" in res.reply.lower() and "doubt" in res.reply.lower(), res.reply)

    # Are there doubt-solving sessions?
    sid = f"test-ds-{uuid.uuid4()}"
    res = engine.handle_message(sid, "Are there doubt-solving sessions?")
    check("2c. Doubt solving recognized", res.intent in ("doubt_clinics", "middle_school_doubt_clinics"), res.intent)

    # Do students get practical labs?
    sid = f"test-pl-{uuid.uuid4()}"
    res = engine.handle_message(sid, "Do students get practical labs?")
    check("2d. Practical labs field recognized", res.intent in ("practical_labs", "middle_school_practical_labs"), res.intent)
    check("2d. Explains practical labs & problem sets", "practical labs" in res.reply.lower(), res.reply)

    # Can parents track progress?
    sid = f"test-prog-{uuid.uuid4()}"
    # Setting middle school context first
    engine.handle_message(sid, "Tell me about middle school")
    res = engine.handle_message(sid, "Can parents track progress?")
    check("2e. Parent progress in Middle School recognized", "progress dashboard" in res.reply.lower() or "progress" in res.reply.lower(), res.reply)

    # Grade 7 query
    sid = f"test-g7-{uuid.uuid4()}"
    res = engine.handle_message(sid, "What do you offer for Grade 7?")
    check("2f. Grade 7 routes to Middle School", res.intent in ("middle_school_grades", "middle_school_overview"), res.intent)
    check("2f. Mentions Middle School Grades 6–8", "middle school" in res.reply.lower(), res.reply)

    # Do you teach Grades 6–8?
    sid = f"test-g68-{uuid.uuid4()}"
    res = engine.handle_message(sid, "Do you teach Grades 6–8?")
    check("2g. Grades 6-8 routes to Middle School", res.intent in ("middle_school_grades", "middle_school_overview"), res.intent)

    # -------------------------------------------------------------------------
    # 3. Reference Resolution and Memory Correction
    # -------------------------------------------------------------------------
    print("\n--- 3. Contextual Reference Resolution & Memory Correction ---")

    sid_ref = f"test-ref-{uuid.uuid4()}"
    database.ensure_session(sid_ref)

    # Turn 1: User asks about programs
    r1 = engine.handle_message(sid_ref, "Tell me about the programs")
    check("3a. Turn 1 lists multiple programs", "Foundation Years" in r1.reply and "Middle School" in r1.reply and "Senior School" in r1.reply, r1.reply)

    # Turn 2: User asks "tell me more about the first one"
    r2 = engine.handle_message(sid_ref, "tell me more about the first one")
    check("3b. Turn 2 'first one' resolves to Foundation Years", "foundation" in r2.reply.lower() and "grades 3" in r2.reply.lower(), r2.reply)
    check("3b. Turn 2 matched entry is Foundation Years", "program-foundation-years" in r2.matched_entry_ids, r2.matched_entry_ids)

    # Turn 3: User corrects "no before foundation"
    r3 = engine.handle_message(sid_ref, "no before foundation")
    check("3c. Turn 3 'no before foundation' recognizes Foundation is earliest", "earliest" in r3.reply.lower() or "first" in r3.reply.lower(), r3.reply)
    check("3c. Turn 3 does not fall back to generic capability menu", "What can I help you with" not in r3.reply, r3.reply)

    # Turn 4: User asks "what about the other one"
    r4 = engine.handle_message(sid_ref, "what about the other one")
    check("3d. Turn 4 'what about the other one' advances to Middle School", "middle school" in r4.reply.lower() or "grades 6" in r4.reply.lower(), r4.reply)
    check("3d. Turn 4 matched entry is Middle School", "program-middle-school" in r4.matched_entry_ids, r4.matched_entry_ids)

    # Turn 5: User says "after middle school"
    r5 = engine.handle_message(sid_ref, "after middle school")
    check("3e. Turn 5 'after middle school' resolves to Senior School / Grades 9-10", "senior" in r5.reply.lower() or "9" in r5.reply.lower(), r5.reply)

    # Turn 6: User says "before middle school"
    r6 = engine.handle_message(sid_ref, "before middle school")
    check("3f. Turn 6 'before middle school' resolves to Foundation Years", "foundation" in r6.reply.lower(), r6.reply)

    # -------------------------------------------------------------------------
    # 4. Personalized Mentoring Concept vs Confident Speaker
    # -------------------------------------------------------------------------
    print("\n--- 4. Personalized Mentoring Concept vs Program ---")

    sid_pm = f"test-pm-{uuid.uuid4()}"
    database.ensure_session(sid_pm)

    # Bare "personalized mentoring"
    r_pm1 = engine.handle_message(sid_pm, "personalized mentoring")
    check("4a. 'personalized mentoring' routes to concept definition", r_pm1.intent == "personalized_mentoring_general", r_pm1.intent)
    check("4a. Explains personal mentor guidance without full Confident Speaker overview", "personal mentor" in r_pm1.reply.lower() and "public speaking" not in r_pm1.reply.lower(), r_pm1.reply)

    # "what is personalized mentoring"
    r_pm2 = engine.handle_message(sid_pm, "what is personalized mentoring")
    check("4b. 'what is personalized mentoring' routes to general definition", r_pm2.intent == "personalized_mentoring_general", r_pm2.intent)
    check("4b. Does not dump Confident Speaker overview", "interview skills" not in r_pm2.reply.lower(), r_pm2.reply)

    # "how does personalized mentoring work in confident speaker"
    r_pm3 = engine.handle_message(sid_pm, "how does personalized mentoring work in confident speaker")
    check("4c. Explicit Confident Speaker mentoring routes to program mentoring", r_pm3.intent == "confident_speaker_mentoring", r_pm3.intent)
    check("4c. Mentions personal mentor rather than rotating group", "rotating" in r_pm3.reply.lower() or "personal mentor" in r_pm3.reply.lower(), r_pm3.reply)

    # -------------------------------------------------------------------------
    # 5. Context Switching (Non-sticky conversation state)
    # -------------------------------------------------------------------------
    print("\n--- 5. Dynamic Context Switching ---")

    sid_sw = f"test-sw-{uuid.uuid4()}"
    database.ensure_session(sid_sw)

    # User asks about Foundation
    r_sw1 = engine.handle_message(sid_sw, "tell me about foundation")
    check("5a. Foundation overview returned", "foundation" in r_sw1.reply.lower(), r_sw1.reply)

    # User immediately switches to Grades 6-8
    r_sw2 = engine.handle_message(sid_sw, "what about grades 6-8")
    check("5b. Switch from Foundation to Middle School succeeds", "middle school" in r_sw2.reply.lower() or "grades 6" in r_sw2.reply.lower(), r_sw2.reply)
    check("5b. Does not continue talking about Foundation Years", "environmental studies" not in r_sw2.reply.lower(), r_sw2.reply)

    # User switches to Confident Speaker
    r_sw3 = engine.handle_message(sid_sw, "tell me about confident speaker")
    check("5c. Switch to Confident Speaker succeeds", "speaking" in r_sw3.reply.lower() or "confident" in r_sw3.reply.lower(), r_sw3.reply)

    # User asks general mentoring question after Confident Speaker
    r_sw4 = engine.handle_message(sid_sw, "what is personalized mentoring")
    check("5d. General mentoring question not anchored to Confident Speaker", r_sw4.intent == "personalized_mentoring_general", r_sw4.intent)
    check("5d. Returns general definition", "guidance from a personal mentor" in r_sw4.reply.lower() or "individual attention" in r_sw4.reply.lower(), r_sw4.reply)

    # -------------------------------------------------------------------------
    # 6. Varied Follow-Up Suggestions Rotation
    # -------------------------------------------------------------------------
    print("\n--- 6. Varied Follow-Up Suggestions ---")

    pool_foundation = personality.FOLLOW_UP_POOLS["foundation"]
    pool_middle = personality.FOLLOW_UP_POOLS["middle"]
    
    # Test generation with empty recent history
    sug1 = personality.get_varied_follow_up_suggestions("middle", "middle_school_overview", [], count=2)
    check("6a. Generates 2 suggestions for middle school", len(sug1) == 2, str(sug1))
    check("6a. Suggestions come from middle school pool", all(s in pool_middle for s in sug1), str(sug1))

    # Test rotation on next turn: does not repeat same suggestions
    sug2 = personality.get_varied_follow_up_suggestions("middle", "middle_school_doubt_clinics", sug1, count=2)
    check("6b. Rotates suggestions without immediate repeat", sug2 != sug1, f"sug1={sug1}, sug2={sug2}")

    # Check API response format has suggestions attached
    res_api = engine.handle_message(sid_sw, "tell me about middle school")
    check("6c. ReplyResult contains structured suggestions", res_api.suggestions is not None and len(res_api.suggestions) > 0, str(res_api.suggestions))
    check("6c. Reply text is not contaminated with 'Would you like to know'", "Would you like to know" not in res_api.reply, res_api.reply)

    # -------------------------------------------------------------------------
    # 7. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    if failures:
        print(f"FAILED ({len(failures)} checks failed):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"ALL {passed_count} NATURAL INFERENCE & MEMORY TESTS PASSED (100% SUCCESS)!")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    run_tests()
