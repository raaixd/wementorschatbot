"""
Regression tests for feature scope assignment in WeMentors Chatbot.

Verifies that "doubt clinics" and "practical labs / practical problem sets" are
treated as general academic program features across WeMentors programs, rather than
being falsely scoped exclusively to Middle School (Grades 6–8).

Test coverage includes:
1. General feature questions (no Middle School injection)
2. Explicit program context questions (Grade 7, Grade 8, Middle School)
3. Multi-turn context resolution (isolated general vs anaphoric program reference)
4. Deterministic fallback mode (LLM_ENABLED = False)
5. Underlying retrieval ranking and scope preservation
"""

import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import config, database, personality
from app.conversation import ConversationEngine
from app.knowledge import load_entries
from app.retrieval import Retriever

passed = 0
failed = 0


def check(name: str, condition: bool, details: str = "") -> None:
    global passed, failed
    if condition:
        print(f"[PASS] {name}")
        passed += 1
    else:
        print(f"[FAIL] {name}")
        if details:
            print(f"       Details: {details}")
        failed += 1


def run_tests():
    print("=" * 80)
    print("RUNNING FEATURE SCOPE ASSIGNMENT REGRESSION TESTS")
    print("=" * 80)

    database.init_db()
    entries = load_entries()
    engine = ConversationEngine(entries)

    # =========================================================================
    # 1. RETRIEVAL RANKING TESTS
    # =========================================================================
    print("\n--- 1. Underlying Retrieval Scope & Ranking ---")
    kb = load_entries()
    retriever = Retriever(kb)

    # General doubt query should rank general doubt-solving-sessions #1
    res_gen_doubt = retriever.search("How do doubt clinics work?", top_k=2)
    check(
        "1a. General doubt clinics ranks general entry #1",
        bool(res_gen_doubt and res_gen_doubt[0].entry.id == "doubt-solving-sessions"),
        f"Top entry was: {res_gen_doubt[0].entry.id if res_gen_doubt else 'None'}",
    )

    res_gen_doubt_offer = retriever.search("Do you offer doubt clinics?", top_k=2)
    check(
        "1b. 'Do you offer doubt clinics?' ranks general entry #1",
        bool(res_gen_doubt_offer and res_gen_doubt_offer[0].entry.id == "doubt-solving-sessions"),
        f"Top entry was: {res_gen_doubt_offer[0].entry.id if res_gen_doubt_offer else 'None'}",
    )

    # Explicit Middle School query should rank middle school entry #1
    res_ms_doubt = retriever.search("Does Middle School have doubt clinics?", top_k=2)
    check(
        "1c. Middle School doubt clinics ranks middle school entry #1",
        bool(res_ms_doubt and res_ms_doubt[0].entry.id == "middle-school-doubt-clinics"),
        f"Top entry was: {res_ms_doubt[0].entry.id if res_ms_doubt else 'None'}",
    )

    res_g7_doubt = retriever.search("How do doubt clinics work for Grade 7?", top_k=2)
    check(
        "1d. Grade 7 doubt clinics ranks middle school entry #1",
        bool(res_g7_doubt and res_g7_doubt[0].entry.id == "middle-school-doubt-clinics"),
        f"Top entry was: {res_g7_doubt[0].entry.id if res_g7_doubt else 'None'}",
    )

    res_g8_labs = retriever.search("What practical labs are available for Grade 8?", top_k=2)
    check(
        "1e. Grade 8 practical labs ranks middle school practical labs #1",
        bool(res_g8_labs and res_g8_labs[0].entry.id == "middle-school-practical-labs"),
        f"Top entry was: {res_g8_labs[0].entry.id if res_g8_labs else 'None'}",
    )

    # =========================================================================
    # 2. GENERAL FEATURE QUESTIONS (NO PROGRAM INJECTION)
    # =========================================================================
    print("\n--- 2. General Feature Questions ---")
    general_queries = [
        ("How do doubt clinics work?", "doubt_clinics"),
        ("What are doubt clinics?", "doubt_clinics"),
        ("Tell me about practical labs.", "practical_labs"),
        ("What are practical problem sets?", "practical_labs"),
        ("Do you offer doubt clinics?", "doubt_clinics"),
        ("Do you have practical problem sets?", "practical_labs"),
        ("How do practical labs work?", "practical_labs"),
    ]

    for q, expected_intent in general_queries:
        sid = f"test-gen-{uuid.uuid4()}"
        res = engine.handle_message(sid, q)
        reply_lower = res.reply.lower()

        check(
            f"2. Intent for '{q}' is '{expected_intent}'",
            res.intent == expected_intent,
            f"Got intent: {res.intent}",
        )
        check(
            f"2. Reply for '{q}' does NOT inject 'middle school'",
            "middle school" not in reply_lower,
            f"Reply: {res.reply}",
        )
        check(
            f"2. Reply for '{q}' does NOT inject 'grades 6–8' or 'grade 7'",
            ("grades 6–8" not in reply_lower and "grade 7" not in reply_lower and "grade 8" not in reply_lower and "grade 6" not in reply_lower),
            f"Reply: {res.reply}",
        )

    # =========================================================================
    # 3. EXPLICIT PROGRAM CONTEXT QUESTIONS
    # =========================================================================
    print("\n--- 3. Explicit Program Context Questions ---")

    # How do doubt clinics work for Grade 7?
    sid = f"test-exp-{uuid.uuid4()}"
    res = engine.handle_message(sid, "How do doubt clinics work for Grade 7?")
    check(
        "3a. Grade 7 doubt clinics intent recognized as middle_school_doubt_clinics",
        res.intent == "middle_school_doubt_clinics",
        f"Got intent: {res.intent}",
    )
    check(
        "3a. Grade 7 doubt clinics mentions doubt clinics or doubt solving",
        ("doubt" in res.reply.lower()),
        f"Reply: {res.reply}",
    )

    # Does Middle School have doubt clinics?
    sid = f"test-exp-{uuid.uuid4()}"
    res = engine.handle_message(sid, "Does Middle School have doubt clinics?")
    check(
        "3b. Middle School doubt clinics intent recognized as middle_school_doubt_clinics",
        res.intent == "middle_school_doubt_clinics",
        f"Got intent: {res.intent}",
    )
    check(
        "3b. Reply mentions doubt clinics or doubt solving in middle school",
        ("doubt" in res.reply.lower()),
        f"Reply: {res.reply}",
    )

    # What practical labs are available for Grade 8?
    sid = f"test-exp-{uuid.uuid4()}"
    res = engine.handle_message(sid, "What practical labs are available for Grade 8?")
    check(
        "3c. Grade 8 practical labs intent recognized as middle_school_practical_labs",
        res.intent == "middle_school_practical_labs",
        f"Got intent: {res.intent}",
    )
    check(
        "3c. Reply mentions practical labs or problem sets",
        ("practical" in res.reply.lower() or "labs" in res.reply.lower()),
        f"Reply: {res.reply}",
    )

    # =========================================================================
    # 4. MULTI-TURN CONTEXT RESOLUTION
    # =========================================================================
    print("\n--- 4. Multi-Turn Context Resolution ---")

    # Multi-turn Case A: Middle School overview -> General doubt question
    sid = f"test-multi-a-{uuid.uuid4()}"
    res1 = engine.handle_message(sid, "Tell me about the Middle School programme.")
    check("4a. Turn 1 Middle School overview intent", res1.intent == "middle_school_overview", res1.intent)
    mem = database.get_conversation_memory(sid)
    check("4a. Turn 1 active_program set to 'middle'", mem.active_program == "middle", mem.active_program)

    res2 = engine.handle_message(sid, "How do doubt clinics work?")
    check(
        "4a. Turn 2 'How do doubt clinics work?' recognized as general 'doubt_clinics' intent",
        res2.intent == "doubt_clinics",
        f"Got intent: {res2.intent}",
    )
    check(
        "4a. Turn 2 reply does NOT inherit 'Middle School'",
        "middle school" not in res2.reply.lower(),
        f"Reply: {res2.reply}",
    )

    # Multi-turn Case B: Middle School overview -> Anaphoric doubt question ("in that programme")
    sid = f"test-multi-b-{uuid.uuid4()}"
    res1 = engine.handle_message(sid, "Tell me about the Middle School programme.")
    check("4b. Turn 1 Middle School overview intent", res1.intent == "middle_school_overview", res1.intent)

    res2 = engine.handle_message(sid, "How do doubt clinics work in that programme?")
    check(
        "4b. Turn 2 '...in that programme?' resolves to 'middle_school_doubt_clinics'",
        res2.intent == "middle_school_doubt_clinics",
        f"Got intent: {res2.intent}",
    )
    check(
        "4b. Turn 2 reply explains Middle School doubt clinics",
        "doubt" in res2.reply.lower(),
        f"Reply: {res2.reply}",
    )

    # Multi-turn Case C: Middle School overview -> General practical labs question
    sid = f"test-multi-c-{uuid.uuid4()}"
    engine.handle_message(sid, "Tell me about the Middle School programme.")
    res_pl = engine.handle_message(sid, "Tell me about practical labs.")
    check(
        "4c. Turn 2 'Tell me about practical labs.' recognized as general 'practical_labs' intent",
        res_pl.intent == "practical_labs",
        f"Got intent: {res_pl.intent}",
    )
    check(
        "4c. Turn 2 reply does NOT inherit 'Middle School'",
        "middle school" not in res_pl.reply.lower(),
        f"Reply: {res_pl.reply}",
    )

    # Multi-turn Case D: Middle School overview -> Anaphoric practical labs question ("in that programme")
    sid = f"test-multi-d-{uuid.uuid4()}"
    engine.handle_message(sid, "Tell me about the Middle School programme.")
    res_pl_anaphoric = engine.handle_message(sid, "What practical labs are available in that programme?")
    check(
        "4d. Turn 2 'What practical labs are available in that programme?' resolves to 'middle_school_practical_labs'",
        res_pl_anaphoric.intent == "middle_school_practical_labs",
        f"Got intent: {res_pl_anaphoric.intent}",
    )

    # =========================================================================
    # 5. DETERMINISTIC FALLBACK MODE (LLM_ENABLED = False)
    # =========================================================================
    print("\n--- 5. Deterministic Fallback Mode (LLM_ENABLED = False) ---")
    orig_llm_enabled = config.LLM_ENABLED
    config.LLM_ENABLED = False

    try:
        # General question in fallback mode
        sid = f"test-fb-{uuid.uuid4()}"
        res = engine.handle_message(sid, "How do doubt clinics work?")
        check(
            "5a. Fallback mode general doubt clinics intent is 'doubt_clinics'",
            res.intent == "doubt_clinics",
            f"Got intent: {res.intent}",
        )
        check(
            "5a. Fallback mode response does NOT contain 'Middle School'",
            "middle school" not in res.reply.lower(),
            f"Reply: {res.reply}",
        )

        sid = f"test-fb-pl-{uuid.uuid4()}"
        res = engine.handle_message(sid, "Tell me about practical labs.")
        check(
            "5b. Fallback mode general practical labs intent is 'practical_labs'",
            res.intent == "practical_labs",
            f"Got intent: {res.intent}",
        )
        check(
            "5b. Fallback mode practical labs response does NOT contain 'Middle School'",
            "middle school" not in res.reply.lower(),
            f"Reply: {res.reply}",
        )

        # Explicit context in fallback mode
        sid = f"test-fb-ms-{uuid.uuid4()}"
        res = engine.handle_message(sid, "How do doubt clinics work for Grade 7?")
        check(
            "5c. Fallback mode Grade 7 doubt clinics intent is 'middle_school_doubt_clinics'",
            res.intent == "middle_school_doubt_clinics",
            f"Got intent: {res.intent}",
        )
        check(
            "5c. Fallback mode Grade 7 doubt clinics provides verified response",
            "doubt" in res.reply.lower(),
            f"Reply: {res.reply}",
        )
    finally:
        config.LLM_ENABLED = orig_llm_enabled

    print("\n" + "=" * 80)
    print(f"FEATURE SCOPE ASSIGNMENT TESTS: {passed} PASSED, {failed} FAILED")
    print("=" * 80)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
