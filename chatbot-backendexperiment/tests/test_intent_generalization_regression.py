"""
Regression test suite for WeMentors Chatbot Language Understanding,
Intent Generalization, Verified Positioning, Response Quality, and Fallbacks.

Verifies:
1. Generalized equivalent user questions for General WeMentors Info (Section 1 & 10)
2. Verified positioning embedded naturally (Section 2)
3. Board-exam preparation queries for Grades 9-10 (Section 3 & 10)
4. Confident Speaker queries (Section 4 & 10)
5. Response formatting and next-step CTAs (Section 5)
6. Semantic understanding of informal phrasings (Section 6)
7. Context preservation and anti-contamination (Section 7)
8. Intent-specific fallback behavior (Section 8)
9. Response quality and anti-hallucination checks (Section 9)
10. Cross-intent relevance tests (Section 10)
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
        safe_detail = detail.encode("ascii", errors="replace").decode("ascii")
        print(f"[FAIL] {name} -> {safe_detail}")


def run_tests():
    database.init_db()
    entries = load_entries()
    engine = ConversationEngine(entries)

    print("\n" + "=" * 75)
    print("RUNNING INTENT GENERALIZATION & LANGUAGE UNDERSTANDING REGRESSIONS")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. General WeMentors Information Variations (Sections 1 & 10)
    # -------------------------------------------------------------------------
    print("\n--- 1. General WeMentors Information Variations ---")
    general_queries = [
        "What is WeMentors?",
        "What are WeMentors?",
        "I wanna know more about WeMentors",
        "Tell me about your academy",
        "Tell me about your organization",
        "What do you guys do?",
        "Can you explain WeMentors?",
        "What is your academy about?",
        "Give me some information about WeMentors",
        "Give me information about your academy",
        "Tell me more about your classes",
        "How does WeMentors work?",
        "What does WeMentors offer?",
        "Can you introduce me to WeMentors?",
        "I wanna know more about you",
        "Tell me what you guys do",
        "Can you introduce your services?",
        "How does your academy help students?",
    ]

    for q in general_queries:
        sid = f"reg-gen-{uuid.uuid4()}"
        database.ensure_session(sid)
        res = engine.handle_message(sid, q)
        reply_lower = res.reply.lower()

        check(f"General query recognized: '{q}'", res.intent == "general_info" or "wementors" in reply_lower, f"intent={res.intent}")
        check(f"General query has personal mentor: '{q}'", "personal mentor" in reply_lower, res.reply)
        check(f"General query has progress tracking/parent updates: '{q}'", "progress" in reply_lower and "week" in reply_lower, res.reply)
        check(f"General query mentions verified offerings: '{q}'", "foundation" in reply_lower or "middle" in reply_lower or "senior" in reply_lower or "confident speaker" in reply_lower, res.reply)
        check(f"General query mentions demo CTA: '{q}'", "book free demo" in reply_lower, res.reply)
        check(f"General query avoids missing-records disclaimer: '{q}'", "don't have that specific detail" not in reply_lower and "not in my verified records" not in reply_lower, res.reply)
        check(f"General query avoids fee disclaimer: '{q}'", "fee details are shared individually" not in reply_lower and "tuition fee" not in reply_lower, res.reply)

    # -------------------------------------------------------------------------
    # 2. Board-Exam Preparation Queries (Sections 3 & 10)
    # -------------------------------------------------------------------------
    print("\n--- 2. Board-Exam Preparation Queries (Grades 9-10) ---")
    board_queries = [
        "How will you prepare my child for board exams?",
        "How do you prepare students for boards?",
        "Can you help my child with board exams?",
        "What support do you provide for Class 9 and 10?",
        "How will you help my child score well in boards?",
        "Do you have a board exam preparation course?",
        "How do you prepare Class 9 and 10 students?",
        "How will you prep my child for board exams?",
        "How do you prepare Class 10 students?",
        "Can you help my child with boards?",
        "What support do you provide for Grade 9?",
        "How will the mentor work with my child?",
        "How will you prepare my child for boards?",
        "My son is in 10th, how can you help?",
        "Can your mentors help with board preparation?",
        "I need academic support for my daughter's board exams.",
    ]

    for q in board_queries:
        sid = f"reg-board-{uuid.uuid4()}"
        database.ensure_session(sid)
        res = engine.handle_message(sid, q)
        reply_lower = res.reply.lower()

        check(f"Board query recognized: '{q}'", res.intent in ("board_exam", "faq") and res.matched_entry_ids == ["program-senior-school"], f"intent={res.intent} matched={res.matched_entry_ids}")
        check(f"Board query mentions Grades 9-10: '{q}'", "9" in reply_lower and "10" in reply_lower, res.reply)
        check(f"Board query mentions personal mentor (not rotating): '{q}'", "personal mentor" in reply_lower and "rotating" in reply_lower, res.reply)
        check(f"Board query mentions individual attention: '{q}'", "individual" in reply_lower, res.reply)
        check(f"Board query mentions weekly parent updates: '{q}'", "progress" in reply_lower and ("week" in reply_lower or "parent" in reply_lower), res.reply)
        check(f"Board query directs to Book Free Demo: '{q}'", "book free demo" in reply_lower, res.reply)
        check(f"Board query avoids hallucinated mark guarantees: '{q}'", "guaranteed marks" not in reply_lower and "guaranteed rank" not in reply_lower, res.reply)

    # -------------------------------------------------------------------------
    # 3. Confident Speaker Program Queries (Sections 4 & 10)
    # -------------------------------------------------------------------------
    print("\n--- 3. Confident Speaker Program Queries ---")
    speaker_queries = [
        "Tell me about the Confident Speaker program",
        "What is your confident speaker course?",
        "How does the confident speaker program work?",
        "Can you help my child become a confident speaker?",
        "What do students learn in Confident Speaker?",
        "Is there personalized mentoring for confident speaking?",
        "How will you work on my child's speaking confidence?",
        "How can my child become better at speaking?",
        "Tell me about your speaking confidence program",
        "Do you have something for public speaking confidence?",
        "i want to learn to speak confidently",
        "I want to learn to speak confidently",
        "how can I speak with confidence?",
    ]

    for q in speaker_queries:
        sid = f"reg-speaker-{uuid.uuid4()}"
        database.ensure_session(sid)
        res = engine.handle_message(sid, q)
        reply_lower = res.reply.lower()

        check(
            f"Speaker query recognized: '{q}'",
            (res.intent.startswith("confident_speaker") or res.intent in ("confident_speaker", "faq"))
            and any("confident-speaker" in m for m in res.matched_entry_ids),
            f"intent={res.intent} matched={res.matched_entry_ids}",
        )
        check(f"Speaker query mentions personalized mentoring: '{q}'", "personalized mentoring" in reply_lower or "personal mentor" in reply_lower, res.reply)
        check(f"Speaker query mentions individual work/attention: '{q}'", "individual" in reply_lower, res.reply)
        check(f"Speaker query mentions guided practice/feedback: '{q}'", "practice" in reply_lower or "feedback" in reply_lower, res.reply)
        check(f"Speaker query directs to Book Free Demo: '{q}'", "book free demo" in reply_lower, res.reply)
        check(f"Speaker query does not promise overnight fluency: '{q}'", "fluent overnight" not in reply_lower and "eliminate all anxiety" not in reply_lower, res.reply)

    # -------------------------------------------------------------------------
    # 4. Response Formatting (Section 5)
    # -------------------------------------------------------------------------
    print("\n--- 4. Response Formatting & Practical Next Steps ---")
    sid_fmt = f"reg-fmt-{uuid.uuid4()}"
    database.ensure_session(sid_fmt)

    # Demo booking instruction
    r_book = engine.handle_message(sid_fmt, "How can I book a free demo?")
    check("How to book mentions 'Book Free Demo' top-right", "click the **book free demo** option at the top-right of the website" in r_book.reply.lower(), r_book.reply)

    # -------------------------------------------------------------------------
    # 5. Conversation Context & Anti-Contamination (Section 7)
    # -------------------------------------------------------------------------
    print("\n--- 5. Context Preservation & Anti-Contamination ---")
    sid_ctx = f"reg-ctx-{uuid.uuid4()}"
    database.ensure_session(sid_ctx)

    # Turn 1: User asks about fees
    r_fee = engine.handle_message(sid_ctx, "What are your fees?")
    database.log_message(sid_ctx, "user", "What are your fees?")
    database.log_message(sid_ctx, "assistant", r_fee.reply, intent=r_fee.intent, matched_entry_ids="fees-and-pricing")
    check("Context Turn 1 answered fee question", "fee" in r_fee.reply.lower(), r_fee.reply)

    # Turn 2: User asks what programs are offered (must NOT mention fees)
    r_prog = engine.handle_message(sid_ctx, "What programs do you offer?")
    check("Context Turn 2 answered programs question", "foundation" in r_prog.reply.lower() and "senior" in r_prog.reply.lower(), r_prog.reply)
    check("Context Turn 2 stripped fee contamination", "fee details are shared individually" not in r_prog.reply.lower(), r_prog.reply)

    # Turn 3: User asks about Confident Speaker program
    sid_ctx2 = f"reg-ctx2-{uuid.uuid4()}"
    database.ensure_session(sid_ctx2)
    r_spk = engine.handle_message(sid_ctx2, "Tell me about the Confident Speaker program")
    database.log_message(sid_ctx2, "user", "Tell me about the Confident Speaker program")
    database.log_message(sid_ctx2, "assistant", r_spk.reply, intent=r_spk.intent, matched_entry_ids="program-confident-speaker")
    check("Context Turn 1 answers Confident Speaker", "confident speaker" in r_spk.reply.lower(), r_spk.reply)

    # Turn 4: Follow-up 'How does it work?' resolves to Confident Speaker
    r_follow = engine.handle_message(sid_ctx2, "How does it work?")
    check("Follow-up resolves to Confident Speaker", any("confident-speaker" in m for m in r_follow.matched_entry_ids), f"matched={r_follow.matched_entry_ids}")

    # Turn 5: Topic switch to board exams immediately updates active intent
    r_switch = engine.handle_message(sid_ctx2, "What about board exams?")
    check("Topic switch updates active intent to board exams", r_switch.intent == "board_exam" and r_switch.matched_entry_ids == ["program-senior-school"], f"intent={r_switch.intent} matched={r_switch.matched_entry_ids}")

    # -------------------------------------------------------------------------
    # 6. Intent-Specific Fallbacks (Section 8)
    # -------------------------------------------------------------------------
    print("\n--- 6. Intent-Specific Fallbacks ---")
    sid_fb = f"reg-fb-{uuid.uuid4()}"
    database.ensure_session(sid_fb)

    # Unclear board-exam question
    r_unclear_board = engine.handle_message(sid_fb, "board exams xyz unclear query with many non-matching words here")
    check("Unclear board returns board fallback", r_unclear_board.reply == personality.UNCLEAR_BOARD_FALLBACK, r_unclear_board.reply)

    # Unclear course/program question
    r_unclear_prog = engine.handle_message(sid_fb, "curriculum program details for non-existent subject astrophysics degree xyz")
    check("Unclear program returns program fallback", r_unclear_prog.reply == personality.UNCLEAR_PROGRAM_FALLBACK, r_unclear_prog.reply)

    # Unsupported details question
    r_unsupported = engine.handle_message(sid_fb, "What is the duration of the course?")
    check("Unsupported details returns unsupported fallback", r_unsupported.reply == personality.UNSUPPORTED_DETAILS_FALLBACK, r_unsupported.reply)

    # Ambiguous general question
    r_ambiguous = engine.handle_message(sid_fb, "Do you provide hostel or transport facilities for students?")
    check("Ambiguous general returns general fallback", r_ambiguous.reply == personality.AMBIGUOUS_GENERAL_FALLBACK, r_ambiguous.reply)

    # -------------------------------------------------------------------------
    # 7. Relevance and Anti-Hallucination Tests (Section 10)
    # -------------------------------------------------------------------------
    print("\n--- 7. Relevance and Anti-Hallucination Tests ---")
    sid_rel = f"reg-rel-{uuid.uuid4()}"
    database.ensure_session(sid_rel)

    # Python question does not trigger fee response
    r_py = engine.handle_message(sid_rel, "Tell me about your Python course")
    check("Python course does NOT trigger fee response", "fee" not in r_py.reply.lower() and "pricing" not in r_py.reply.lower(), r_py.reply)
    check(
        "Python course honestly clarifies not verified",
        "not currently listed" in r_py.reply.lower()
        or "confirmed details" in r_py.reply.lower()
        or "verified listing" in r_py.reply.lower()
        or "not in my verified records" in r_py.reply.lower(),
        r_py.reply,
    )

    # Board exam question does not receive generic academy intro only
    r_board_check = engine.handle_message(sid_rel, "How do you prepare students for boards?")
    check("Board exam question does not receive generic intro only", "grades 9–10" in r_board_check.reply.lower() or "grades 9-10" in r_board_check.reply.lower(), r_board_check.reply)

    # Confident speaker does not receive board-exam response
    r_speaker_check = engine.handle_message(sid_rel, "Tell me about the Confident Speaker program")
    check("Confident speaker does not receive board response", "board-exam" not in r_speaker_check.reply.lower() and "spoken english" in r_speaker_check.reply.lower(), r_speaker_check.reply)

    # Mentions Book Free Demo for booking question
    r_demo = engine.handle_message(sid_rel, "Where can I book a free demo?")
    check("Mentions Book Free Demo for demo booking", "book free demo" in r_demo.reply.lower(), r_demo.reply)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    if failures:
        print(f"FAILED ({len(failures)} checks failed):")
        for f in failures:
            safe_f = f.encode("ascii", errors="replace").decode("ascii")
            print(f"  - {safe_f}")
        sys.exit(1)
    else:
        print(f"ALL {passed_count} REGRESSION CHECKS PASSED PERFECTLY (100% SUCCESS)!")
        print("=" * 75 + "\n")


if __name__ == "__main__":
    run_tests()
