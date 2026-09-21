"""
Regression tests for mentor matching and inquiry handling across grades, boards, and subjects,
verifying the clean two-paragraph response style matching user specifications.
"""

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import database
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
        safe_detail = detail.encode("ascii", errors="replace").decode("ascii")
        print(f"[FAIL] {name} -> {safe_detail}")


def run_tests():
    database.init_db()
    engine = ConversationEngine(load_entries())

    print("\n" + "=" * 80)
    print("RUNNING MENTOR MATCHING & INQUIRY REGRESSION TESTS")
    print("=" * 80)

    # 1. Exact query from user screenshot: "Can you find a mentor for Grade 7 ICSE Math & Science?"
    r1 = engine.handle_message("t-m1", "Can you find a mentor for Grade 7 ICSE Math & Science?")
    r1_text = r1.reply
    print(f"\n[Response Preview for 'Can you find a mentor for Grade 7 ICSE Math & Science?']:\n{r1_text}\n")
    check("1a. Grade 7 ICSE Math & Science intent is mentor_matching", r1.intent == "mentor_matching", r1.intent)
    check("1b. Starts with affirmative **Yes!**", r1_text.startswith("**Yes!**") or r1_text.startswith("Yes!"), r1_text)
    check("1c. Mentions dedicated 1:1 mentor", "dedicated 1:1 mentor" in r1_text or "1:1 mentor" in r1_text, r1_text)
    check("1d. Mentions specialized in the ICSE curriculum", "specialized in the icse curriculum" in r1_text.lower(), r1_text)
    check("1e. Mentions concept mastery, doubt clearing, weekly progress updates", "concept mastery" in r1_text.lower() and "doubt clearing" in r1_text.lower() and "weekly progress" in r1_text.lower(), r1_text)
    check("1f. Closes with demo invite", "ready to experience a session? you can book a free 30-minute demo class today!" in r1_text.lower().replace("\u2011", "-") or "free 30-minute demo class" in r1_text.lower().replace("\u2011", "-"), r1_text)
    check("1g. Avoids duplicate demo CTA", r1_text.count("Book Free Demo") <= 1 and r1_text.count("demo class") <= 1, r1_text)
    check("1h. Avoids generic fallback", "academic programs (grades 3" not in r1_text.lower(), r1_text)

    # 2. General Grade + Board variations
    r2 = engine.handle_message("t-m2", "can you find a mentor for grade 7 icse and math")
    check("2a. Grade 7 ICSE Math recognized as mentor_matching", r2.intent == "mentor_matching", r2.intent)
    check("2b. Grade 7 ICSE Math starts with Yes!", r2.reply.startswith("**Yes!**") or r2.reply.startswith("Yes!"), r2.reply)
    check("2c. Grade 7 ICSE Math mentions ICSE", "icse" in r2.reply.lower(), r2.reply)

    r3 = engine.handle_message("t-m3", "looking for a mentor for grade 7 icse math")
    check("3a. Looking for mentor grade 7 icse math recognized", r3.intent == "mentor_matching", r3.intent)
    check("3b. Looking for mentor avoids fallback", "academic programs (grades 3" not in r3.reply.lower(), r3.reply)

    r4 = engine.handle_message("t-m4", "can you find a mentor for grade 5 cbse science")
    check("4a. Grade 5 CBSE Science recognized", r4.intent == "mentor_matching", r4.intent)
    check("4b. Grade 5 CBSE Science mentions CBSE curriculum", "cbse" in r4.reply.lower(), r4.reply)

    r5 = engine.handle_message("t-m5", "do you have mentors for grade 9 cbse math")
    check("5a. Grade 9 CBSE Math recognized", r5.intent == "mentor_matching", r5.intent)
    check("5b. Grade 9 CBSE Math mentions CBSE curriculum", "cbse" in r5.reply.lower(), r5.reply)
    check("5c. Grade 9 CBSE Math preserves 2-paragraph style without senior school boilerplate", "rotating roster" not in r5.reply.lower(), r5.reply)

    r6 = engine.handle_message("t-m6", "can you allocate a mentor for grade 7")
    check("6a. Allocate mentor for grade 7 recognized", r6.intent == "mentor_matching", r6.intent)
    check("6b. Allocate mentor mentions mentor", "mentor" in r6.reply.lower(), r6.reply)
    check("6c. Allocate mentor starts with Yes!", r6.reply.startswith("**Yes!**") or r6.reply.startswith("Yes!"), r6.reply)

    # 3. Guardrail integrity
    r7 = engine.handle_message("t-m7", "can you find a mentor for grade 1 or 2")
    check("7a. Grade 1-2 unavailable preserved", r7.intent == "grade_1_2_unavailable", r7.intent)

    r8 = engine.handle_message("t-m8", "What about Grade 7 Maths sessions?")
    check("8a. Grade 7 Maths sessions preserved", r8.intent == "grade7_maths_sessions", r8.intent)

    # 4. Deterministic template fallback verification (NO_API_MODE)
    from app import config as app_cfg
    orig_llm = app_cfg.LLM_ENABLED
    try:
        app_cfg.LLM_ENABLED = False
        r_tmpl = engine.handle_message("t-m9", "Can you find a mentor for Grade 7 ICSE Math & Science?")
        check("9a. Template mode intent is mentor_matching", r_tmpl.intent == "mentor_matching", r_tmpl.intent)
        check("9b. Template mode starts with **Yes!**", r_tmpl.reply.startswith("**Yes!**"), r_tmpl.reply)
        check("9c. Template mode matches ICSE curriculum", "the icse curriculum" in r_tmpl.reply.lower(), r_tmpl.reply)
        check("9d. Template mode includes closing demo invite", "ready to experience a session? you can book a free 30-minute demo class today!" in r_tmpl.reply.lower(), r_tmpl.reply)
        check("9e. Template mode avoids duplicate CTA", r_tmpl.reply.count("Book Free Demo") <= 1 and r_tmpl.reply.count("demo class") <= 1, r_tmpl.reply)
    finally:
        app_cfg.LLM_ENABLED = orig_llm

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed_count} passed, {len(failures)} failed")
    print("=" * 80)

    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
