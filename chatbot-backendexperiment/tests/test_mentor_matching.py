"""
Regression tests for mentor matching and inquiry handling across grades, boards, and subjects.
"""

import sys
from pathlib import Path

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

    # 1. Grade 7 ICSE Math mentor request
    r1 = engine.handle_message("t-m1", "can you find a mentor for grade 7 icse and math")
    r1_lower = r1.reply.lower()
    check("1a. Grade 7 ICSE Math intent recognized", r1.intent in ("mentor_matching", "faq"), r1.intent)
    check("1b. Grade 7 ICSE Math mentions Middle School / Grade 7", "middle school" in r1_lower or "grade 7" in r1_lower or "grades 6" in r1_lower, r1.reply)
    check("1c. Grade 7 ICSE Math mentions ICSE", "icse" in r1_lower, r1.reply)
    check("1d. Grade 7 ICSE Math mentions Math", "math" in r1_lower, r1.reply)
    check("1e. Grade 7 ICSE Math mentions mentor", "mentor" in r1_lower, r1.reply)
    check("1f. Grade 7 ICSE Math avoids generic fallback", "academic programs (grades 3" not in r1_lower, r1.reply)

    # 2. General Grade + Board variations
    r2 = engine.handle_message("t-m2", "can you find me a mentor for grade 7 icse")
    check("2a. Grade 7 ICSE recognized", r2.intent in ("mentor_matching", "faq"), r2.intent)
    check("2b. Grade 7 ICSE avoids fallback", "academic programs (grades 3" not in r2.reply.lower(), r2.reply)

    r3 = engine.handle_message("t-m3", "looking for a mentor for grade 7 icse math")
    check("3a. Looking for mentor grade 7 icse math recognized", r3.intent in ("mentor_matching", "faq"), r3.intent)
    check("3b. Looking for mentor avoids fallback", "academic programs (grades 3" not in r3.reply.lower(), r3.reply)

    r4 = engine.handle_message("t-m4", "can you find a mentor for grade 5 cbse science")
    check("4a. Grade 5 CBSE Science recognized", r4.intent in ("mentor_matching", "faq"), r4.intent)
    check("4b. Grade 5 CBSE Science mentions foundation or science", "foundation" in r4.reply.lower() or "science" in r4.reply.lower(), r4.reply)

    r5 = engine.handle_message("t-m5", "do you have mentors for grade 9 cbse math")
    check("5a. Grade 9 CBSE Math recognized", r5.intent in ("mentor_matching", "faq"), r5.intent)
    check("5b. Grade 9 CBSE Math mentions 9 or 10 or senior", "9" in r5.reply.lower() or "senior" in r5.reply.lower(), r5.reply)

    r6 = engine.handle_message("t-m6", "can you allocate a mentor for grade 7")
    check("6a. Allocate mentor for grade 7 recognized", r6.intent in ("mentor_matching", "faq"), r6.intent)
    check("6b. Allocate mentor mentions mentor", "mentor" in r6.reply.lower(), r6.reply)

    # 3. Guardrail integrity
    r7 = engine.handle_message("t-m7", "can you find a mentor for grade 1 or 2")
    check("7a. Grade 1-2 unavailable preserved", r7.intent == "grade_1_2_unavailable", r7.intent)

    r8 = engine.handle_message("t-m8", "What about Grade 7 Maths sessions?")
    check("8a. Grade 7 Maths sessions preserved", r8.intent == "grade7_maths_sessions", r8.intent)

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed_count} passed, {len(failures)} failed")
    print("=" * 80)

    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
