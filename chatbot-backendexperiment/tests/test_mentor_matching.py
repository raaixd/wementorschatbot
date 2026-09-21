"""
Regression tests for mentor matching and inquiry handling across grades, boards, and subjects,
verifying concise, natural two-paragraph responses starting with 'Yes,' and closing with the
concise demo CTA.
"""

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import config as app_cfg, database, llm
from app.conversation import ConversationEngine
from app.knowledge import load_entries

failures = []
passed_count = 0

CONCISE_DEMO_CTA = "Ready to experience a session? You can book a free 30-minute demo using the Book Free Demo button."


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

    # -------------------------------------------------------------------------
    # 1. Exact query: "Can you find a mentor for Grade 7 ICSE Math & Science?"
    # -------------------------------------------------------------------------
    # Test with template mode first for deterministic assertion
    orig_llm = app_cfg.LLM_ENABLED
    app_cfg.LLM_ENABLED = False
    try:
        r1 = engine.handle_message("t-m1", "Can you find a mentor for Grade 7 ICSE Math & Science?")
        r1_text = r1.reply
        print(f"\n[Response Preview for 'Can you find a mentor for Grade 7 ICSE Math & Science?']:\n{r1_text}\n")
        check("1a. Grade 7 ICSE Math & Science intent is mentor_matching", r1.intent == "mentor_matching", r1.intent)
        check("1b. Starts with affirmative 'Yes,'", r1_text.startswith("Yes,"), r1_text)
        check("1b_no_exclamation. Does not start with 'Yes!' or '**Yes!**'", not r1_text.startswith("Yes!") and not r1_text.startswith("**Yes!**"), r1_text)
        check("1c. Mentions dedicated 1:1 mentor", "dedicated 1:1 mentor" in r1_text or "1:1 mentor" in r1_text, r1_text)
        check("1d. Mentions specialized in the ICSE curriculum", "the icse curriculum" in r1_text.lower(), r1_text)
        check("1e. Mentions concept mastery, doubt clearing, ongoing progress support", "concept mastery" in r1_text.lower() and "doubt clearing" in r1_text.lower() and "progress support" in r1_text.lower(), r1_text)
        check("1f. Closes with concise demo invite", CONCISE_DEMO_CTA.lower() in r1_text.lower(), r1_text)
        check("1g. Avoids duplicate demo CTA", r1_text.count("Book Free Demo") <= 1 and r1_text.count("demo") <= 1, r1_text)
        check("1h. Avoids verbose boilerplate", "students receive individual attention and dedicated mentor feedback" not in r1_text.lower() and "rotating roster" not in r1_text.lower(), r1_text)
        check("1i. Avoids generic fallback", "academic programs (grades 3" not in r1_text.lower(), r1_text)

        # -------------------------------------------------------------------------
        # 2. REQUIRED TEST QUERIES A-E
        # -------------------------------------------------------------------------
        print("\n--- Testing Required Queries A-E ---")

        # Query A: "Can I get a mentor?"
        r_a = engine.handle_message("t-qa", "Can I get a mentor?")
        check("A1. 'Can I get a mentor?' intent is mentor_matching", r_a.intent == "mentor_matching", r_a.intent)
        check("A2. 'Can I get a mentor?' starts with 'Yes,'", r_a.reply.startswith("Yes,"), r_a.reply)
        check("A3. 'Can I get a mentor?' does not use exclamation in affirmative", not r_a.reply.startswith("Yes!"), r_a.reply)
        check("A4. 'Can I get a mentor?' confirms 1:1 mentoring", "1:1 mentoring" in r_a.reply or "1:1 mentor" in r_a.reply, r_a.reply)
        check("A5. 'Can I get a mentor?' includes concept mastery and doubt clearing", "concept mastery" in r_a.reply.lower() and "doubt clearing" in r_a.reply.lower(), r_a.reply)
        check("A6. 'Can I get a mentor?' uses concise demo CTA", CONCISE_DEMO_CTA.lower() in r_a.reply.lower(), r_a.reply)
        check("A7. 'Can I get a mentor?' excludes verbose repeated sentences", "students receive individual attention and dedicated mentor feedback" not in r_a.reply.lower(), r_a.reply)

        # Query B: "Do you provide personal mentors?"
        r_b = engine.handle_message("t-qb", "Do you provide personal mentors?")
        check("B1. 'Do you provide personal mentors?' intent is mentor_matching", r_b.intent == "mentor_matching", r_b.intent)
        check("B2. 'Do you provide personal mentors?' starts with 'Yes,'", r_b.reply.startswith("Yes,"), r_b.reply)
        check("B3. 'Do you provide personal mentors?' has concise demo CTA", CONCISE_DEMO_CTA.lower() in r_b.reply.lower(), r_b.reply)
        check("B4. 'Do you provide personal mentors?' confirms 1:1 mentoring", "1:1 mentoring" in r_b.reply or "1:1 mentor" in r_b.reply, r_b.reply)

        # Query C: "I want a mentor for class 5"
        r_c = engine.handle_message("t-qc", "I want a mentor for class 5")
        check("C1. 'I want a mentor for class 5' intent is mentor_matching", r_c.intent == "mentor_matching", r_c.intent)
        check("C2. 'I want a mentor for class 5' starts with 'Yes,'", r_c.reply.startswith("Yes,"), r_c.reply)
        check("C3. 'I want a mentor for class 5' incorporates Grade 5 context", "grade 5" in r_c.reply.lower(), r_c.reply)
        check("C4. 'I want a mentor for class 5' mentions core subjects", "mathematics" in r_c.reply.lower() or "science" in r_c.reply.lower(), r_c.reply)
        check("C5. 'I want a mentor for class 5' has concise demo CTA", CONCISE_DEMO_CTA.lower() in r_c.reply.lower(), r_c.reply)

        # Query D: "Can my child get a mentor for Grade 7 ICSE?"
        r_d = engine.handle_message("t-qd", "Can my child get a mentor for Grade 7 ICSE?")
        check("D1. 'Can my child get a mentor for Grade 7 ICSE?' intent is mentor_matching", r_d.intent == "mentor_matching", r_d.intent)
        check("D2. 'Can my child get a mentor for Grade 7 ICSE?' starts with 'Yes,'", r_d.reply.startswith("Yes,"), r_d.reply)
        check("D3. 'Can my child get a mentor for Grade 7 ICSE?' incorporates ICSE and Grade 7", "icse" in r_d.reply.lower() and "grade 7" in r_d.reply.lower(), r_d.reply)
        check("D4. 'Can my child get a mentor for Grade 7 ICSE?' has concise demo CTA", CONCISE_DEMO_CTA.lower() in r_d.reply.lower(), r_d.reply)

        # Query E: "I want a mentor for Grade 7 Math"
        r_e = engine.handle_message("t-qe", "I want a mentor for Grade 7 Math")
        check("E1. 'I want a mentor for Grade 7 Math' intent is mentor_matching", r_e.intent == "mentor_matching", r_e.intent)
        check("E2. 'I want a mentor for Grade 7 Math' starts with 'Yes,'", r_e.reply.startswith("Yes,"), r_e.reply)
        check("E3. 'I want a mentor for Grade 7 Math' incorporates Grade 7 Math", "grade 7 math" in r_e.reply.lower() or ("grade 7" in r_e.reply.lower() and "math" in r_e.reply.lower()), r_e.reply)
        check("E4. 'I want a mentor for Grade 7 Math' has concise demo CTA", CONCISE_DEMO_CTA.lower() in r_e.reply.lower(), r_e.reply)

        # -------------------------------------------------------------------------
        # 3. Additional Common Inquiries
        # -------------------------------------------------------------------------
        print("\n--- Testing Additional Variations ---")
        additional_queries = [
            ("Do you provide mentors?", "Do you provide mentors?"),
            ("Can my child get a mentor?", "Can my child get a mentor?"),
            ("I want a mentor", "I want a mentor"),
            ("Is personal mentoring available?", "Is personal mentoring available?"),
            ("Can I have a personal mentor?", "Can I have a personal mentor?"),
        ]
        for label, q in additional_queries:
            res = engine.handle_message(f"t-add-{hash(q)}", q)
            check(f"Addnl: '{label}' starts with 'Yes,'", res.reply.startswith("Yes,"), res.reply)
            check(f"Addnl: '{label}' has no exclamation in affirmative", not res.reply.startswith("Yes!"), res.reply)
            check(f"Addnl: '{label}' intent is mentor_matching", res.intent == "mentor_matching", res.intent)
            check(f"Addnl: '{label}' has concise demo CTA", CONCISE_DEMO_CTA.lower() in res.reply.lower(), res.reply)

        # -------------------------------------------------------------------------
        # 4. General Grade + Board variations
        # -------------------------------------------------------------------------
        print("\n--- Testing Board Variations ---")
        r2 = engine.handle_message("t-m2", "can you find a mentor for grade 7 icse and math")
        check("2a. Grade 7 ICSE Math recognized as mentor_matching", r2.intent == "mentor_matching", r2.intent)
        check("2b. Grade 7 ICSE Math starts with Yes,", r2.reply.startswith("Yes,"), r2.reply)
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
        check("6c. Allocate mentor starts with Yes,", r6.reply.startswith("Yes,"), r6.reply)

        # -------------------------------------------------------------------------
        # 5. Guardrail integrity
        # -------------------------------------------------------------------------
        print("\n--- Testing Guardrail Integrity ---")
        r7 = engine.handle_message("t-m7", "can you find a mentor for grade 1 or 2")
        check("7a. Grade 1-2 unavailable preserved", r7.intent == "grade_1_2_unavailable", r7.intent)

        r8 = engine.handle_message("t-m8", "What about Grade 7 Maths sessions?")
        check("8a. Grade 7 Maths sessions preserved", r8.intent == "grade7_maths_sessions", r8.intent)

        r_pm = engine.handle_message("t-pm", "What is personalized mentoring?")
        check("8b. General mentoring concept definition preserved", r_pm.intent == "personalized_mentoring_general", r_pm.intent)

        # -------------------------------------------------------------------------
        # 6. LLM Provider Output Sanitization Verification
        # -------------------------------------------------------------------------
        print("\n--- Testing LLM Output Sanitization with FakeProvider ---")
        app_cfg.LLM_ENABLED = True

        class FakeEnthusiasticProvider(llm.LLMProvider):
            name = "fake"

            def generate(self, system_prompt, messages):
                return (
                    "**Yes!** We match your child with a dedicated 1:1 mentor specialized in their curriculum. "
                    "Each session includes personalized concept mastery and doubt clearing.\n\n"
                    "Ready to experience a session? You can book a free 30-minute demo class today! "
                    "Students receive individual attention and dedicated mentor feedback."
                )

        llm.set_provider_for_testing(FakeEnthusiasticProvider())
        r_llm = engine.handle_message("t-llm-sanitize", "Can I get a mentor?")
        check("6a. LLM output sanitized to start with 'Yes,'", r_llm.reply.startswith("Yes,"), r_llm.reply)
        check("6b. LLM output strips '**Yes!**'", "**Yes!**" not in r_llm.reply and not r_llm.reply.startswith("Yes!"), r_llm.reply)
        check("6c. LLM output strips verbose demo copy", "students receive individual attention and dedicated mentor feedback" not in r_llm.reply.lower(), r_llm.reply)
        check("6d. LLM output uses concise demo CTA", CONCISE_DEMO_CTA.lower() in r_llm.reply.lower(), r_llm.reply)

    finally:
        app_cfg.LLM_ENABLED = orig_llm
        llm.reset_provider_cache()

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed_count} passed, {len(failures)} failed")
    print("=" * 80)

    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
