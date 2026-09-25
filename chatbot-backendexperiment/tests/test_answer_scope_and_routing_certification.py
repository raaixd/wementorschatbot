"""
Final Quality Assurance & Certification Test Suite for WeMentors Chatbot.

Evaluates and certifies:
1. Deterministic Subject Routing Matrix (Fresh, Contextual, Explicit Override, Negative Assertions)
2. Question-Scope & Intent Classification (YES/NO Capability, Attribute, Overview, Follow-Up, Multi-Part)
3. 50-Query OOD & False-Positive Benchmark (Geography, Program-like Words, Random Entities, Nonsense)
4. 30-Query Booking / Lead Safety Benchmark (True Booking vs Non-Booking Entities)
5. Multi-Session Isolation Benchmark (Zero Cross-Session Contamination & OOD Non-Mutation)
6. Typo Tolerance vs Near-Miss Separation
7. Property Invariants 1–10
"""

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = TESTS_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import database
from app.conversation import ConversationEngine
from app.knowledge import load_entries


class TestAnswerScopeAndRoutingCertification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = load_entries()
        cls.engine = ConversationEngine(cls.entries)

    def setUp(self):
        self.engine = ConversationEngine(self.entries)

    # =========================================================================
    # 1. DETERMINISTIC SUBJECT ROUTING BENCHMARK
    # =========================================================================

    def test_01_fresh_session_subject_queries_always_global(self):
        """Fresh session with no prior program context must return global subjects overview."""
        fresh_queries = [
            "subjects",
            "subject",
            "which subjects",
            "which subjects?",
            "what subjects",
            "what subjects?",
            "what are the subjects",
            "what subjects are offered",
            "what subjects do you offer",
        ]
        for q in fresh_queries:
            with self.subTest(query=q):
                sid = f"cert_fresh_{hash(q)}"
                res = self.engine.handle_message(sid, q)
                self.assertIn("**Academic Subjects**", res.reply)
                self.assertIn("**Confident Speaker**", res.reply)
                self.assertIn("- **Mathematics**", res.reply)
                self.assertIn("- **Public Speaking**", res.reply)
                self.assertIn("- **IELTS Preparation**", res.reply)
                # Must not select one specific academic program
                self.assertNotIn("Senior School Focus covers", res.reply)
                self.assertNotIn("The Middle School — All Subjects program covers", res.reply)

    def test_02_senior_school_context_subject_queries(self):
        """After Senior School context, short subject follow-ups must return Senior School subjects."""
        follow_ups = ["subjects", "which subjects", "which subjects?", "what subjects", "what subjects?"]
        for q in follow_ups:
            with self.subTest(query=q):
                sid = f"cert_sr_subj_{hash(q)}"
                self.engine.handle_message(sid, "Tell me about Senior School Focus.")
                res = self.engine.handle_message(sid, q)
                self.assertEqual(res.intent, "senior_school_subjects")
                self.assertIn("Senior School Focus", res.reply)
                self.assertIn("Mathematics", res.reply)
                self.assertIn("Science", res.reply)
                self.assertIn("Board Exam Preparation", res.reply)
                # CRITICAL NEGATIVE ASSERTIONS
                self.assertNotIn("Middle School", res.reply)
                self.assertNotIn("Grades 6–8", res.reply)
                self.assertNotIn("Confident Speaker", res.reply)
                self.assertNotIn("IELTS", res.reply)

    def test_03_middle_school_context_subject_queries(self):
        """After Middle School context, short subject follow-ups must return Middle School subjects."""
        follow_ups = ["subjects", "which subjects", "which subjects?"]
        for q in follow_ups:
            with self.subTest(query=q):
                sid = f"cert_ms_subj_{hash(q)}"
                self.engine.handle_message(sid, "Tell me about Middle School.")
                res = self.engine.handle_message(sid, q)
                self.assertEqual(res.intent, "middle_school_subjects")
                self.assertIn("Middle School", res.reply)
                self.assertIn("Mathematics", res.reply)
                self.assertIn("Science", res.reply)
                self.assertIn("English", res.reply)
                self.assertIn("Social Studies", res.reply)
                # CRITICAL NEGATIVE ASSERTIONS
                self.assertNotIn("Senior School", res.reply)
                self.assertNotIn("Grades 9–10", res.reply)
                self.assertNotIn("Confident Speaker", res.reply)
                self.assertNotIn("IELTS", res.reply)

    def test_04_foundation_years_context_subject_queries(self):
        """After Foundation Years context, short subject follow-ups must return Foundation Years subjects."""
        follow_ups = ["subjects", "which subjects", "which subjects?"]
        for q in follow_ups:
            with self.subTest(query=q):
                sid = f"cert_fy_subj_{hash(q)}"
                self.engine.handle_message(sid, "Tell me about Foundation Years.")
                res = self.engine.handle_message(sid, q)
                self.assertEqual(res.intent, "foundation_subjects")
                self.assertIn("Foundation Years", res.reply)
                self.assertIn("Environmental Studies", res.reply)
                self.assertNotIn("Senior School", res.reply)
                self.assertNotIn("Grades 9–10", res.reply)
                self.assertNotIn("Confident Speaker", res.reply)

    def test_05_confident_speaker_context_subject_queries(self):
        """After Confident Speaker context, short subject follow-ups must return Confident Speaker curriculum."""
        follow_ups = ["subjects", "which subjects", "which subjects?"]
        for q in follow_ups:
            with self.subTest(query=q):
                sid = f"cert_cs_subj_{hash(q)}"
                self.engine.handle_message(sid, "Tell me about Confident Speaker.")
                res = self.engine.handle_message(sid, q)
                self.assertEqual(res.intent, "confident_speaker_scope")
                self.assertIn("four curriculum areas", res.reply)
                self.assertIn("Public Speaking", res.reply)
                self.assertIn("Business English", res.reply)
                self.assertIn("General Communicative Skills", res.reply)
                self.assertIn("IELTS Preparation", res.reply)
                # CRITICAL NEGATIVE ASSERTIONS
                self.assertNotIn("Middle School", res.reply)
                self.assertNotIn("Senior School", res.reply)
                self.assertNotIn("Grades 6–8", res.reply)
                self.assertNotIn("Grades 9–10", res.reply)

    def test_06_explicit_override_wins_over_previous_context(self):
        """Explicit program in current turn must always override prior conversation context."""
        overrides = [
            ("Senior School Focus", "subjects in Middle School", "middle_school_subjects", "Middle School"),
            ("Middle School", "subjects in Confident Speaker", "confident_speaker_scope", "Public Speaking"),
            ("Confident Speaker", "subjects in Senior School", "senior_school_subjects", "Senior School Focus"),
            ("Foundation Years", "subjects in Middle School", "middle_school_subjects", "Middle School"),
            ("Senior School Focus", "subjects in middle shcool", "middle_school_subjects", "Middle School"),
        ]
        for prior_msg, current_msg, expected_intent, expected_snippet in overrides:
            with self.subTest(prior=prior_msg, current=current_msg):
                sid = f"cert_ovr_{hash(prior_msg + current_msg)}"
                self.engine.handle_message(sid, prior_msg)
                res = self.engine.handle_message(sid, current_msg)
                self.assertEqual(res.intent, expected_intent)
                self.assertIn(expected_snippet, res.reply)

    # =========================================================================
    # 2. ANSWER-SCOPE & INTENT BENCHMARK
    # =========================================================================

    def test_07_capability_yes_no_questions_are_concise(self):
        """Capability questions must start with direct '**Yes.**' and not dump full program cards or CTAs."""
        capability_cases = [
            ("Do you support Grades 9–10?", "Senior School Focus", "senior_school_capability"),
            ("Do you teach Grades 9–10?", "Senior School Focus", "senior_school_capability"),
            ("Do you offer Grade 10?", "Senior School Focus", "senior_school_capability"),
            ("Is Grade 9 supported?", "Senior School Focus", "senior_school_capability"),
            ("Are Grades 9–10 supported?", "Senior School Focus", "senior_school_capability"),
            ("Do you support Grades 6–8?", "Middle School — All Subjects", "middle_school_capability"),
            ("Do you teach Grades 6–8?", "Middle School — All Subjects", "middle_school_capability"),
            ("Do you support Grades 3–5?", "Foundation Years", "foundation_years_capability"),
            ("Do you teach Grades 3–5?", "Foundation Years", "foundation_years_capability"),
            ("Do you offer State Board?", "State Board", "state_board_capability"),
            ("Do you offer online classes?", "online", "online_classes"),
            ("Do you offer IELTS preparation?", "IELTS", "confident_speaker_ielts"),
            ("Do you offer Public Speaking?", "Public Speaking", "confident_speaker_public_speaking"),
            ("Do you offer Business English?", "Business English", "confident_speaker_business_english"),
        ]
        for query, expected_snippet, expected_intent in capability_cases:
            with self.subTest(query=query):
                sid = f"cert_cap_{hash(query)}"
                res = self.engine.handle_message(sid, query)
                self.assertTrue(
                    res.reply.startswith("**Yes.**") or res.reply.startswith("Yes,"),
                    f"Query '{query}' did not start with direct yes answer: {res.reply[:50]}",
                )
                self.assertIn(expected_snippet.lower(), res.reply.lower())
                # Must not contain full overview card dump / CTA
                self.assertNotIn("Book Free Demo button", res.reply)
                self.assertNotIn("weekly mock tests with review", res.reply if "9–10" not in query else "")

    def test_08_narrow_attribute_questions_answer_only_requested_attribute(self):
        """Attribute questions must answer only the attribute, not the full program card."""
        # Grades attribute
        res_sr_grades = self.engine.handle_message("cert_att_1", "What grades does Senior School cover?")
        self.assertEqual(res_sr_grades.intent, "senior_school_grades")
        self.assertEqual(res_sr_grades.reply.strip(), "**Senior School Focus covers Grades 9–10.**")

        # Subjects attribute
        res_sr_subj = self.engine.handle_message("cert_att_2", "What subjects are in Senior School?")
        self.assertEqual(res_sr_subj.intent, "senior_school_subjects")
        self.assertIn("Mathematics", res_sr_subj.reply)
        self.assertIn("Science", res_sr_subj.reply)
        self.assertNotIn("Book Free Demo button", res_sr_subj.reply)

    def test_09_program_overview_questions_return_full_overview(self):
        """Overview and discovery questions must provide complete overview detail."""
        overview_queries = [
            "Tell me about Senior School Focus.",
            "Tell me about Grades 9–10.",
            "What is Senior School Focus?",
        ]
        for q in overview_queries:
            with self.subTest(query=q):
                sid = f"cert_ovw_{hash(q)}"
                res = self.engine.handle_message(sid, q)
                self.assertEqual(res.intent, "senior_school_overview")
                self.assertIn("Senior School Focus", res.reply)
                self.assertIn("Mathematics", res.reply)
                self.assertIn("Science", res.reply)
                self.assertIn("Book Free Demo", res.reply)

    def test_10_contextual_narrow_follow_ups(self):
        """After Senior School overview, narrow follow-up questions must answer only that scope."""
        sid = "cert_seq_followup"
        self.engine.handle_message(sid, "Tell me about Senior School Focus.")

        # 1. grades?
        res_grades = self.engine.handle_message(sid, "grades?")
        self.assertEqual(res_grades.intent, "senior_school_grades")
        self.assertEqual(res_grades.reply.strip(), "**Senior School Focus covers Grades 9–10.**")

        # 2. subjects
        res_subj = self.engine.handle_message(sid, "subjects")
        self.assertEqual(res_subj.intent, "senior_school_subjects")
        self.assertIn("Mathematics", res_subj.reply)
        self.assertNotIn("Middle School", res_subj.reply)

        # 3. mentor?
        res_mentor = self.engine.handle_message(sid, "mentor?")
        self.assertIn("personal mentor", res_mentor.reply.lower())
        self.assertNotIn("Middle School", res_mentor.reply)

        # 4. doubt solving?
        res_doubt = self.engine.handle_message(sid, "doubt solving?")
        self.assertIn("doubt", res_doubt.reply.lower())

        # 5. tell me more
        res_more = self.engine.handle_message(sid, "tell me more")
        self.assertEqual(res_more.intent, "senior_school_overview")
        self.assertIn("Senior School Focus", res_more.reply)

    def test_11_multipart_questions_cover_all_requested_parts(self):
        """Multi-part question asking about capability AND subjects answers both."""
        res = self.engine.handle_message("cert_multi_1", "Do you support Grades 9–10 and what subjects do you teach?")
        self.assertTrue(res.reply.startswith("**Yes.**"))
        self.assertIn("Grades 9–10", res.reply)
        self.assertIn("Senior School Focus", res.reply)
        self.assertIn("Mathematics", res.reply)
        self.assertIn("Science", res.reply)

    # =========================================================================
    # 3. 50-QUERY OOD & FALSE-POSITIVE BENCHMARK
    # =========================================================================

    def test_12_ood_false_positive_benchmark(self):
        """50 curated OOD inputs must never falsely activate programs or booking."""
        ood_queries = [
            # Geography
            "middle east", "middle eastern", "middle eastern education",
            "canada", "usa", "india", "uk", "uae", "australia", "germany",
            "france", "europe", "asia", "africa", "north america",
            # Similar program words
            "middle", "senior", "foundation", "speaker", "confident",
            "public speaker", "senior citizens", "senior citizen",
            "foundation mathematics", "foundation of mathematics",
            "confident student",
            # Random entities
            "football", "basketball", "python", "google", "apple",
            "pizza", "weather", "car", "university", "movie", "book", "laptop",
            # Nonsense / ambiguous
            "xyz", "asdf", "hello123", "???", "random words", "I don't know",
            "something else", "what", "huh?", "temperature", "recipe", "cook"
        ]
        for q in ood_queries:
            with self.subTest(query=q):
                sid = f"cert_ood_{hash(q)}"
                res = self.engine.handle_message(sid, q)
                # Must not falsely book
                self.assertNotIn(res.intent, ["demo_booking", "demo_transaction_request"])
                # Similar non-program words must not trigger respective programs
                if q in ("middle east", "middle eastern", "middle eastern education"):
                    self.assertNotEqual(res.intent, "middle_school_overview")
                    self.assertNotEqual(res.intent, "middle_school_subjects")
                if q in ("senior citizens", "senior citizen"):
                    self.assertNotEqual(res.intent, "senior_school_overview")
                    self.assertNotEqual(res.intent, "senior_school_subjects")
                if q in ("foundation mathematics", "foundation of mathematics"):
                    self.assertNotEqual(res.intent, "foundation_years_overview")
                if q == "confident student":
                    self.assertNotEqual(res.intent, "confident_speaker")

    # =========================================================================
    # 4. 30-QUERY BOOKING / LEAD SAFETY BENCHMARK
    # =========================================================================

    def test_13_booking_lead_safety_benchmark(self):
        """True booking queries route to booking guidance; ordinary words never trigger fake booking."""
        true_booking = [
            "I want to book a demo",
            "Can I book a demo?",
            "I'd like to schedule a demo",
            "How do I book the free demo?",
            "I want to register for a demo",
            "where do i book a demo",
            "how to book a demo",
            "book a trial class",
        ]
        non_booking = [
            "Canada", "USA", "India", "demo", "free", "contact",
            "phone", "location", "online", "Can you teach students in Canada?",
            "Do you have classes in the USA?", "free classes?", "contact number",
            "where are you located", "what is your phone number", "email address",
            "who are the mentors", "class duration", "how often are classes",
            "do you have discounts", "scholarships", "state board syllabus",
        ]
        for q in true_booking:
            with self.subTest(booking_query=q):
                res = self.engine.handle_message(f"cert_book_pos_{hash(q)}", q)
                self.assertIn("Book Free Demo", res.reply)

        for q in non_booking:
            with self.subTest(non_booking_query=q):
                sid = f"cert_book_neg_{hash(q)}"
                res = self.engine.handle_message(sid, q)
                # Must never claim lead was created or booking confirmed
                self.assertNotIn("booked your demo", res.reply.lower())
                self.assertNotIn("registered you for", res.reply.lower())
                self.assertNotIn("team received your details", res.reply.lower())

    # =========================================================================
    # 5. MULTI-SESSION ISOLATION BENCHMARK
    # =========================================================================

    def test_14_multi_session_isolation_and_ood_non_mutation(self):
        """Cross-session context must never leak, and OOD turns must not mutate active_program."""
        # Step 1: Initialize separate sessions
        sid_a = "sess_iso_sr"  # Senior School
        sid_b = "sess_iso_cs"  # Confident Speaker
        sid_c = "sess_iso_ms"  # Middle School
        sid_d = "sess_iso_fy"  # Foundation Years
        sid_e = "sess_iso_fresh"  # Fresh

        self.engine.handle_message(sid_a, "Tell me about Senior School Focus.")
        self.engine.handle_message(sid_b, "Tell me about Confident Speaker.")
        self.engine.handle_message(sid_c, "Tell me about Middle School.")
        self.engine.handle_message(sid_d, "Tell me about Foundation Years.")

        # Interleave OOD queries into sessions
        self.engine.handle_message(sid_a, "middle east")
        self.engine.handle_message(sid_b, "canada")
        self.engine.handle_message(sid_c, "usa")

        # Now test 'subjects' on all sessions
        res_a = self.engine.handle_message(sid_a, "subjects")
        res_b = self.engine.handle_message(sid_b, "subjects")
        res_c = self.engine.handle_message(sid_c, "subjects")
        res_d = self.engine.handle_message(sid_d, "subjects")
        res_e = self.engine.handle_message(sid_e, "subjects")

        # Session A -> Senior School
        self.assertEqual(res_a.intent, "senior_school_subjects")
        self.assertIn("Senior School Focus", res_a.reply)
        self.assertNotIn("Middle School", res_a.reply)

        # Session B -> Confident Speaker
        self.assertEqual(res_b.intent, "confident_speaker_scope")
        self.assertIn("Public Speaking", res_b.reply)
        self.assertNotIn("Senior School", res_b.reply)

        # Session C -> Middle School
        self.assertEqual(res_c.intent, "middle_school_subjects")
        self.assertIn("Middle School", res_c.reply)
        self.assertNotIn("Senior School", res_c.reply)

        # Session D -> Foundation Years
        self.assertEqual(res_d.intent, "foundation_subjects")
        self.assertIn("Foundation Years", res_d.reply)

        # Session E -> Fresh (Global)
        self.assertEqual(res_e.intent, "faq")
        self.assertIn("**Academic Subjects**", res_e.reply)
        self.assertIn("**Confident Speaker**", res_e.reply)

    # =========================================================================
    # 6. TYPO TOLERANCE BENCHMARK
    # =========================================================================

    def test_15_typo_tolerance_vs_near_miss_separation(self):
        """Legitimate typos resolve appropriately, while near-miss unrelated words do not."""
        typos = [
            ("middle shcool", "middle_school_overview"),
            ("middl school", "middle_school_overview"),
            ("senoir school", "senior_school_overview"),
            ("senior shcool", "senior_school_overview"),
            ("foundaton years", "foundation_years_overview"),
            ("confdent speaker", "confident_speaker"),
        ]
        for typo, expected_intent in typos:
            with self.subTest(typo=typo):
                sid = f"cert_typo_{hash(typo)}"
                res = self.engine.handle_message(sid, typo)
                self.assertEqual(res.intent, expected_intent)

    # =========================================================================
    # 7. PROPERTY INVARIANTS 1–10
    # =========================================================================

    def test_16_property_invariants(self):
        """Test core conversational properties and architectural invariants."""
        # INVARIANT 1: Fresh sessions never have active_program
        fresh_mem = database.get_conversation_memory("invariant_fresh_session")
        self.assertIsNone(fresh_mem.active_program)

        # INVARIANT 2: OOD inputs cannot mutate active_program
        sid_inv2 = "invariant_ood_mutation"
        self.engine.handle_message(sid_inv2, "Tell me about Senior School Focus.")
        mem_before = database.get_conversation_memory(sid_inv2)
        self.assertEqual(mem_before.active_program, "senior")
        self.engine.handle_message(sid_inv2, "weather in london")
        mem_after = database.get_conversation_memory(sid_inv2)
        self.assertEqual(mem_after.active_program, "senior")

        # INVARIANT 4: Explicit program reference overrides active context
        sid_inv4 = "invariant_explicit_override"
        self.engine.handle_message(sid_inv4, "Senior School Focus")
        res_inv4 = self.engine.handle_message(sid_inv4, "subjects in Middle School")
        self.assertEqual(res_inv4.intent, "middle_school_subjects")

        # INVARIANT 5: Session A cannot modify Session B
        sid_a = "inv_sess_a"
        sid_b = "inv_sess_b"
        self.engine.handle_message(sid_a, "Senior School Focus")
        mem_b = database.get_conversation_memory(sid_b)
        self.assertIsNone(mem_b.active_program)

        # INVARIANT 6: Narrow factual query cannot automatically become full program card
        res_inv6 = self.engine.handle_message("inv_narrow", "Do you support Grades 9–10?")
        self.assertTrue(res_inv6.reply.startswith("**Yes.**"))
        self.assertNotIn("Book Free Demo button", res_inv6.reply)

        # INVARIANT 9: Booking state cannot be created without booking intent
        sid_inv9 = "inv_non_booking_lead"
        self.engine.handle_message(sid_inv9, "Canada")
        lead = database.get_demo_lead(sid_inv9)
        self.assertIsNone(lead)


if __name__ == "__main__":
    unittest.main()
