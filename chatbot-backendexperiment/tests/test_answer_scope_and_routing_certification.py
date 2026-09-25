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
import uuid
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

    def test_17_active_program_isolation_regression(self):
        """Certify that assistant responses, retrieval results, and generic discovery queries

        never mutate or create active_program context across 15 canonical regression scenarios.
        """
        import uuid

        def _turn(sid: str, msg: str):
            database.log_message(sid, "user", msg)
            res = self.engine.handle_message(sid, msg)
            database.log_message(sid, "assistant", res.reply, intent=res.intent)
            mem = database.get_conversation_memory(sid)
            return res, mem

        # 1. Fresh: subjects -> subjects => global -> global
        sid1 = f"reg_01_{uuid.uuid4()}"
        r1_1, m1_1 = _turn(sid1, "subjects")
        self.assertEqual(r1_1.intent, "faq")
        self.assertIsNone(m1_1.active_program)
        self.assertIn("**Academic Subjects**", r1_1.reply)
        self.assertIn("**Confident Speaker**", r1_1.reply)
        r1_2, m1_2 = _turn(sid1, "subjects")
        self.assertEqual(r1_2.intent, "faq")
        self.assertIsNone(m1_2.active_program)
        self.assertIn("**Academic Subjects**", r1_2.reply)
        self.assertIn("**Confident Speaker**", r1_2.reply)

        # 2. Fresh: subjects -> which subjects => global -> global
        sid2 = f"reg_02_{uuid.uuid4()}"
        _turn(sid2, "subjects")
        r2_2, m2_2 = _turn(sid2, "which subjects")
        self.assertEqual(r2_2.intent, "faq")
        self.assertIsNone(m2_2.active_program)
        self.assertIn("**Academic Subjects**", r2_2.reply)

        # 3. Fresh: subjects -> what subjects do you offer? => global -> global
        sid3 = f"reg_03_{uuid.uuid4()}"
        _turn(sid3, "subjects")
        r3_2, m3_2 = _turn(sid3, "what subjects do you offer?")
        self.assertEqual(r3_2.intent, "faq")
        self.assertIsNone(m3_2.active_program)
        self.assertIn("**Academic Subjects**", r3_2.reply)

        # 4. Fresh: subjects -> tell me about the subjects => global -> global
        sid4 = f"reg_04_{uuid.uuid4()}"
        _turn(sid4, "subjects")
        r4_2, m4_2 = _turn(sid4, "tell me about the subjects")
        self.assertEqual(r4_2.intent, "faq")
        self.assertIsNone(m4_2.active_program)
        self.assertIn("**Academic Subjects**", r4_2.reply)

        # 5. Fresh: subjects -> programs -> subjects => global -> global
        sid5 = f"reg_05_{uuid.uuid4()}"
        _turn(sid5, "subjects")
        r5_prog, m5_prog = _turn(sid5, "programs")
        self.assertIsNone(m5_prog.active_program)
        r5_3, m5_3 = _turn(sid5, "subjects")
        self.assertEqual(r5_3.intent, "faq")
        self.assertIsNone(m5_3.active_program)
        self.assertIn("**Academic Subjects**", r5_3.reply)

        # 6. Fresh: programs -> subjects => global
        sid6 = f"reg_06_{uuid.uuid4()}"
        _turn(sid6, "programs")
        r6_2, m6_2 = _turn(sid6, "subjects")
        self.assertEqual(r6_2.intent, "faq")
        self.assertIsNone(m6_2.active_program)
        self.assertIn("**Academic Subjects**", r6_2.reply)

        # 7. Fresh: programs -> which subjects => global
        sid7 = f"reg_07_{uuid.uuid4()}"
        _turn(sid7, "programs")
        r7_2, m7_2 = _turn(sid7, "which subjects")
        self.assertEqual(r7_2.intent, "faq")
        self.assertIsNone(m7_2.active_program)
        self.assertIn("**Academic Subjects**", r7_2.reply)

        # 8. Explicit context: Confident Speaker -> subjects -> subjects => CS -> CS -> CS
        sid8 = f"reg_08_{uuid.uuid4()}"
        _turn(sid8, "Confident Speaker")
        r8_2, m8_2 = _turn(sid8, "subjects")
        self.assertEqual(r8_2.intent, "confident_speaker_scope")
        self.assertEqual(m8_2.active_program, "confident_speaker")
        self.assertIn("The program covers four curriculum areas:", r8_2.reply)
        r8_3, m8_3 = _turn(sid8, "subjects")
        self.assertEqual(r8_3.intent, "confident_speaker_scope")
        self.assertEqual(m8_3.active_program, "confident_speaker")

        # 9. Explicit context: Middle School -> subjects -> subjects => MS -> MS -> MS
        sid9 = f"reg_09_{uuid.uuid4()}"
        _turn(sid9, "Middle School")
        r9_2, m9_2 = _turn(sid9, "subjects")
        self.assertEqual(r9_2.intent, "middle_school_subjects")
        self.assertEqual(m9_2.active_program, "middle")
        self.assertIn("Middle School", r9_2.reply)
        r9_3, m9_3 = _turn(sid9, "subjects")
        self.assertEqual(r9_3.intent, "middle_school_subjects")
        self.assertEqual(m9_3.active_program, "middle")

        # 10. Explicit context: Senior School -> subjects -> subjects => SS -> SS -> SS
        sid10 = f"reg_10_{uuid.uuid4()}"
        _turn(sid10, "Senior School")
        r10_2, m10_2 = _turn(sid10, "subjects")
        self.assertEqual(r10_2.intent, "senior_school_subjects")
        self.assertEqual(m10_2.active_program, "senior")
        self.assertIn("Senior School Focus", r10_2.reply)
        r10_3, m10_3 = _turn(sid10, "subjects")
        self.assertEqual(r10_3.intent, "senior_school_subjects")
        self.assertEqual(m10_3.active_program, "senior")

        # 11. Explicit context: Foundation Years -> subjects -> subjects => FY -> FY -> FY
        sid11 = f"reg_11_{uuid.uuid4()}"
        _turn(sid11, "Foundation Years")
        r11_2, m11_2 = _turn(sid11, "subjects")
        self.assertEqual(r11_2.intent, "foundation_subjects")
        self.assertEqual(m11_2.active_program, "foundation")
        self.assertIn("Foundation Years", r11_2.reply)
        r11_3, m11_3 = _turn(sid11, "subjects")
        self.assertEqual(r11_3.intent, "foundation_subjects")
        self.assertEqual(m11_3.active_program, "foundation")

        # 12. Assistant-content contamination:
        # User: subjects -> Assistant: [global response containing Confident Speaker] -> User: subjects => GLOBAL SUBJECTS
        sid12 = f"reg_12_{uuid.uuid4()}"
        r12_1, _ = _turn(sid12, "subjects")
        self.assertIn("Confident Speaker", r12_1.reply)
        r12_2, m12_2 = _turn(sid12, "subjects")
        self.assertEqual(r12_2.intent, "faq")
        self.assertIsNone(m12_2.active_program)
        self.assertIn("**Academic Subjects**", r12_2.reply)

        # 13. Program-list contamination:
        # User: programs -> Assistant: [all four programs] -> User: subjects => GLOBAL SUBJECTS
        sid13 = f"reg_13_{uuid.uuid4()}"
        r13_1, _ = _turn(sid13, "programs")
        self.assertIn("Foundation Years", r13_1.reply)
        self.assertIn("Middle School", r13_1.reply)
        self.assertIn("Senior School", r13_1.reply)
        self.assertIn("Confident Speaker", r13_1.reply)
        r13_2, m13_2 = _turn(sid13, "subjects")
        self.assertEqual(r13_2.intent, "faq")
        self.assertIsNone(m13_2.active_program)
        self.assertIn("**Academic Subjects**", r13_2.reply)

        # 14. Retrieval contamination:
        # User: subjects -> active_program MUST remain None regardless of retrieved entries
        sid14 = f"reg_14_{uuid.uuid4()}"
        r14_1, m14_1 = _turn(sid14, "subjects")
        self.assertIsNone(m14_1.active_program)

        # 15. Cross-session:
        # Session A: Confident Speaker -> subjects (CS subjects)
        # Session B: subjects => GLOBAL SUBJECTS
        sid_a = f"reg_15a_{uuid.uuid4()}"
        sid_b = f"reg_15b_{uuid.uuid4()}"
        _turn(sid_a, "Confident Speaker")
        ra_sub, ma_sub = _turn(sid_a, "subjects")
        self.assertEqual(ra_sub.intent, "confident_speaker_scope")
        self.assertEqual(ma_sub.active_program, "confident_speaker")

        rb_sub, mb_sub = _turn(sid_b, "subjects")
        self.assertEqual(rb_sub.intent, "faq")
        self.assertIsNone(mb_sub.active_program)
        self.assertIn("**Academic Subjects**", rb_sub.reply)
        self.assertIn("**Confident Speaker**", rb_sub.reply)

    # =========================================================================
    # 18. DELIVERY MODE AND CLASS SIZE CERTIFICATION
    # =========================================================================
    def test_18_delivery_mode_and_class_size_certification(self):
        """Certify distinct routing for delivery mode, class frequency, and class size."""
        def _turn(sid, q):
            res = self.engine.handle_message(sid, q)
            mem = database.get_conversation_memory(sid)
            return res, mem

        # 1. Standalone online/delivery queries
        online_queries = [
            "are they online",
            "Are classes online?",
            "Do you teach online?",
            "Are the classes online?",
            "How are classes conducted?",
            "Where are classes conducted?",
            "Is it online?",
            "Are classes virtual?",
            "Do you use Google Meet?",
        ]
        for q in online_queries:
            with self.subTest(query=q):
                sid = f"sub_online_{uuid.uuid4()}"
                r, _ = _turn(sid, q)
                self.assertEqual(r.intent, "online_classes")
                self.assertIn("online", r.reply.lower())
                self.assertNotIn("5 classes per week", r.reply.lower())
                self.assertNotIn("3–5 classes per week", r.reply.lower())

        # 2. Standalone frequency queries
        freq_queries = [
            "classes",
            "How many classes per week?",
            "How often are classes?",
            "How many classes are there?",
            "How many classes are there per week?",
        ]
        for q in freq_queries:
            with self.subTest(query=q):
                sid = f"sub_freq_{uuid.uuid4()}"
                r, _ = _turn(sid, q)
                self.assertIn("class_frequency", r.intent)
                self.assertIn("5 classes per week", r.reply)

        # 3. Standalone class size / batch size queries
        size_queries = [
            "how many students in class",
            "How many students are in a class?",
            "How many students per class?",
            "How many kids are in one class?",
            "batch size",
            "What is the batch size?",
            "How big are the classes?",
            "How many students are in a batch?",
            "Are classes one-on-one?",
            "Is it 1:1 or group?",
        ]
        for q in size_queries:
            with self.subTest(query=q):
                sid = f"sub_size_{uuid.uuid4()}"
                r, _ = _turn(sid, q)
                self.assertEqual(r.intent, "class_size")
                self.assertIn("1:1", r.reply)
                self.assertIn("8 students", r.reply)
                self.assertNotIn("5 classes per week", r.reply)

        # 4. Multi-part queries
        sid_multi1 = f"sub_m1_{uuid.uuid4()}"
        r_m1, _ = _turn(sid_multi1, "Are they online and how many classes per week?")
        self.assertEqual(r_m1.intent, "online_and_frequency")
        self.assertIn("online", r_m1.reply.lower())
        self.assertIn("5 classes per week", r_m1.reply)

        sid_multi2 = f"sub_m2_{uuid.uuid4()}"
        r_m2, _ = _turn(sid_multi2, "How many students are in a class and how many classes per week?")
        self.assertEqual(r_m2.intent, "class_size_and_frequency")
        self.assertIn("8 students", r_m2.reply)
        self.assertIn("5 classes per week", r_m2.reply)

        # 5. Multi-turn: frequency -> online (MUST NOT inherit frequency)
        sid_seq1 = f"sub_seq1_{uuid.uuid4()}"
        r_seq1_1, _ = _turn(sid_seq1, "How many classes per week?")
        self.assertIn("class_frequency", r_seq1_1.intent)
        r_seq1_2, _ = _turn(sid_seq1, "Are they online?")
        self.assertEqual(r_seq1_2.intent, "online_classes")
        self.assertNotIn("5 classes per week", r_seq1_2.reply)

        sid_seq2 = f"sub_seq2_{uuid.uuid4()}"
        r_seq2_1, _ = _turn(sid_seq2, "classes")
        self.assertIn("class_frequency", r_seq2_1.intent)
        r_seq2_2, _ = _turn(sid_seq2, "are they online")
        self.assertEqual(r_seq2_2.intent, "online_classes")
        self.assertNotIn("5 classes per week", r_seq2_2.reply)

        # 6. Multi-turn: program context preservation with online delivery
        prog_online_pairs = [
            ("Foundation Years", "foundation"),
            ("Middle School", "middle"),
            ("Senior School", "senior"),
            ("Confident Speaker", "confident_speaker"),
        ]
        for prog_input, expected_prog in prog_online_pairs:
            with self.subTest(program=prog_input):
                sid_prog = f"sub_prog_{uuid.uuid4()}"
                _turn(sid_prog, prog_input)
                r_on, m_on = _turn(sid_prog, "Are they online?")
                self.assertEqual(r_on.intent, "online_classes")
                self.assertEqual(m_on.active_program, expected_prog)
                self.assertIn("online", r_on.reply.lower())
                self.assertNotIn("5 classes per week", r_on.reply)

        # 7. Multi-turn: frequency -> class size, online -> batch size, class size -> frequency
        sid_seq3 = f"sub_seq3_{uuid.uuid4()}"
        _turn(sid_seq3, "How many classes per week?")
        r_seq3_2, _ = _turn(sid_seq3, "How many students are in each class?")
        self.assertEqual(r_seq3_2.intent, "class_size")

        sid_seq4 = f"sub_seq4_{uuid.uuid4()}"
        _turn(sid_seq4, "Are they online?")
        r_seq4_2, _ = _turn(sid_seq4, "What's the batch size?")
        self.assertEqual(r_seq4_2.intent, "class_size")

        sid_seq5 = f"sub_seq5_{uuid.uuid4()}"
        _turn(sid_seq5, "How many students are in a class?")
        r_seq5_2, _ = _turn(sid_seq5, "How many classes per week?")
        self.assertIn("class_frequency", r_seq5_2.intent)

        # 8. Program context preservation with class size
        prog_size_pairs = [
            ("Foundation Years", "how many students in class", "foundation"),
            ("Middle School", "batch size", "middle"),
            ("Senior School", "are classes 1:1?", "senior"),
            ("Confident Speaker", "how many students in a class?", "confident_speaker"),
        ]
        for prog_input, size_query, expected_prog in prog_size_pairs:
            with self.subTest(program=prog_input, query=size_query):
                sid_ps = f"sub_ps_{uuid.uuid4()}"
                _turn(sid_ps, prog_input)
                r_sz, m_sz = _turn(sid_ps, size_query)
                self.assertEqual(r_sz.intent, "class_size")
                self.assertEqual(m_sz.active_program, expected_prog)
                self.assertIn("8 students", r_sz.reply)
                self.assertNotIn("5 classes per week", r_sz.reply)


if __name__ == "__main__":
    unittest.main()

