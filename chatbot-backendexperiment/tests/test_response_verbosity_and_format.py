"""
Regression and verification test suite for WeMentors chatbot response length,
adaptive verbosity, and question-scoped formatting behavior.

Verifies:
1. Simple factual queries receive concise, direct, question-scoped responses.
2. Unsolicited 'Book Free Demo' CTAs are NOT appended to simple factual questions.
3. Relevant factual information (subjects, grades, curricula) is preserved.
4. Follow-up queries resolve correctly using conversational memory.
5. Comparisons remain clean and structured without duplicated CTAs.
6. Detailed/complex queries remain sufficiently comprehensive (no over-compression).
"""

import sys
import unittest
from pathlib import Path

# Add backend directory to sys.path
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = TESTS_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.conversation import ConversationEngine
from app.knowledge import load_entries
from app import personality, database


class TestResponseVerbosityAndFormatting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_db()
        cls.entries = load_entries()
        cls.engine = ConversationEngine(cls.entries)

    def test_01_which_subjects_concise(self):
        """1. 'Which subjects?' must be concise and question-scoped with no demo CTA."""
        res = self.engine.handle_message("sess_verb_1", "Which subjects?")
        reply = res.reply
        # Preserves core academic subjects
        self.assertIn("Mathematics", reply)
        self.assertIn("Science", reply)
        self.assertIn("English", reply)
        # Preserves communication tracks
        self.assertIn("Public Speaking", reply)
        self.assertIn("Business English", reply)
        self.assertIn("General Communicative Skills", reply)
        self.assertIn("IELTS Preparation", reply)
        # Does NOT include grade-by-grade breakdowns or unrequested details
        self.assertNotIn("study habits", reply.lower())
        self.assertNotIn("doubt clinics", reply.lower())
        self.assertNotIn("syllabus mastery", reply.lower())
        # Does NOT include unsolicited demo CTA
        self.assertNotIn("book free demo", reply.lower())

    def test_02_what_grades_supported_concise(self):
        """2. 'What grades do you support?' must be concise and question-scoped with no demo CTA."""
        res = self.engine.handle_message("sess_verb_2", "What grades do you support?")
        reply = res.reply
        self.assertIn("3", reply)
        self.assertIn("10", reply)
        self.assertIn("Foundation Years", reply)
        self.assertIn("Middle School", reply)
        self.assertIn("Senior School Focus", reply)
        # Does NOT include subjects or unrequested demo CTA
        self.assertNotIn("mathematics", reply.lower())
        self.assertNotIn("book free demo", reply.lower())
        self.assertIn("which-grades-supported", res.matched_entry_ids)

    def test_03_what_programs_do_you_offer(self):
        """3. 'What programs do you offer?' lists the 4 programs clearly."""
        res = self.engine.handle_message("sess_verb_3", "What programs do you offer?")
        reply = res.reply
        self.assertIn("Foundation Years", reply)
        self.assertIn("Middle School", reply)
        self.assertIn("Senior School Focus", reply)
        self.assertIn("Confident Speaker", reply)

    def test_04_do_you_teach_mathematics(self):
        """4. 'Do you teach mathematics?' is direct, concise, and has no demo CTA."""
        res = self.engine.handle_message("sess_verb_4", "Do you teach mathematics?")
        reply = res.reply
        self.assertTrue(reply.lower().startswith("yes"))
        self.assertIn("Mathematics", reply)
        self.assertNotIn("book free demo", reply.lower())
        # Concise answer
        self.assertLessEqual(len(reply.split("\n")), 3)

    def test_05_tell_me_about_foundation_years(self):
        """5. 'Tell me about Foundation Years' gives structured details with demo CTA."""
        res = self.engine.handle_message("sess_verb_5", "Tell me about Foundation Years.")
        reply = res.reply
        self.assertIn("Foundation Years", reply)
        self.assertTrue("3" in reply and "5" in reply)
        self.assertIn("book free demo", reply.lower())

    def test_06_what_is_confident_speaker(self):
        """6. 'What is Confident Speaker?' gives structured program overview."""
        res = self.engine.handle_message("sess_verb_6", "What is Confident Speaker?")
        reply = res.reply
        self.assertIn("Confident Speaker", reply)
        self.assertIn("Public Speaking", reply)
        self.assertIn("Business English", reply)
        self.assertIn("book free demo", reply.lower())

    def test_07_how_do_i_book_a_demo(self):
        """7. 'How do I book a demo?' gives clear booking guidance."""
        res = self.engine.handle_message("sess_verb_7", "How do I book a demo?")
        reply = res.reply
        self.assertIn("book free demo", reply.lower())

    def test_08_compare_programs_no_duplicate_cta(self):
        """8. 'Compare Foundation Years and Middle School' produces clean comparison without duplicate CTAs."""
        res = self.engine.handle_message("sess_verb_8", "Compare Foundation Years and Middle School.")
        reply = res.reply
        self.assertEqual(res.intent, "comparison")
        self.assertIn("Foundation Years", reply)
        self.assertIn("Middle School", reply)
        # Should NOT duplicate the Book Free Demo CTA in the middle
        self.assertNotIn("Ready to explore the program? You can book a free 30-minute demo using the **Book Free Demo** button.", reply)

    def test_09_follow_up_tell_me_more_about_first_one(self):
        """9. 'Tell me more about the first one' uses conversational context correctly."""
        sid = "sess_verb_9"
        self.engine.handle_message(sid, "What programs do you offer?")
        res2 = self.engine.handle_message(sid, "Tell me more about the first one.")
        reply = res2.reply
        self.assertIn("Foundation Years", reply)
        self.assertTrue("3" in reply and "5" in reply)

    def test_10_what_about_english_focused(self):
        """10. 'What about English?' directly answers about English without boilerplate."""
        res = self.engine.handle_message("sess_verb_10", "What about English?")
        reply = res.reply
        self.assertIn("English", reply)
        self.assertIn("Foundation Years", reply)
        self.assertIn("Middle School", reply)
        self.assertIn("Confident Speaker", reply)
        # Must NOT include verbose injected boilerplate
        self.assertNotIn("Personalized mentoring with a personal mentor is provided.", reply)
        self.assertNotIn("Students receive individual attention and dedicated mentor feedback.", reply)
        self.assertNotIn("book free demo", reply.lower())

    def test_11_complex_query_not_overcompressed(self):
        """11. Complex multi-part query preserves detailed structured answer."""
        complex_q = (
            "Tell me everything about the Senior School Focus program, "
            "including subjects, teaching approach, classes, and how mentoring works."
        )
        res = self.engine.handle_message("sess_verb_11", complex_q)
        reply = res.reply
        self.assertIn("Senior School Focus", reply)
        self.assertTrue("9" in reply and "10" in reply)
        self.assertIn("Mathematics", reply)
        self.assertIn("Science", reply)
        self.assertIn("Key features", reply)
        # Complex program inquiry legitimately includes demo CTA
        self.assertIn("book free demo", reply.lower())

    def test_12_simple_factual_questions_no_demo_cta(self):
        """12. Simple factual questions do not receive unsolicited demo CTAs."""
        simple_queries = [
            ("How long are classes?", "50 minutes"),
            ("Are classes one-on-one?", "one-on-one"),
            ("Do you offer CBSE?", "CBSE"),
            ("Do you offer science?", "Science"),
        ]
        for q, expected_snippet in simple_queries:
            with self.subTest(query=q):
                res = self.engine.handle_message(f"sess_fact_{hash(q)}", q)
                reply = res.reply
                self.assertIn(expected_snippet.lower(), reply.lower())
                self.assertNotIn("book free demo", reply.lower())

    # =========================================================================
    # SECTION 14 REGRESSION TESTS (Global response-length, subject-query & context)
    # =========================================================================

    def test_13_cs_standalone_full_overview(self):
        """1 & 2. Standalone Confident Speaker query returns full verified overview."""
        for q in ["Confident Speaker", "What is Confident Speaker?"]:
            with self.subTest(query=q):
                res = self.engine.handle_message(f"sess_cs_stand_{hash(q)}", q)
                reply = res.reply
                self.assertIn("**Confident Speaker**", reply)
                self.assertIn("Curriculum:", reply)
                self.assertIn("Public Speaking", reply)
                self.assertIn("Business English", reply)
                self.assertIn("General Communicative Skills", reply)
                self.assertIn("IELTS Preparation", reply)
                self.assertIn("Key features:", reply)
                self.assertIn("book free demo", reply.lower())

    def test_14_cs_contextual_short_curriculum(self):
        """3, 4, 5. Active Confident Speaker context with short subject/skill inquiry returns concise response."""
        follow_ups = [
            ("Which subjects?", "3"),
            ("What subjects are in this?", "4"),
            ("What skills are covered?", "5"),
        ]
        for q, case_id in follow_ups:
            with self.subTest(query=q, case=case_id):
                sid = f"sess_cs_ctx_{case_id}"
                self.engine.handle_message(sid, "Tell me about Confident Speaker.")
                res = self.engine.handle_message(sid, q)
                reply = res.reply
                self.assertEqual(res.intent, "confident_speaker_scope")
                # Preferred concise response format
                self.assertIn("The program covers four curriculum areas:", reply)
                self.assertIn("- **Public Speaking** — build confidence and speaking skills for school and college.", reply)
                self.assertIn("- **Business English** — improve professional and workplace communication.", reply)
                self.assertIn("- **General Communicative Skills** — develop practical English for everyday conversations.", reply)
                self.assertIn("- **IELTS Preparation** — prepare for IELTS with guided practice and mentoring.", reply)
                # Must NOT repeat full program card or demo CTA
                self.assertNotIn("**Confident Speaker**\n\nPractical", reply)
                self.assertNotIn("Key features:", reply)
                self.assertNotIn("Grades:", reply)
                self.assertNotIn("book free demo", reply.lower())
                self.assertNotIn("rote grammar drills", reply.lower())

    def test_15_cs_explicit_short_subject_query(self):
        """6. 'What subjects are in Confident Speaker?' returns concise response."""
        res = self.engine.handle_message("sess_cs_sub_explicit", "What subjects are in Confident Speaker?")
        reply = res.reply
        self.assertEqual(res.intent, "confident_speaker_scope")
        self.assertIn("The program covers four curriculum areas:", reply)
        self.assertIn("- **Public Speaking**", reply)
        self.assertIn("- **Business English**", reply)
        self.assertIn("- **General Communicative Skills**", reply)
        self.assertIn("- **IELTS Preparation**", reply)
        self.assertNotIn("book free demo", reply.lower())
        self.assertNotIn("rote grammar drills", reply.lower())

    def test_16_middle_school_explicit_and_typo(self):
        """7 & 8. Middle School explicit subject queries and typo normalization."""
        queries = [
            "What subjects in Middle School?",
            "What subjects are in Middle School?",
            "Which subjects does Middle School have?",
            "Middle School subjects?",
            "What subjects in middle shcool?",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.engine.handle_message(f"sess_ms_exp_{hash(q)}", q)
                reply = res.reply
                self.assertEqual(res.intent, "middle_school_subjects")
                self.assertIn("**Middle School (Grades 6–8)**", reply)
                self.assertIn("- **Mathematics**", reply)
                self.assertIn("- **Science**", reply)
                self.assertIn("- **English**", reply)
                self.assertIn("- **Social Studies**", reply)
                self.assertIn("doubt-solving clinics and practical labs", reply.lower())
                self.assertNotIn("book free demo", reply.lower())

    def test_17_middle_school_contextual(self):
        """9. 'Middle School' -> 'Which subjects?' resolves to Middle School subjects."""
        sid = "sess_ms_contextual"
        self.engine.handle_message(sid, "Tell me about Middle School.")
        res = self.engine.handle_message(sid, "Which subjects?")
        reply = res.reply
        self.assertEqual(res.intent, "middle_school_subjects")
        self.assertIn("**Middle School (Grades 6–8)**", reply)
        self.assertIn("- **Mathematics**", reply)
        self.assertIn("- **Science**", reply)
        self.assertIn("- **English**", reply)
        self.assertIn("- **Social Studies**", reply)
        self.assertNotIn("book free demo", reply.lower())

    def test_18_foundation_years_contextual_and_explicit(self):
        """10. Foundation Years explicit and contextual subject inquiries."""
        # Contextual
        sid = "sess_fy_contextual"
        self.engine.handle_message(sid, "Tell me about Foundation Years.")
        res = self.engine.handle_message(sid, "Which subjects?")
        reply = res.reply
        self.assertEqual(res.intent, "foundation_subjects")
        self.assertIn("**Foundation Years (Grades 3–5)**", reply)
        self.assertIn("- **Mathematics**", reply)
        self.assertIn("- **Science**", reply)
        self.assertIn("- **English**", reply)
        self.assertIn("- **Environmental Studies (EVS)**", reply)
        self.assertNotIn("book free demo", reply.lower())

        # Explicit
        res_exp = self.engine.handle_message("sess_fy_exp_1", "Which subjects in Foundation?")
        self.assertEqual(res_exp.intent, "foundation_subjects")
        self.assertIn("**Foundation Years (Grades 3–5)**", res_exp.reply)

        res_cov = self.engine.handle_message("sess_fy_exp_2", "What does Foundation Years cover?")
        self.assertEqual(res_cov.intent, "foundation_subjects")
        self.assertIn("**Foundation Years (Grades 3–5)**", res_cov.reply)

    def test_19_senior_school_contextual_and_explicit(self):
        """11. Senior School Focus explicit and contextual subject inquiries."""
        # Contextual
        sid = "sess_ss_contextual"
        self.engine.handle_message(sid, "Tell me about Senior School Focus.")
        res = self.engine.handle_message(sid, "Which subjects?")
        reply = res.reply
        self.assertEqual(res.intent, "senior_school_subjects")
        self.assertIn("**Senior School Focus (Grades 9–10)**", reply)
        self.assertIn("- **Mathematics**", reply)
        self.assertIn("- **Science**", reply)
        self.assertIn("- **Board Exam Preparation**", reply)
        self.assertNotIn("book free demo", reply.lower())

        # Explicit
        res_exp = self.engine.handle_message("sess_ss_exp_1", "Which subjects in Senior School?")
        self.assertEqual(res_exp.intent, "senior_school_subjects")
        self.assertIn("**Senior School Focus (Grades 9–10)**", res_exp.reply)

        res_cov = self.engine.handle_message("sess_ss_exp_2", "What does Senior School Focus cover?")
        self.assertEqual(res_cov.intent, "senior_school_subjects")
        self.assertIn("**Senior School Focus (Grades 9–10)**", res_cov.reply)

    def test_20_global_which_subjects_query(self):
        """12. Fresh conversation 'Which subjects?' returns global structured overview."""
        global_queries = [
            "Which subjects?",
            "What subjects do you offer?",
            "What subjects are available?",
            "Which subjects do you teach?",
        ]
        for q in global_queries:
            with self.subTest(query=q):
                res = self.engine.handle_message(f"sess_glob_subj_{hash(q)}", q)
                reply = res.reply
                self.assertEqual(res.intent, "faq")
                self.assertIn("**Academic Subjects**", reply)
                self.assertIn("- **Mathematics**", reply)
                self.assertIn("- **Science**", reply)
                self.assertIn("- **English**", reply)
                self.assertIn("- **Any additional subjects included in the supported academic curriculum**", reply)
                self.assertIn("**Confident Speaker**", reply)
                self.assertIn("- **Public Speaking**", reply)
                self.assertIn("- **Business English**", reply)
                self.assertIn("- **General Communicative Skills**", reply)
                self.assertIn("- **IELTS Preparation**", reply)
                self.assertNotIn("book free demo", reply.lower())

    def test_21_context_isolation(self):
        """13, 14, 15. Context isolation: no cross-program subject leakage."""
        # 13. Confident Speaker active -> Which subjects? (no academic leakage)
        sid_cs = "sess_iso_cs"
        self.engine.handle_message(sid_cs, "Tell me about Confident Speaker.")
        res_cs = self.engine.handle_message(sid_cs, "Which subjects?")
        self.assertIn("Public Speaking", res_cs.reply)
        self.assertNotIn("Mathematics", res_cs.reply)
        self.assertNotIn("Science", res_cs.reply)
        self.assertNotIn("Social Studies", res_cs.reply)

        # 14. Middle School active -> Which subjects? (no Confident Speaker leakage)
        sid_ms = "sess_iso_ms"
        self.engine.handle_message(sid_ms, "Tell me about Middle School.")
        res_ms = self.engine.handle_message(sid_ms, "Which subjects?")
        self.assertIn("Social Studies", res_ms.reply)
        self.assertNotIn("Public Speaking", res_ms.reply)
        self.assertNotIn("Business English", res_ms.reply)
        self.assertNotIn("IELTS", res_ms.reply)

        # 15. Foundation Years active -> Which subjects? (no Middle/Senior leakage)
        sid_fy = "sess_iso_fy"
        self.engine.handle_message(sid_fy, "Tell me about Foundation Years.")
        res_fy = self.engine.handle_message(sid_fy, "Which subjects?")
        self.assertIn("Environmental Studies", res_fy.reply)
        self.assertNotIn("Social Studies", res_fy.reply)
        self.assertNotIn("Board Exam", res_fy.reply)
        self.assertNotIn("Public Speaking", res_fy.reply)

    def test_22_explicit_beats_conversation_context(self):
        """16 & 17. Explicit program in current message overrides previous context."""
        # 16. Context = Confident Speaker; message = "What subjects in Middle School?"
        sid_cs_to_ms = "sess_ctx_override_1"
        self.engine.handle_message(sid_cs_to_ms, "Tell me about Confident Speaker.")
        res1 = self.engine.handle_message(sid_cs_to_ms, "What subjects in Middle School?")
        self.assertEqual(res1.intent, "middle_school_subjects")
        self.assertIn("Middle School", res1.reply)
        self.assertIn("Social Studies", res1.reply)
        self.assertNotIn("Public Speaking", res1.reply)

        # 17. Context = Middle School; message = "What subjects in Confident Speaker?"
        sid_ms_to_cs = "sess_ctx_override_2"
        self.engine.handle_message(sid_ms_to_cs, "Tell me about Middle School.")
        res2 = self.engine.handle_message(sid_ms_to_cs, "What subjects in Confident Speaker?")
        self.assertEqual(res2.intent, "confident_speaker_scope")
        self.assertIn("Public Speaking", res2.reply)
        self.assertNotIn("Social Studies", res2.reply)

    def test_23_response_length_and_format_distinction(self):
        """18, 19, 20. Length and formatting distinction across query types."""
        # 18. Subject-only questions must NOT return full program overviews
        res_sub_only = self.engine.handle_message("sess_dist_1", "What subjects are in Middle School?")
        self.assertNotIn("**Middle School — All Subjects**", res_sub_only.reply)
        self.assertNotIn("Ready to explore the program?", res_sub_only.reply)

        # 19. Program overview questions SHOULD return the fuller verified overview
        res_overview = self.engine.handle_message("sess_dist_2", "Tell me about Middle School.")
        self.assertIn("**Middle School — All Subjects**", res_overview.reply)
        self.assertIn("Key features:", res_overview.reply)
        self.assertIn("Ready to explore the program?", res_overview.reply)

        # 20. Complex multi-part questions receive detailed structured answers
        complex_q = (
            "Tell me everything about Middle School including subjects, teaching approach, classes, and mentoring."
        )
        res_complex = self.engine.handle_message("sess_dist_3", complex_q)
        self.assertIn("Middle School", res_complex.reply)
        self.assertIn("core subjects", res_complex.reply.lower())
        self.assertIn("book free demo", res_complex.reply.lower())


    def test_24_bare_subjects_always_global(self):
        """Bare 'subjects' must always return the global subject overview,
        regardless of which program was previously discussed."""
        bare_queries = ["subjects", "subjects?",
                        "what are the subjects", "what are the subjects?"]

        # 1. Fresh session
        for q in bare_queries:
            with self.subTest(query=q, context="fresh"):
                res = self.engine.handle_message(f"sess_bare_fresh_{hash(q)}", q)
                self.assertEqual(res.intent, "faq")
                self.assertIn("**Academic Subjects**", res.reply)
                self.assertIn("**Confident Speaker**", res.reply)
                self.assertIn("- **Public Speaking**", res.reply)
                self.assertIn("- **IELTS Preparation**", res.reply)

        # 2-5. After each program context
        programs = [
            ("Senior School Focus", "senior_school_overview"),
            ("Tell me about Middle School.", "middle_school_overview"),
            ("Tell me about Confident Speaker.", "confident_speaker"),
            ("Tell me about Foundation Years.", "foundation_years_overview"),
        ]
        for setup_msg, _expected_setup_intent in programs:
            with self.subTest(query="subjects", context=setup_msg):
                sid = f"sess_bare_{hash(setup_msg)}_subjects"
                self.engine.handle_message(sid, setup_msg)
                res = self.engine.handle_message(sid, "subjects")
                self.assertEqual(res.intent, "faq")
                self.assertIn("**Academic Subjects**", res.reply)
                self.assertIn("**Confident Speaker**", res.reply)
                self.assertNotIn("Grades 6", res.reply)


        # Exact Senior School -> "subjects" regression (the original bug)
        sid_bug = "sess_senior_subjects_bug"
        self.engine.handle_message(sid_bug, "Senior School Focus")
        res_bug = self.engine.handle_message(sid_bug, "subjects")
        self.assertNotIn("Middle School", res_bug.reply)
        self.assertNotIn("Grades 6–8", res_bug.reply)
        self.assertIn("**Academic Subjects**", res_bug.reply)
        self.assertIn("**Confident Speaker**", res_bug.reply)
        self.assertIn("- **Public Speaking**", res_bug.reply)
        self.assertIn("- **IELTS Preparation**", res_bug.reply)

    def test_25_explicit_subjects_in_program(self):
        """'subjects in <program>' must still route to program-specific responses."""
        # subjects in Middle School
        res_ms = self.engine.handle_message("sess_expl_ms", "subjects in Middle School")
        self.assertEqual(res_ms.intent, "middle_school_subjects")
        self.assertIn("Middle School", res_ms.reply)
        self.assertIn("Social Studies", res_ms.reply)

        # subjects in Senior School
        res_ss = self.engine.handle_message("sess_expl_ss", "subjects in Senior School")
        self.assertEqual(res_ss.intent, "senior_school_subjects")
        self.assertIn("Senior School Focus", res_ss.reply)

        # subjects in Confident Speaker
        res_cs = self.engine.handle_message("sess_expl_cs", "subjects in Confident Speaker")
        self.assertEqual(res_cs.intent, "confident_speaker_scope")
        self.assertIn("Public Speaking", res_cs.reply)


if __name__ == "__main__":
    unittest.main()
