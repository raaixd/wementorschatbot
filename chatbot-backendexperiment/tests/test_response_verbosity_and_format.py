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


if __name__ == "__main__":
    unittest.main()
