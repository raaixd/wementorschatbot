"""
Test Suite: Grade 1 & 2 Availability and International Eligibility
Validates that:
1. Queries about first grade or second grade (Grade 1/2, Class 1/2, 1st/2nd grade)
   are clearly informed that WeMentors does not offer courses for them yet, and explains
   that academic programs start from Grade 3.
2. Queries asking if a learner can join from Saudi Arabia or any other country
   are affirmatively answered that yes, anyone globally can join from any country.
3. Multi-turn context correctly prioritizes current explicit referents.
4. "First", "Second", "Saudi", "Arabia" are never captured as user names.
5. DEMO_TRANSACTION_ENABLED remains False.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.conversation import ConversationEngine
from app.leads import DEMO_TRANSACTION_ENABLED
from app.knowledge import load_entries


@pytest.fixture(scope="module")
def engine():
    entries = load_entries()
    return ConversationEngine(entries)


class TestGrade1And2Availability:
    """Test suite for Grade 1 and Grade 2 unavailability responses."""

    def test_01_first_grade_course_inquiry(self, engine):
        res = engine.handle_message("sess_g1_01", "do you have courses for first grade")
        assert res.intent == "grade_1_2_unavailable"
        assert "not offer courses for first grade or second grade yet" in res.reply.lower()
        assert "grade 3" in res.reply.lower()

    def test_02_second_grade_standalone(self, engine):
        res = engine.handle_message("sess_g2_01", "second grade")
        assert res.intent == "grade_1_2_unavailable"
        assert "not offer courses for first grade or second grade yet" in res.reply.lower()

    def test_03_first_grade_standalone(self, engine):
        res = engine.handle_message("sess_g1_02", "first grade")
        assert res.intent == "grade_1_2_unavailable"

    def test_04_grade_1_and_grade_2(self, engine):
        for q in ["grade 1", "grade 2", "class 1", "class 2", "1st grade", "2nd grade"]:
            res = engine.handle_message(f"sess_{q}", q)
            assert res.intent == "grade_1_2_unavailable", f"Failed for query: {q}"
            assert "not offer courses for first grade or second grade yet" in res.reply.lower()

    def test_05_child_in_1st_grade(self, engine):
        res = engine.handle_message("sess_g1_03", "can my child in 1st grade join")
        assert res.intent == "grade_1_2_unavailable"
        assert "not offer courses for first grade or second grade yet" in res.reply.lower()

    def test_06_context_switch_to_second_grade(self, engine):
        session_id = "sess_ctx_g2"
        res1 = engine.handle_message(session_id, "Tell me about Middle School")
        assert res1.intent == "middle_school_overview"

        res2 = engine.handle_message(session_id, "What about second grade?")
        assert res2.intent == "grade_1_2_unavailable"
        assert "not offer courses for first grade or second grade yet" in res2.reply.lower()

    def test_07_grade_10_and_grade_3_not_affected(self, engine):
        # Must not falsely trigger for Grade 10 or Grade 3
        res_10 = engine.handle_message("sess_neg_10", "What support do you provide for Grade 10?")
        assert res_10.intent == "board_exam"
        assert "grade_1_2_unavailable" != res_10.intent

        res_3 = engine.handle_message("sess_neg_3", "What is Foundation Years for Grade 3?")
        assert "grade_1_2_unavailable" != res_3.intent


class TestInternationalEligibility:
    """Test suite for Saudi Arabia and global / international joining inquiries."""

    def test_01_join_from_saudi_arabia(self, engine):
        res = engine.handle_message("sess_intl_01", "can i join from saudi arabia")
        assert res.intent == "international_eligibility"
        assert "anyone globally can join from any country" in res.reply.lower()
        assert "saudi arabia" in res.reply.lower()
        assert "online" in res.reply.lower()
        assert "time zone" in res.reply.lower()

    def test_02_join_from_us_or_uae(self, engine):
        res = engine.handle_message("sess_intl_02", "can someone join from the US or UAE?")
        assert res.intent == "international_eligibility"
        assert "anyone globally can join from any country" in res.reply.lower()

    def test_03_accept_students_from_other_countries(self, engine):
        res = engine.handle_message("sess_intl_03", "do you accept students from other countries?")
        assert res.intent == "international_eligibility"
        assert "anyone globally can join from any country" in res.reply.lower()

    def test_04_attend_classes_from_dubai(self, engine):
        res = engine.handle_message("sess_intl_04", "can i attend classes from dubai")
        assert res.intent == "international_eligibility"
        assert "anyone globally can join from any country" in res.reply.lower()

    def test_05_international_students_join(self, engine):
        res = engine.handle_message("sess_intl_05", "can international students join")
        assert res.intent == "international_eligibility"
        assert "anyone globally can join from any country" in res.reply.lower()

    def test_06_anyone_globally_join(self, engine):
        res = engine.handle_message("sess_intl_06", "can anyone globally join")
        assert res.intent == "international_eligibility"
        assert "anyone globally can join from any country" in res.reply.lower()

    def test_07_context_switch_to_saudi_arabia(self, engine):
        session_id = "sess_ctx_intl"
        res1 = engine.handle_message(session_id, "Tell me about Confident Speaker")
        assert res1.intent == "confident_speaker"

        res2 = engine.handle_message(session_id, "Can I join from Saudi Arabia?")
        assert res2.intent == "international_eligibility"
        assert "anyone globally can join from any country" in res2.reply.lower()


class TestGuardrailsAndSafety:
    """Test demo capability gates and name extraction boundaries."""

    def test_demo_capability_disabled(self):
        assert DEMO_TRANSACTION_ENABLED is False

    def test_non_names_are_not_captured_as_names(self, engine):
        for phrase in ["Saudi", "Arabia", "First", "Second"]:
            res = engine.handle_message(f"sess_name_{phrase}", phrase)
            assert f"Nice to meet you, {phrase}" not in res.reply


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
