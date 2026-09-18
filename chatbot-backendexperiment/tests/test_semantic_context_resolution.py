"""
Test Suite: Semantic Context & Entity Resolution
Validates that:
1. "practice sessions" resolves differently depending on active program / topic.
2. An explicit entity/program in the CURRENT user message overrides previous context (CURRENT EXPLICIT REFERENT > PREVIOUS CONTEXT).
3. Confident Speaker practice sessions vs Academic course practice sessions are cleanly distinguished.
4. Non-regression of name detection, cancellation, demo safety, location, and eligibility.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.conversation import ConversationEngine
from app.leads import DEMO_TRANSACTION_ENABLED
from app.knowledge import load_entries
import app.leads as leads


@pytest.fixture(scope="module")
def engine():
    entries = load_entries()
    return ConversationEngine(entries)


class TestSemanticContextResolution:
    """Test matrix for program context resolution and practice session distinctions."""

    def test_01_confident_speaker_then_practice_sessions(self, engine):
        """1. Confident Speaker -> 'What happens during practice sessions?' -> Confident Speaker practice answer."""
        session_id = "test_sess_01"
        res1 = engine.handle_message(session_id, "Tell me about Confident Speaker")
        assert res1.intent == "confident_speaker"
        assert "confident speaker" in res1.reply.lower()

        res2 = engine.handle_message(session_id, "What happens during practice sessions?")
        assert res2.intent == "confident_speaker_activities"
        assert "guided speaking practice" in res2.reply.lower()
        assert "practical conversation" in res2.reply.lower()
        assert "regular feedback" in res2.reply.lower()

    def test_02_confident_speaker_then_academic_practice_sessions(self, engine):
        """2. Confident Speaker -> 'What about practice sessions of academic courses?' -> academic-course answer, NOT Confident Speaker answer."""
        session_id = "test_sess_02"
        res1 = engine.handle_message(session_id, "Tell me about Confident Speaker")
        assert res1.intent == "confident_speaker"

        res2 = engine.handle_message(session_id, "What about practice sessions of academic courses?")
        assert res2.intent == "academic_sessions"
        assert "guided speaking practice" not in res2.reply.lower()
        assert "practical conversation" not in res2.reply.lower()
        assert "concept clarity" in res2.reply.lower()
        assert "personal mentor" in res2.reply.lower()

    def test_03_confident_speaker_then_academic_classes(self, engine):
        """3. Confident Speaker -> 'What about academic classes?' -> switches to academic context."""
        session_id = "test_sess_03"
        res1 = engine.handle_message(session_id, "Tell me about Confident Speaker")
        assert res1.intent == "confident_speaker"

        res2 = engine.handle_message(session_id, "What about academic classes?")
        assert res2.intent == "academic_courses_overview"
        assert "foundation years" in res2.reply.lower()
        assert "middle school" in res2.reply.lower()
        assert "senior school" in res2.reply.lower()
        assert "guided speaking practice" not in res2.reply.lower()

    def test_04_academic_program_then_sessions(self, engine):
        """4. Academic program -> 'What happens during sessions?' -> academic answer."""
        session_id = "test_sess_04"
        res1 = engine.handle_message(session_id, "Tell me about Middle School")
        assert "middle" in res1.intent

        res2 = engine.handle_message(session_id, "What happens during sessions?")
        assert res2.intent == "academic_sessions"
        assert "guided speaking practice" not in res2.reply.lower()
        assert "practical conversation" not in res2.reply.lower()
        assert "concept clarity" in res2.reply.lower()

    def test_05_academic_program_then_speaking_practice(self, engine):
        """5. Academic program -> 'What about speaking practice?' -> only switches to Confident Speaker if speaking is indicated."""
        session_id = "test_sess_05"
        res1 = engine.handle_message(session_id, "Tell me about Middle School")
        assert "middle" in res1.intent

        # 'speaking practice' explicitly specifies speaking/Confident Speaker
        res2 = engine.handle_message(session_id, "What about speaking practice?")
        assert res2.intent == "confident_speaker_activities"
        assert "guided speaking practice" in res2.reply.lower()

        # But 'practice for Grade 7' must NOT switch to Confident Speaker!
        session_id_b = "test_sess_05b"
        engine.handle_message(session_id_b, "Tell me about Middle School")
        res_b = engine.handle_message(session_id_b, "What about practice for Grade 7?")
        assert res_b.intent in ("grade7_maths_sessions", "academic_sessions", "middle_school_grades", "middle_school_overview")
        assert "guided speaking practice" not in res_b.reply.lower()

    def test_06_standalone_confident_speaker_practice_sessions(self, engine):
        """6. 'What about Confident Speaker practice sessions?' -> Confident Speaker answer."""
        session_id = "test_sess_06"
        res = engine.handle_message(session_id, "What about Confident Speaker practice sessions?")
        assert res.intent == "confident_speaker_activities"
        assert "guided speaking practice" in res.reply.lower()
        assert "practical conversation" in res.reply.lower()

    def test_07_standalone_grade7_maths_sessions(self, engine):
        """7. 'What about Grade 7 Maths sessions?' -> Grade 7 Maths / Middle School academic context."""
        session_id = "test_sess_07"
        res = engine.handle_message(session_id, "What about Grade 7 Maths sessions?")
        assert res.intent == "grade7_maths_sessions"
        assert "guided speaking practice" not in res.reply.lower()
        assert "math" in res.reply.lower()
        assert "middle school" in res.reply.lower()

    def test_08_academic_to_confident_speaker_switch(self, engine):
        """8. 'Actually, tell me about Confident Speaker.' -> switch from academic context to Confident Speaker."""
        session_id = "test_sess_08"
        engine.handle_message(session_id, "Tell me about Middle School")
        res = engine.handle_message(session_id, "Actually, tell me about Confident Speaker.")
        assert res.intent == "confident_speaker"
        assert "confident speaker" in res.reply.lower()
        assert "spoken english" in res.reply.lower() or "public speaking" in res.reply.lower()

    def test_09_confident_speaker_to_academic_courses_switch(self, engine):
        """9. 'Actually, what about academic courses?' -> switch from Confident Speaker context to academic context."""
        session_id = "test_sess_09"
        engine.handle_message(session_id, "Tell me about Confident Speaker")
        res = engine.handle_message(session_id, "Actually, what about academic courses?")
        assert res.intent == "academic_courses_overview"
        assert "foundation years" in res.reply.lower()
        assert "middle school" in res.reply.lower()
        assert "guided speaking practice" not in res.reply.lower()

    def test_10_non_regression_safeguards(self, engine):
        """10. Verify name detection, cancellation, demo safety, location, and eligibility do not regress."""
        assert DEMO_TRANSACTION_ENABLED is False

        # Name detection
        session_name = "test_sess_name"
        res_name = engine.handle_message(session_name, "My name is Raaid")
        assert res_name.intent == "explicit_name"
        assert "Raaid" in res_name.reply

        # Cancellation
        session_cancel = "test_sess_cancel"
        res_cancel = engine.handle_message(session_cancel, "Never mind")
        assert res_cancel.intent == "cancellation"
        assert "Nevermind" not in res_cancel.reply

        # Role disclosure not a name
        session_adult = "test_sess_adult"
        res_adult = engine.handle_message(session_adult, "Can I join if I'm an adult?")
        assert res_adult.intent == "eligibility_adult"
        assert "Adult" not in res_adult.reply or "Adults can join" in res_adult.reply

        # Location query not demo booking
        session_loc = "test_sess_loc"
        res_loc = engine.handle_message(session_loc, "Where are you located?")
        assert res_loc.intent == "location"
        assert "live and online" in res_loc.reply.lower()
        assert "book free demo" not in res_loc.reply.lower()

        # Demo transaction refusal
        session_demo = "test_sess_demo"
        res_demo = engine.handle_message(session_demo, "Book me a demo")
        assert res_demo.intent == "demo_transaction_request"
        assert "can't submit the demo booking directly" in res_demo.reply.lower() or "directly from the chat" in res_demo.reply.lower()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))

