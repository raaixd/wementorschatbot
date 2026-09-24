"""
Targeted Verification Test Suite for Verified Knowledge Gaps:
1. Grades 11-12 / JEE / NEET — NOT SUPPORTED
2. State Board Curricula — SUPPORTED
3. Missed Class / Rescheduling Policy (genuine reason + specific topic)
4. Scholarships / Discounts — NONE CURRENTLY
5. Mentor Qualifications — HARD-CODED ONLY FOR EXPLICIT MENTOR QUESTIONS
   (with negative injection checks for unrelated queries)
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import database, personality
from app.knowledge import load_entries
from app.conversation import ConversationEngine


@pytest.fixture(scope="module")
def engine():
    os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test_knowledge_gaps.db")
    database.init_db()
    entries = load_entries()
    return ConversationEngine(entries)


# ==============================================================================
# 1. Grades 11–12 / JEE / NEET — NOT SUPPORTED
# ==============================================================================
class TestGrade11_12_JEE_NEET_Unsupported:
    @pytest.mark.parametrize(
        "query",
        [
            "Do you teach class 11?",
            "Do you teach class 12?",
            "Do you teach 11th standard physics?",
            "Do you provide 12th CBSE coaching?",
            "Do you offer JEE preparation?",
            "Do you offer NEET coaching?",
            "Can I join for class 11?",
            "Do you have programs for senior secondary students?",
            "Do you offer JEE coaching?",
            "Do you offer NEET preparation?",
            "Do you teach physics in Grade 11?",
            "CBSE Grade 12 coaching",
        ],
    )
    def test_unsupported_queries(self, engine, query):
        res = engine.handle_message(f"sess_unsupported_{hash(query)}", query)
        reply = res.reply.lower()

        # Must be classified as unsupported intent
        assert res.intent == "grade_11_12_unsupported"

        # Clearly states not offered / not supported
        assert (
            "not currently offer" in reply
            or "not offered" in reply
            or "does not offer" in reply
        )
        assert "11" in reply or "12" in reply or "jee" in reply or "neet" in reply

        # Preserves Grade 3–10 academic structure
        assert "grades 3–10" in reply or "grades 3-10" in reply or "foundation years" in reply

        # Clarifies Confident Speaker is available to learners of all ages
        assert "confident speaker" in reply
        assert "all ages" in reply

        # Must NOT imply they are planned or available
        assert "coming soon" not in reply
        assert "will launch" not in reply
        assert "planned" not in reply

    def test_confident_speaker_for_older_students_not_blocked(self, engine):
        # A question specifically about Confident Speaker for a Grade 11 student
        # should direct to Confident Speaker, not get confused with academic coaching
        res = engine.handle_message("sess_cs_11", "Can a class 11 student join Confident Speaker?")
        assert "confident speaker" in res.reply.lower()
        assert res.intent != "grade_11_12_unsupported" or "confident speaker" in res.reply.lower()


# ==============================================================================
# 2. State Board Curricula — SUPPORTED
# ==============================================================================
class TestStateBoardCurriculaSupported:
    @pytest.mark.parametrize(
        "query",
        [
            "Do you support state boards?",
            "Do you teach state board students?",
            "Can you teach according to my state board syllabus?",
            "Do you support state board curriculum?",
            "Do you teach students following a state board?",
        ],
    )
    def test_state_board_inquiries(self, engine, query):
        res = engine.handle_message(f"sess_stateboard_{hash(query)}", query)
        reply = res.reply.lower()

        # Must confirm State Board support
        assert "state board" in reply
        assert (
            "yes" in reply
            or "supports state board" in reply
            or "in addition to cbse" in reply
            or "cbse, icse" in reply
        )

        # Must preserve existing curricula
        assert "cbse" in reply

        # Must not claim fake accreditation
        assert "accredited by" not in reply
        assert "official state government" not in reply

    def test_general_curricula_includes_state_board(self, engine):
        res = engine.handle_message("sess_curricula_all", "Which curricula are supported?")
        reply = res.reply.lower()
        assert "cbse" in reply
        assert "icse" in reply
        assert "state board" in reply


# ==============================================================================
# 3. Missed Class / Rescheduling Policy
# ==============================================================================
class TestMissedClassPolicy:
    @pytest.mark.parametrize(
        "query",
        [
            "What happens if my child misses a class?",
            "Can I reschedule a missed class?",
            "What if my son misses his class?",
            "My daughter couldn't attend her class, what happens?",
            "Can a missed class be rescheduled?",
            "Do you provide a makeup class?",
        ],
    )
    def test_missed_class_policy(self, engine, query):
        res = engine.handle_message(f"sess_missed_{hash(query)}", query)
        reply = res.reply.lower()

        assert res.intent == "missed_classes"

        # Must be conditional on genuine reason
        assert "genuine reason" in reply

        # Must specify rescheduled class for that specific topic
        assert "specific topic" in reply
        assert "rescheduled" in reply or "catch-up" in reply or "catch up" in reply

        # Must NOT invent arbitrary constraints or guarantees
        assert "within 24 hours" not in reply
        assert "refund" not in reply
        assert "maximum of" not in reply
        assert "cancellation fee" not in reply


# ==============================================================================
# 4. Scholarships / Discounts — NONE CURRENTLY
# ==============================================================================
class TestScholarshipsAndDiscounts:
    @pytest.mark.parametrize(
        "query",
        [
            "Do you offer scholarships?",
            "Are there any scholarships?",
            "Do you have discounts?",
            "Do you offer sibling discounts?",
            "Is there any discount available?",
            "Do you have financial assistance?",
            "Can I get a discount?",
            "Do you offer financial aid?",
        ],
    )
    def test_scholarships_discounts_none_offered(self, engine, query):
        res = engine.handle_message(f"sess_discount_{hash(query)}", query)
        reply = res.reply.lower()

        assert res.intent == "scholarships_discounts"

        # Clearly states scholarships/discounts are not currently offered
        assert (
            "not currently offer scholarships or discounts" in reply
            or "does not currently offer" in reply
        )

        # Must NOT invent discounts or financial aid offers
        assert "we offer a 10%" not in reply
        assert "sibling discount of" not in reply
        assert "promo code" not in reply

        # Must NOT say "contact us to see if a discount is available"
        assert "see if a discount is available" not in reply
        assert "negotiate" not in reply


# ==============================================================================
# 5. Mentor Qualifications — HARD-CODED ONLY FOR EXPLICIT MENTOR QUESTIONS
# ==============================================================================
class TestMentorQualifications:
    def test_general_mentor_qualifications(self, engine):
        res = engine.handle_message("sess_mq_gen", "What qualifications do your mentors have?")
        reply = res.reply

        assert res.intent == "mentor_qualifications"

        # English Communication Mentor verified credentials
        assert "Ph.D. in English Literature" in reply
        assert "TEFL Certified" in reply
        assert "IELTS Band 8.0" in reply

        # Academic Mentor verified credentials
        assert "M.Sc. Biotechnology" in reply
        assert "Gold Medalist" in reply
        assert "B.Ed." in reply

        # Must NOT claim unverified details
        assert "years of experience" not in reply.lower()
        assert "background check" not in reply.lower()
        assert "oxford" not in reply.lower()
        assert "harvard" not in reply.lower()

    def test_english_mentor_explicit_questions(self, engine):
        queries = [
            "Tell me about your English mentor.",
            "What are the qualifications of the English communication mentor?",
            "What is the English mentor's background?",
            "What certifications does the English mentor have?",
        ]
        for q in queries:
            res = engine.handle_message(f"sess_mq_en_{hash(q)}", q)
            reply = res.reply
            assert res.intent == "mentor_qualifications"
            assert "Ph.D. in English Literature" in reply
            assert "TEFL Certified" in reply
            assert "IELTS Band 8.0" in reply
            assert "years of experience" not in reply.lower()

    def test_academic_mentor_explicit_questions(self, engine):
        queries = [
            "What qualifications does the academic mentor have?",
            "Tell me about your academic mentor.",
            "What is the academic mentor's background?",
        ]
        for q in queries:
            res = engine.handle_message(f"sess_mq_ac_{hash(q)}", q)
            reply = res.reply
            assert res.intent == "mentor_qualifications"
            assert "M.Sc. Biotechnology" in reply
            assert "Gold Medalist" in reply
            assert "B.Ed." in reply
            assert "years of experience" not in reply.lower()

    def test_are_your_mentors_qualified(self, engine):
        res = engine.handle_message("sess_mq_qual", "Are your mentors qualified?")
        assert res.intent == "mentor_qualifications"
        assert "Ph.D. in English Literature" in res.reply
        assert "M.Sc. Biotechnology" in res.reply

    # ==========================================================================
    # NEGATIVE INJECTION TESTS:
    # Mentor qualifications must NOT be injected into unrelated answers!
    # ==========================================================================
    @pytest.mark.parametrize(
        "query",
        [
            "What is Confident Speaker?",
            "How does personalized mentoring work?",
            "What subjects do you teach?",
            "How are classes conducted?",
            "Tell me about Grade 8 Maths.",
        ],
    )
    def test_no_mentor_qualification_injection(self, engine, query):
        res = engine.handle_message(f"sess_mq_neg_{hash(query)}", query)
        reply = res.reply.lower()

        # Must NOT mention specific qualifications in unrelated responses
        assert "ph.d" not in reply
        assert "biotechnology" not in reply
        assert "gold medalist" not in reply
        assert "tefl" not in reply
        assert "ielts band 8.0" not in reply
        assert "b.ed" not in reply


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
