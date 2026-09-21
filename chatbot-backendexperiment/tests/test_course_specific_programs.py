"""
Test Suite: Course-Specific Programs and Multi-Turn Context Resolution

Validates:
1. Canonical structured card responses for all 4 programs (Foundation Years, Middle School — All Subjects, Senior School Focus, Confident Speaker).
2. Grade-to-program routing:
   - Grade 3-5 -> Foundation Years
   - Grade 6-8 -> Middle School — All Subjects
   - Grade 9-10 -> Senior School Focus
   - Grade 1-2 -> grade_1_2_unavailable (with strict exclusion of Grade 10)
3. Confident Speaker is All Ages (students, professionals, homemakers).
4. Generic feature preservation (doubt clinics, practical labs remain general unless bound to Middle School).
5. Targeted aspect responses for subjects, features, grades (no full card dump for narrow queries).
6. Multi-turn context resolution for 'it', 'this program', 'the course'.
7. Clean context switching without leaking previous program details.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.conversation import ConversationEngine
from app.knowledge import load_entries


@pytest.fixture(scope="module")
def engine():
    entries = load_entries()
    return ConversationEngine(entries)


class TestCourseSpecificPrograms:
    """Test suite for the 4 canonical programs, grade routing, and context resolution."""

    def test_01_foundation_years_overview_structure(self, engine):
        res = engine.handle_message("test_fy_01", "Tell me about Foundation Years")
        assert res.intent == "foundation_years_overview"
        assert res.matched_entry_ids == ["program-foundation-years"]
        assert "**Foundation Years**" in res.reply
        assert "**Grades:** Grades 3–5" in res.reply
        assert "**Focus:**" in res.reply
        assert "- Strong fundamentals" in res.reply
        assert "- Curiosity-first learning" in res.reply
        assert "**Key features:**" in res.reply
        assert "- Concept games" in res.reply
        assert "- Visual learning" in res.reply
        assert "- Weekly progress notes for parents" in res.reply
        assert "Ready to explore the program? You can book a free 30-minute demo using the **Book Free Demo** button." in res.reply

    def test_02_middle_school_overview_structure(self, engine):
        res = engine.handle_message("test_ms_01", "What is the Middle School program?")
        assert res.intent == "middle_school_overview"
        assert res.matched_entry_ids == ["program-middle-school"]
        assert "**Middle School — All Subjects**" in res.reply
        assert "**Grades:** Grades 6–8" in res.reply
        assert "**Focus:**" in res.reply
        assert "- All subjects" in res.reply
        assert "- Doubt-solving" in res.reply
        assert "- Practical labs" in res.reply
        assert "**Key features:**" in res.reply
        assert "- Concept-based learning" in res.reply
        assert "- Practical problem sets" in res.reply
        assert "- Fortnightly doubt clinics" in res.reply
        assert "- Progress dashboard access" in res.reply
        assert "Ready to explore the program? You can book a free 30-minute demo using the **Book Free Demo** button." in res.reply

    def test_03_senior_school_focus_overview_structure(self, engine):
        res = engine.handle_message("test_ss_01", "What does Senior School Focus offer?")
        assert res.intent == "senior_school_overview"
        assert res.matched_entry_ids == ["program-senior-school"]
        assert "**Senior School Focus**" in res.reply
        assert "**Grades:** Grades 9–10" in res.reply
        assert "**Focus:**" in res.reply
        assert "- Mathematics" in res.reply
        assert "- Science" in res.reply
        assert "- Board Prep" in res.reply
        assert "**Key features:**" in res.reply
        assert "- Board-exam-precision coaching" in res.reply
        assert "- Dedicated personal mentor with individual attention" in res.reply
        assert "- Weekly mock tests with review" in res.reply
        assert "- Priority doubt-clearing access" in res.reply
        assert "Ready to explore the program? You can book a free 30-minute demo using the **Book Free Demo** button." in res.reply

    def test_04_confident_speaker_overview_structure(self, engine):
        res = engine.handle_message("test_cs_01", "Tell me about Confident Speaker")
        assert res.intent == "confident_speaker"
        assert res.matched_entry_ids == ["program-confident-speaker"]
        assert "**Confident Speaker**" in res.reply
        assert "**Grades:** All Ages" in res.reply
        assert "**Focus:**" in res.reply
        assert "- Spoken English" in res.reply
        assert "- Public Speaking" in res.reply
        assert "- Interview Skills" in res.reply
        assert "**Key features:**" in res.reply
        assert "- Conversation-first method, no rote grammar drills" in res.reply
        assert "- Small batches with individual attention for maximum speaking time" in res.reply
        assert "- Guided speaking practice & practical conversation with regular feedback" in res.reply
        assert "- Dedicated tracks for students, professionals & homemakers" in res.reply
        assert "Ready to explore the program? You can book a free 30-minute demo using the **Book Free Demo** button." in res.reply

    def test_05_grade_4_routing(self, engine):
        res = engine.handle_message("test_g4", "What program is available for Grade 4?")
        assert res.intent == "foundation_years_overview"
        assert "**Foundation Years**" in res.reply
        assert "Grades 3–5" in res.reply

    def test_06_grade_7_routing(self, engine):
        res = engine.handle_message("test_g7", "What program is available for Grade 7?")
        assert res.intent == "middle_school_overview"
        assert "**Middle School — All Subjects**" in res.reply
        assert "Grades 6–8" in res.reply

    def test_07_grade_10_routing_and_exclusion_fix(self, engine):
        res = engine.handle_message("test_g10", "What program is available for Grade 10?")
        assert res.intent == "senior_school_overview"
        assert "**Senior School Focus**" in res.reply
        assert "Grades 9–10" in res.reply
        assert "grade_1_2_unavailable" != res.intent
        assert "first grade or second grade" not in res.reply.lower()

        # Standalone Grade 10 query
        res_bare = engine.handle_message("test_g10_bare", "Grade 10")
        assert res_bare.intent == "senior_school_overview"
        assert "**Senior School Focus**" in res_bare.reply

    def test_08_generic_doubt_clinics_preservation(self, engine):
        res = engine.handle_message("test_doubt_gen", "How do doubt clinics work?")
        assert res.intent == "doubt_clinics"
        assert "middle school" not in res.reply.lower()
        assert "weMentors provides dedicated doubt-solving sessions" in res.reply.lower() or "fortnightly doubt clinics" in res.reply.lower()

    def test_09_generic_practical_labs_preservation(self, engine):
        res = engine.handle_message("test_labs_gen", "Tell me about practical labs.")
        assert res.intent == "practical_labs"
        assert "middle school" not in res.reply.lower()
        assert "practical labs" in res.reply.lower()

    def test_10_targeted_aspect_responses_no_card_dump(self, engine):
        # Foundation Years subjects
        res_fy_sub = engine.handle_message("test_aspect_fy", "What subjects are taught in Foundation Years?")
        assert res_fy_sub.intent == "foundation_subjects"
        assert "**Foundation Years**" not in res_fy_sub.reply  # Must be targeted, not full card
        assert "Mathematics, Science, English" in res_fy_sub.reply

        # Middle School subjects
        res_ms_sub = engine.handle_message("test_aspect_ms", "What subjects are covered in Middle School?")
        assert res_ms_sub.intent == "middle_school_subjects"
        assert "**Middle School — All Subjects**" not in res_ms_sub.reply
        assert "Mathematics, Science, English, and Social Studies" in res_ms_sub.reply

        # Senior School subjects
        res_ss_sub = engine.handle_message("test_aspect_ss", "What subjects are in Senior School Focus?")
        assert res_ss_sub.intent == "senior_school_subjects"
        assert "**Senior School Focus**" not in res_ss_sub.reply
        assert "Mathematics and Science" in res_ss_sub.reply

        # Confident Speaker scope
        res_cs_scope = engine.handle_message("test_aspect_cs", "What skills are covered in Confident Speaker?")
        assert res_cs_scope.intent == "confident_speaker_scope"
        assert "**Confident Speaker**" not in res_cs_scope.reply
        assert "Spoken English, Public Speaking, and Interview Skills" in res_cs_scope.reply

    def test_11_multiturn_pronoun_it_resolution(self, engine):
        # Foundation Years -> What grades is it for?
        sess_fy = "sess_mt_fy_it"
        engine.handle_message(sess_fy, "Tell me about Foundation Years")
        res_fy_g = engine.handle_message(sess_fy, "What grades is it for?")
        assert res_fy_g.intent == "foundation_grades"
        assert "Grades 3–5" in res_fy_g.reply

        # Senior School Focus -> What grades is it for?
        sess_ss = "sess_mt_ss_it"
        engine.handle_message(sess_ss, "What does Senior School Focus offer?")
        res_ss_g = engine.handle_message(sess_ss, "What grades is it for?")
        assert res_ss_g.intent == "senior_school_grades"
        assert "Grades 9–10" in res_ss_g.reply

    def test_12_multiturn_pronoun_this_program_resolution(self, engine):
        # Foundation Years -> What are the key features of this program?
        sess_fy = "sess_mt_fy_tp"
        engine.handle_message(sess_fy, "Tell me about Foundation Years")
        res_fy_f = engine.handle_message(sess_fy, "What are the key features of this program?")
        assert res_fy_f.intent == "foundation_features"
        assert "concept games" in res_fy_f.reply.lower()

        # Senior School Focus -> What are its key features?
        sess_ss = "sess_mt_ss_tp"
        engine.handle_message(sess_ss, "What does Senior School Focus offer?")
        res_ss_f = engine.handle_message(sess_ss, "What are its key features?")
        assert res_ss_f.intent == "senior_school_features"
        assert "board-exam readiness" in res_ss_f.reply.lower()

    def test_13_multiturn_the_course_resolution(self, engine):
        sess = "sess_mt_course"
        engine.handle_message(sess, "What does Senior School Focus offer?")
        res_sub = engine.handle_message(sess, "What subjects are taught in this course?")
        assert res_sub.intent == "senior_school_subjects"
        assert "mathematics and science" in res_sub.reply.lower()

    def test_14_clean_program_switch_no_leak(self, engine):
        sess = "sess_switch"
        # Turn 1: Foundation Years
        res1 = engine.handle_message(sess, "Tell me about Foundation Years")
        assert res1.intent == "foundation_years_overview"

        # Turn 2: Switch to Senior School Focus
        res2 = engine.handle_message(sess, "What does Senior School Focus offer?")
        assert res2.intent == "senior_school_overview"

        # Turn 3: Follow-up about "it" should resolve to Senior School, NOT Foundation
        res3 = engine.handle_message(sess, "What subjects are in it?")
        assert res3.intent == "senior_school_subjects"
        assert "mathematics and science" in res3.reply.lower()
        assert "environmental studies" not in res3.reply.lower()

    def test_15_confident_speaker_audience_all_ages(self, engine):
        res = engine.handle_message("test_cs_aud", "Who is Confident Speaker for?")
        assert res.intent == "confident_speaker_audience"
        assert "students" in res.reply.lower()
        assert "professionals" in res.reply.lower()
        assert "homemakers" in res.reply.lower()

    def test_16_grade_1_and_2_unavailable_guardrail(self, engine):
        res1 = engine.handle_message("test_g1_avail", "What program is available for Grade 1?")
        assert res1.intent == "grade_1_2_unavailable"
        assert "not offer courses for first grade or second grade yet" in res1.reply.lower()

        res2 = engine.handle_message("test_g2_avail", "What about 2nd grade?")
        assert res2.intent == "grade_1_2_unavailable"
        assert "not offer courses for first grade or second grade yet" in res2.reply.lower()


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

