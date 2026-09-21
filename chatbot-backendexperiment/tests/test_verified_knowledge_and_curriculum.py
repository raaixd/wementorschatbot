"""
Test suite for verified WeMentors knowledge and behavior:
- Confident Speaker 4 curriculum areas (Public Speaking, Business English, General Communicative Skills, IELTS Preparation)
- Online class delivery (WeMentors LMS + Google Meet)
- Regular class duration (~50 min) vs free demo class duration (30 min)
- Class frequency (Academic typically 5/week vs Confident Speaker typically 3-5/week)
- Personalized learning / mentoring adapted to individual pace and learning needs
- Missed classes catch-up sessions
- Parent progress updates via weekly meetings and mentor progress reports
- Academic chapter evaluation and need-based intervention classes
- Critical IELTS vs Senior School board prep disambiguation
- Multi-turn conversational context flows
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


class TestVerifiedKnowledgeAndCurriculum:

    # 1. Full overview contains all four curriculum areas
    def test_01_confident_speaker_overview_curriculum(self, engine):
        res = engine.handle_message("sess_01", "Tell me about Confident Speaker")
        assert res.intent == "confident_speaker"
        assert "Public Speaking" in res.reply
        assert "Business English" in res.reply
        assert "General Communicative Skills" in res.reply
        assert "IELTS Preparation" in res.reply

    # 2. Public Speaking routing
    def test_02_public_speaking_routing(self, engine):
        res = engine.handle_message("sess_02", "Do you teach public speaking?")
        assert res.intent == "confident_speaker_public_speaking"
        assert "Public Speaking" in res.reply
        assert "**Confident Speaker**" not in res.reply  # concise, not full card dump

    # 3. Business English routing
    def test_03_business_english_routing(self, engine):
        res = engine.handle_message("sess_03", "Do you offer Business English?")
        assert res.intent == "confident_speaker_business_english"
        assert "Business English" in res.reply
        assert "**Confident Speaker**" not in res.reply

    # 4. General Communicative Skills routing
    def test_04_general_communicative_skills_routing(self, engine):
        res = engine.handle_message("sess_04", "Do you teach general communication skills?")
        assert res.intent == "confident_speaker_general_communicative"
        assert "General Communicative Skills" in res.reply or "everyday" in res.reply.lower()

    # 5. IELTS routing
    def test_05_ielts_routing(self, engine):
        res = engine.handle_message("sess_05", "Do you offer IELTS preparation?")
        assert res.intent == "confident_speaker_ielts"
        assert "IELTS" in res.reply
        assert "senior-school" not in str(res.matched_entry_ids).lower()
        assert "Senior School" not in res.reply

    # 6. Specific curriculum question does not dump the full card
    def test_06_specific_curriculum_question_no_card_dump(self, engine):
        res = engine.handle_message("sess_06", "What does Confident Speaker teach?")
        assert res.intent == "confident_speaker_curriculum"
        assert "Public Speaking" in res.reply
        assert "Business English" in res.reply
        assert "General Communicative Skills" in res.reply
        assert "IELTS Preparation" in res.reply
        assert "**Grades:** All Ages" not in res.reply  # Not the full overview card

    # 7. Student/college Public Speaking context
    def test_07_student_college_public_speaking(self, engine):
        res = engine.handle_message("sess_07", "I'm a college student and want to improve my communication skills.")
        assert res.intent == "confident_speaker_public_speaking"
        assert "Public Speaking" in res.reply or "communication" in res.reply.lower()

    # 8. Professional/businessperson Business English context
    def test_08_professional_business_english(self, engine):
        res = engine.handle_message("sess_08", "I'm a professional and want to improve my English for work.")
        assert res.intent == "confident_speaker_business_english"
        assert "Business English" in res.reply

    # 9. Everyday-English routing
    def test_09_everyday_english_routing(self, engine):
        res = engine.handle_message("sess_09", "I want to improve my everyday English.")
        assert res.intent == "confident_speaker_general_communicative"
        assert "General Communicative" in res.reply or "everyday" in res.reply.lower()

    # 10. IELTS-specific routing
    def test_10_ielts_prep_phrase(self, engine):
        res = engine.handle_message("sess_10", "I want to prepare for IELTS.")
        assert res.intent == "confident_speaker_ielts"
        assert "IELTS" in res.reply
        assert "Senior School" not in res.reply

    # 11. Online class answer
    def test_11_online_class_answer(self, engine):
        res = engine.handle_message("sess_11", "Are classes online?")
        assert res.intent == "online_classes"
        assert "online" in res.reply.lower()
        assert "Google Meet" in res.reply
        assert "LMS" in res.reply

    # 12. WeMentors LMS answer
    def test_12_lms_answer(self, engine):
        res = engine.handle_message("sess_12", "Do you have your own LMS?")
        assert "LMS" in res.reply
        assert "online" in res.reply.lower()

    # 13. Google Meet answer
    def test_13_google_meet_answer(self, engine):
        res = engine.handle_message("sess_13", "Do you use Google Meet?")
        assert "Google Meet" in res.reply
        assert "online" in res.reply.lower()

    # 14. Regular class duration = ~50 minutes
    def test_14_regular_class_duration(self, engine):
        res = engine.handle_message("sess_14", "How long is each class?")
        assert res.intent == "class_duration"
        assert "50 minutes" in res.reply.lower()

    # 15. Demo duration remains 30 minutes when explicitly asking about the demo
    def test_15_demo_class_duration(self, engine):
        res = engine.handle_message("sess_15", "How long is the demo class?")
        assert res.intent == "demo_information"
        assert "30 minutes" in res.reply.lower() or "30-minute" in res.reply.lower()

    # 16. Academic = typically 5/week
    def test_16_academic_frequency(self, engine):
        res = engine.handle_message("sess_16", "How many classes per week does the academic program have?")
        assert res.intent == "class_frequency_academic"
        assert "5 classes per week" in res.reply.lower() or "5 classes" in res.reply.lower()

    # 17. Confident Speaker = typically 3-5/week
    def test_17_confident_speaker_frequency(self, engine):
        res = engine.handle_message("sess_17", "How many classes per week does Confident Speaker have?")
        assert res.intent == "class_frequency_confident_speaker"
        assert "3–5 classes" in res.reply or "3-5 classes" in res.reply or "3–5" in res.reply

    # 18. No incorrect cross-program frequency leakage (generic frequency)
    def test_18_generic_frequency_both_distinguished(self, engine):
        res = engine.handle_message("sess_18", "How many classes do you have per week?")
        assert res.intent == "class_frequency_general"
        assert "5 classes" in res.reply
        assert "3–5 classes" in res.reply or "3-5 classes" in res.reply

    # 19. Personalized learning explains individual pace/needs
    def test_19_personalized_learning_pace(self, engine):
        res = engine.handle_message("sess_19", "What does personalized mentoring mean?")
        assert res.intent == "personalized_learning_pace"
        assert "individual pace" in res.reply.lower()
        assert "learning needs" in res.reply.lower()

    # 20. Pace-based follow-up works
    def test_20_pace_based_followup(self, engine):
        res = engine.handle_message("sess_20", "What does personalized learning mean?")
        assert "pace" in res.reply.lower()
        assert "needs" in res.reply.lower()

    # 21. Missed class -> catch-up session
    def test_21_missed_class_catchup(self, engine):
        res = engine.handle_message("sess_21", "What happens if my child misses a class?")
        assert res.intent == "missed_classes"
        assert "catch-up" in res.reply.lower() or "catch up" in res.reply.lower()

    # 22. Weekly parent meetings
    def test_22_parent_weekly_meetings(self, engine):
        res = engine.handle_message("sess_22", "How do parents know about their child's progress?")
        assert res.intent == "parent_updates"
        assert "weekly meeting" in res.reply.lower() or "weekly" in res.reply.lower()

    # 23. Mentor progress reports
    def test_23_mentor_progress_reports(self, engine):
        res = engine.handle_message("sess_23", "Do mentors provide updates to parents?")
        assert res.intent == "parent_updates"
        assert "reports" in res.reply.lower() or "progress" in res.reply.lower()

    # 24. Parent updates not incorrectly restricted to Middle School
    def test_24_parent_updates_general(self, engine):
        res = engine.handle_message("sess_24", "Do parents get progress reports?")
        assert res.intent == "parent_updates"
        assert "Middle School covers Grades 6–8" not in res.reply

    # 25. Chapter evaluation
    def test_25_chapter_evaluation(self, engine):
        res = engine.handle_message("sess_25", "What happens after each chapter?")
        assert res.intent == "academic_chapter_evaluation"
        assert "evaluated after each chapter" in res.reply.lower() or "chapter" in res.reply.lower()

    # 26. Evaluation-based intervention classes
    def test_26_evaluation_intervention(self, engine):
        res = engine.handle_message("sess_26", "What happens if a student needs more help?")
        assert res.intent == "academic_chapter_evaluation"
        assert "intervention" in res.reply.lower()

    # 27. Does not imply every student automatically receives intervention
    def test_27_intervention_conditional(self, engine):
        res = engine.handle_message("sess_27", "What happens after each chapter?")
        # should say "can be provided" or "where additional support is needed"
        assert "where additional support is needed" in res.reply.lower() or "can be provided" in res.reply.lower()

    # 28. Does not incorrectly apply academic intervention behavior to Confident Speaker
    def test_28_confident_speaker_not_academic_intervention(self, engine):
        res = engine.handle_message("sess_28", "Tell me about Confident Speaker")
        assert "intervention class" not in res.reply.lower()

    # 29. IELTS does NOT route to Senior School
    def test_29_ielts_not_senior_school(self, engine):
        for q in [
            "Do you offer IELTS preparation?",
            "I want to prepare for IELTS.",
            "Is IELTS included in Confident Speaker?",
            "Does Confident Speaker have an IELTS course?",
        ]:
            res = engine.handle_message(f"sess_29_{hash(q)}", q)
            assert res.intent == "confident_speaker_ielts"
            assert "Senior School" not in res.reply
            assert "Grades 9–10" not in res.reply

    # 30. Board-exam preparation still routes correctly to Senior School
    def test_30_board_exam_prep_routes_senior_school(self, engine):
        res1 = engine.handle_message("sess_30_1", "How does Senior School Focus prepare students for board exams?")
        assert res1.intent in ("senior_school_overview", "board_exam")
        assert "Senior School" in res1.reply

        res2 = engine.handle_message("sess_30_2", "Do you provide board exam preparation?")
        assert res2.intent in ("senior_school_overview", "board_exam")
        assert "Senior School" in res2.reply

    # 31. Existing Grade 3–5 / 6–8 / 9–10 routing remains correct
    def test_31_grade_routings_preserved(self, engine):
        res_g4 = engine.handle_message("sess_31_g4", "Courses for Grade 4")
        assert res_g4.intent == "foundation_years_overview"
        assert "Foundation Years" in res_g4.reply

        res_g7 = engine.handle_message("sess_31_g7", "Courses for Grade 7")
        assert res_g7.intent == "middle_school_overview"
        assert "Middle School" in res_g7.reply

        res_g10 = engine.handle_message("sess_31_g10", "Courses for Grade 10")
        assert res_g10.intent == "senior_school_overview"
        assert "Senior School" in res_g10.reply

    # 32. Existing Grade 1–2 guardrail remains correct
    def test_32_grade_1_2_guardrail(self, engine):
        res_g1 = engine.handle_message("sess_32_g1", "Do you have classes for 1st grade?")
        assert res_g1.intent == "grade_1_2_unavailable"
        assert "does not offer courses for first grade" in res_g1.reply.lower()

    # 33. Multi-turn: Confident Speaker -> IELTS -> Business English -> frequency
    def test_33_multiturn_cs_ielts_business_frequency(self, engine):
        sess = "sess_33"
        r1 = engine.handle_message(sess, "Tell me about Confident Speaker.")
        assert r1.intent == "confident_speaker"
        assert "Curriculum:" in r1.reply

        r2 = engine.handle_message(sess, "Does it include IELTS?")
        assert r2.intent == "confident_speaker_ielts"
        assert "IELTS" in r2.reply

        r3 = engine.handle_message(sess, "What about Business English?")
        assert r3.intent == "confident_speaker_business_english"
        assert "Business English" in r3.reply

        r4 = engine.handle_message(sess, "How many classes are there per week?")
        assert r4.intent == "class_frequency_confident_speaker"
        assert "3–5 classes" in r4.reply or "3-5 classes" in r4.reply

    # 34. Multi-turn: Business English -> duration -> frequency
    def test_34_multiturn_business_duration_frequency(self, engine):
        sess = "sess_34"
        r1 = engine.handle_message(sess, "Tell me about Business English.")
        assert r1.intent == "confident_speaker_business_english"
        assert "Business English" in r1.reply

        r2 = engine.handle_message(sess, "How long are the classes?")
        assert r2.intent == "class_duration"
        assert "50 minutes" in r2.reply.lower()

        r3 = engine.handle_message(sess, "How often are they?")
        assert r3.intent == "class_frequency_confident_speaker"
        assert "3–5" in r3.reply or "3-5" in r3.reply

    # 35. Multi-turn: Academic -> frequency -> chapter evaluation
    def test_35_multiturn_academic_frequency_evaluation(self, engine):
        sess = "sess_35"
        r1 = engine.handle_message(sess, "Tell me about the academic programs.")
        assert r1.intent == "academic_courses_overview"

        r2 = engine.handle_message(sess, "How many classes are there each week?")
        assert r2.intent == "class_frequency_academic"
        assert "5 classes per week" in r2.reply.lower() or "5 classes" in r2.reply.lower()

        r3 = engine.handle_message(sess, "What happens after each chapter?")
        assert r3.intent == "academic_chapter_evaluation"
        assert "chapter" in r3.reply.lower()
        assert "intervention" in r3.reply.lower()

    # 36. Missed class -> catch-up
    def test_36_missed_class_session(self, engine):
        sess = "sess_36"
        res = engine.handle_message(sess, "What happens if my child misses a class?")
        assert res.intent == "missed_classes"
        assert "catch-up" in res.reply.lower() or "catch up" in res.reply.lower()

    # 37. Multi-turn: Personalized learning -> parent updates
    def test_37_multiturn_personalized_parent_updates(self, engine):
        sess = "sess_37"
        r1 = engine.handle_message(sess, "What does personalized mentoring mean?")
        assert r1.intent == "personalized_learning_pace"
        assert "individual pace" in r1.reply.lower()

        r2 = engine.handle_message(sess, "How do parents know about their child's progress?")
        assert r2.intent == "parent_updates"
        assert "weekly" in r2.reply.lower()
        assert "reports" in r2.reply.lower() or "progress" in r2.reply.lower()

    # 38. IELTS queries route to confident_speaker_ielts
    def test_38_ielts_routing_variants(self, engine):
        queries = [
            "ielts prep",
            "IELTS prep",
            "ielts preparation",
            "I want IELTS prep",
            "Do you offer IELTS prep?",
            "I need IELTS preparation",
            "Can you help with IELTS prep?",
            "I'm looking for IELTS prep",
            "Do you provide IELTS coaching?",
            "I want to prepare for IELTS",
        ]
        for q in queries:
            res = engine.handle_message(f"sess_ielts_{hash(q)}", q)
            assert res.intent == "confident_speaker_ielts", f"Failed for {q}: intent was {res.intent}"
            assert "IELTS" in res.reply
            assert "senior-school" not in str(res.matched_entry_ids).lower()
            assert "Senior School" not in res.reply
            assert "Thanks, Ielts Prep" not in res.reply
            assert "demo_booking" != res.intent
            assert "standalone_name" != res.intent

    # 39. IELTS does not route to demo or name even after previous message ends with demo offer
    def test_39_ielts_multiturn_after_demo_offer(self, engine):
        sess = "sess_39_ielts_after_demo"
        # Turn 1: bot response ends with Book Free Demo CTA
        engine.handle_message(sess, "Tell me about Confident Speaker")
        # Turn 2: "ielts prep" must NOT be treated as user name or demo lead
        res = engine.handle_message(sess, "ielts prep")
        assert res.intent == "confident_speaker_ielts"
        assert "IELTS" in res.reply
        assert "Thanks, Ielts Prep" not in res.reply
        assert "standalone_name" != res.intent

    # 40. Communication and communicative skills queries route to confident_speaker_general_communicative
    def test_40_general_communicative_routing(self, engine):
        queries = [
            "communication skills",
            "communicative skills",
            "general communicative skills",
            "I want to improve my communication skills",
            "I want to improve my communicative skills",
            "I want better everyday English",
            "I want to improve my English communication",
        ]
        for q in queries:
            res = engine.handle_message(f"sess_comm_{hash(q)}", q)
            assert res.intent == "confident_speaker_general_communicative", f"Failed for {q}: intent was {res.intent}"
            assert "General Communicative Skills" in res.reply or "everyday English" in res.reply
            assert "rote grammar drills" in res.reply
            assert "**Confident Speaker**" not in res.reply  # concise track response, not full card dump

    # 41. Avoid overmatching: general queries preserve their respective intents
    def test_41_avoid_overmatching_communication(self, engine):
        # Explicit overview asks continue to return full overview
        res_cs = engine.handle_message("sess_41_cs", "Tell me about Confident Speaker")
        assert res_cs.intent == "confident_speaker"
        assert "**Confident Speaker**" in res_cs.reply

        res_mentor = engine.handle_message("sess_41_mentor", "How does mentoring work?")
        assert res_mentor.intent == "mentoring_approach"

        res_attn = engine.handle_message("sess_41_attn", "Do you provide individual attention?")
        assert res_attn.intent == "one_on_one_general"

        res_progs = engine.handle_message("sess_41_progs", "What are the programs?")
        assert res_progs.intent == "faq"

    # 42. Genuine demo/booking queries continue to work
    def test_42_genuine_demo_booking_intents(self, engine):
        for q in ["I want to book a demo", "Can I book a demo?", "I want to schedule a demo", "How do I book a demo?"]:
            res = engine.handle_message(f"sess_demo_{hash(q)}", q)
            assert res.intent == "demo_booking", f"Failed for {q}: intent was {res.intent}"
            assert "Book Free Demo" in res.reply

    # 43. Academic board exam preparation preserved
    def test_43_board_exam_prep_preserved(self, engine):
        res = engine.handle_message("sess_43_board", "How do you prepare students for board exams?")
        assert res.intent in ("board_exam", "senior_school_overview")
        assert "confident_speaker_ielts" != res.intent
        assert "IELTS" not in res.reply
        assert "dedicated personal mentor" in res.reply.lower() or "board" in res.reply.lower()

    # 44. Specific Confident Speaker skills query response
    def test_44_confident_speaker_skills_covered_response(self, engine):
        res = engine.handle_message("sess_44_skills", "What skills are covered in Confident Speaker?")
        assert res.intent == "confident_speaker_scope"
        assert "The program covers four curriculum areas:" in res.reply
        assert "- Public Speaking — build confidence and speaking skills for school and college." in res.reply
        assert "- Business English — improve professional and workplace communication." in res.reply
        assert "- General Communicative Skills — develop practical English for everyday conversations." in res.reply
        assert "- IELTS Preparation — prepare for IELTS with guided practice and mentoring." in res.reply
        assert "rote grammar drills" in res.reply
        assert "Book Free Demo" in res.reply

