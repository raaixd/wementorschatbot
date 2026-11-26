"""Fallback replies and internal wording."""
import sys
import uuid
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import database
from app.conversation import ConversationEngine
from app.knowledge import load_entries
from app.presentation import hide_internal_terms, stated_grade


@pytest.fixture(scope="module")
def engine():
    database.init_db()
    return ConversationEngine(load_entries())


def _ask(engine, message):
    session_id = str(uuid.uuid4())
    database.ensure_session(session_id)
    database.log_message(session_id, "user", message, source="text_input")
    return engine.handle_message(session_id, message).reply


@pytest.mark.parametrize("message,grade", [
    ("my kid is in 7th and hates maths", 7),
    ("class 8", 8),
    ("wats the deal for clas 4", 4),
    ("my 10th grader needs help", 10),
    ("std 5 student", 5),
    ("can we meet on the 7th", None),
    ("is it 5 classes a week", None),
    ("class 5 days a week", None),
    ("grade 7 and grade 9", None),
    ("grade 12", None),
    ("we are in 2026", None),
])
def test_grade_is_only_read_when_clear(message, grade):
    assert stated_grade(message) == grade


def test_unmatched_message_with_a_grade_gets_that_program(engine):
    reply = _ask(engine, "my kid is in 7th and hates maths, can you help?")
    assert "Middle School (Grades 6–8)" in reply
    assert "may not have answered your exact question" in reply


def test_misspelt_fee_question_gets_the_fee_answer(engine):
    reply = _ask(engine, "wats the fees for clas 8")
    assert "Fee details are not published" in reply
    assert "admin@wementors.co" in reply


@pytest.mark.parametrize("raw,clean", [
    ("Fees are not published in our verified knowledge base, since",
     "Fees are not published, since"),
    ("Fee details are not published in our verified records, as", "Fee details are not published, as"),
    ("There is no separate entrance exam in our knowledge base.",
     "There is no separate entrance exam in the information we have published."),
])
def test_internal_terms_are_reworded(raw, clean):
    assert hide_internal_terms(raw) == clean


def test_visitor_never_sees_internal_terms(engine):
    for question in ("how much are the fees", "what is the admission process"):
        reply = _ask(engine, question).lower()
        assert "knowledge base" not in reply and "verified records" not in reply


@pytest.mark.parametrize("question", [
    "do you teach french?",
    "do u teach hindi",
])
def test_lists_do_not_pose_as_a_yes(engine, question):
    assert _ask(engine, question).startswith("I can’t confirm every detail")


@pytest.mark.parametrize("question", [
    "what subjects do you offer",
    "what are your mentors' qualifications",
    "what is the english mentor's background",
])
def test_ordinary_listing_questions_get_no_caveat(engine, question):
    assert "can’t confirm every detail" not in _ask(engine, question)
