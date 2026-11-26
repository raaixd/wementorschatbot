"""Regressions: visitor names crossing sessions, and the outcome-promise filter."""
import sys
import uuid
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import database, llm
from app.conversation import ConversationEngine
from app.knowledge import load_entries


@pytest.fixture(scope="module")
def engine():
    database.init_db()
    return ConversationEngine(load_entries())


def _ask(engine, session_id, message):
    database.ensure_session(session_id)
    database.log_message(session_id, "user", message, source="text_input")
    return engine.handle_message(session_id, message).reply


@pytest.mark.parametrize("closer", ["hello", "help", "thank you"])
def test_name_never_reaches_another_session(engine, closer):
    # These openers return early, which used to leave the greeting on the engine.
    _ask(engine, str(uuid.uuid4()), f"My name is Priya, {closer}")
    other = _ask(engine, str(uuid.uuid4()), "do you offer online classes?")
    assert "Priya" not in other


def test_name_is_still_used_for_its_own_session(engine):
    reply = _ask(engine, str(uuid.uuid4()), "My name is Priya, tell me about Grade 7")
    assert reply.startswith("Nice to meet you, Priya!")


@pytest.mark.parametrize("promise", [
    "We guarantee that your child will top the exam.",
    "We guarantee full marks.",
    "We promise full marks for every student.",
    "Your child will definitely score 95 percent.",
    "Results are guaranteed.",
    "Don't worry, we guarantee a rank.",
    "Your child will become fluent overnight.",
])
def test_promises_are_replaced_whole(promise):
    assert llm.remove_outcome_promises(promise) == llm.NO_GUARANTEE_SENTENCE


@pytest.mark.parametrize("safe", [
    "We cannot guarantee that your child will top the exam.",
    "WeMentors does not guarantee marks, ranks, or exam scores.",
    "We don't promise specific grades.",
    "There are no guaranteed results.",
    "Can you guarantee a rank?",
    "Mentors will definitely help your child practise.",
])
def test_denials_and_ordinary_text_are_untouched(safe):
    assert llm.remove_outcome_promises(safe) == safe


def test_only_the_offending_line_changes_and_llm_output_is_covered():
    text = "## Board prep\n\n- Weekly practice.\n- We guarantee full marks.\n- Doubt clinics."
    expected = f"## Board prep\n\n- Weekly practice.\n- {llm.NO_GUARANTEE_SENTENCE}\n- Doubt clinics."
    assert llm.sanitize_llm_output(text) == expected
