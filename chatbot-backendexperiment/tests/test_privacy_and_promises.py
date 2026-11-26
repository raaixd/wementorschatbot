"""Regressions: visitor names crossing sessions, and the outcome-promise filter."""
import sys
import uuid
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import config, database, llm, personality, scope
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


@pytest.mark.parametrize("invented", [
    "Fees are \u20b92,500 per month.",
    "It costs Rs. 900 a class.",
    "About 3000 rupees monthly.",
    "Email us at sales@wementors.in.",
    "Call +91 99999 11111 today.",
])
def test_llm_output_with_invented_price_or_contact_is_discarded(invented):
    assert llm.sanitize_llm_output(invented) is None


@pytest.mark.parametrize("published", [
    "Email admin@wementors.co or call +91 76111 92227.",
    "WhatsApp +91-90398-03526 is the alternate number.",
    "Classes run about 50 minutes, 3 to 5 times a week, for Grades 3-10.",
])
def test_llm_output_with_published_details_is_kept(published):
    assert llm.sanitize_llm_output(published) == published


@pytest.mark.parametrize("subject", ["maths", "science"])
def test_one_word_subject_reply_sticks_to_published_facts(engine, subject):
    reply = _ask(engine, str(uuid.uuid4()), subject).lower()
    assert "fortnightly doubt clinics" in reply
    for unpublished in ("weekly doubt", "past paper", "mental math", "score improvement", "1-on-1"):
        assert unpublished not in reply


@pytest.mark.parametrize("request_text", [
    "can I get a mentor for guitar",
    "I need a tutor for French",
    "can i get a home tutor in Pune",
    "can you assign a female mentor",
    "do you have a mentor for my 4 year old",
    "can I get a mentor for SAT prep",
    "do you have a mentor who speaks Tamil",
    "can you arrange a mentor at midnight",
])
def test_unpublished_mentor_requests_are_not_confirmed(engine, request_text):
    reply = _ask(engine, str(uuid.uuid4()), request_text)
    assert reply == personality.MENTOR_REQUEST_UNCONFIRMED_RESPONSE


@pytest.mark.parametrize("request_text", [
    "find me a mentor for maths grade 7",
    "Can you find a mentor for Grade 7 ICSE Math & Science?",
    "Can my child get a mentor?",
    "do you have mentors for grade 9 cbse math",
])
def test_published_mentor_requests_are_still_confirmed(engine, request_text):
    assert _ask(engine, str(uuid.uuid4()), request_text).startswith("Yes,")


@pytest.mark.parametrize("model_reply", [
    "No, WeMentors does not offer that.",
    "That is not something I can confirm.",
])
def test_a_model_refusal_is_never_turned_into_yes(engine, monkeypatch, model_reply):
    class Refusing(llm.LLMProvider):
        name = "fake"

        def generate(self, system_prompt, messages):
            return model_reply

    monkeypatch.setattr(config, "LLM_ENABLED", True)
    llm.set_provider_for_testing(Refusing())
    try:
        # In scope for the allowlist, so the model's own answer is shown.
        reply = _ask(engine, str(uuid.uuid4()), "Can my child get a mentor?")
    finally:
        llm.reset_provider_cache()
    assert reply.startswith(model_reply)


def test_only_the_offending_line_changes_and_llm_output_is_covered():
    text = "## Board prep\n\n- Weekly practice.\n- We guarantee full marks.\n- Doubt clinics."
    expected = f"## Board prep\n\n- Weekly practice.\n- {llm.NO_GUARANTEE_SENTENCE}\n- Doubt clinics."
    assert llm.sanitize_llm_output(text) == expected


@pytest.mark.parametrize("question", [
    "do you offer online french classes",
    "do you teach german online",
    "are online classes free",
    "do you offer SAT classes online",
    "can adults learn maths here",
    "My name is Priya, do you offer online french classes",
])
def test_no_bare_yes_when_the_question_mentions_something_unknown(engine, question):
    reply = _ask(engine, str(uuid.uuid4()), question)
    assert not reply.lower().lstrip("*").startswith("yes")
    assert "Yes," not in reply[:60]


@pytest.mark.parametrize("question", [
    "do you offer online classes?",
    "are class timings flexible",
    "do you use google meet",
    "can i join from saudi arabia",
    "My name is Priya, do you offer online classes",
])
def test_ordinary_questions_get_no_caveat(engine, question):
    assert scope.CAVEAT not in _ask(engine, str(uuid.uuid4()), question)


def test_joining_alone_does_not_trigger_the_international_answer(engine):
    reply = _ask(engine, str(uuid.uuid4()), "can my son with autism join")
    assert "Anyone globally" not in reply
