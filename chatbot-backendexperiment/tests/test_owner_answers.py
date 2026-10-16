"""Facts confirmed by the WeMentors team (2026-10-16), plus embedding freshness."""
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app import database
from app.conversation import ConversationEngine
from app.knowledge import load_entries

REPO_ROOT = BACKEND_ROOT.parent


@pytest.fixture(scope="module")
def engine():
    os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test_owner_answers.db")
    database.init_db()
    return ConversationEngine(load_entries())


def _reply(engine, query):
    return engine.handle_message(f"sess_owner_{abs(hash(query))}", query).reply.lower()


@pytest.mark.parametrize("query", ["Do you offer refunds?", "What is your refund policy?", "Can I get my money back?"])
def test_refund_is_case_by_case_and_never_promised(engine, query):
    reply = _reply(engine, query)
    assert "case by case" in reply or "individual situation" in reply
    assert "contact" in reply
    for promise in ("full refund", "100%", "within 7 days", "within 30 days", "guarantee a refund"):
        assert promise not in reply


@pytest.mark.parametrize("query", ["What if my child misses a class?", "Can a missed class be rescheduled?"])
def test_rescheduling_depends_on_genuine_reason_and_mentors(engine, query):
    reply = _reply(engine, query)
    assert "genuine reason" in reply
    assert "mentors" in reply
    assert "specific topic" in reply


def test_demo_booking_says_team_will_reach_out(engine):
    reply = _reply(engine, "Is a demo class available?")
    assert "reach out" in reply
    # Notice period / weekend availability are NOT confirmed, so must not be invented.
    for invented in ("24 hours", "same day", "weekend", "sunday", "saturday"):
        assert invented not in reply


def test_grades_11_12_not_offered_and_no_plans(engine):
    reply = _reply(engine, "Do you teach class 12?")
    assert "does not currently offer" in reply
    assert "no plans" in reply


def test_fees_stay_unpublished(engine):
    reply = _reply(engine, "How much are the fees?")
    assert not re.search(r"(₹|\brs\.?\s*\d|\binr\s*\d)", reply)
    assert "contact" in reply


def test_kb_embeddings_match_knowledge_base():
    """Stale embeddings silently disable semantic retrieval. If this fails, run
    `python scripts/generate_kb_embeddings.py` (needs GEMINI_API_KEY)."""
    kb = REPO_ROOT / "knowledge" / "wementors_kb.json"
    emb = json.loads((REPO_ROOT / "knowledge" / "kb_embeddings.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(kb.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert emb["metadata"]["kb_content_hash"] == actual, "kb_embeddings.json is stale; regenerate it"
    ids = {e["id"] for e in json.loads(kb.read_text(encoding="utf-8"))["entries"]}
    assert set(emb["embeddings"]) == ids
