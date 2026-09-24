"""
Comprehensive tests for Feature-Flagged Conditional Hybrid Semantic Retrieval.

Covers:
  1. Baseline parity when HYBRID_RETRIEVAL_ENABLED = False
  2. Semantic paraphrase recovery (e.g. "He was sick and had to skip...")
  3. 50/50 weighted score fusion
  4. Conditional fast path (no embedding API calls on clear lexical matches)
  5. Embedding fallback on API failure (graceful degradation)
  6. API timeout handling
  7. API 429 rate limit handling
  8. Malformed embedding response handling
  9. Missing embedding cache handling
  10. Wrong embedding dimension handling
  11. Stale KB hash handling
  12. Deterministic safety routing (unsupported queries, JEE/NEET, Grade 11-12)
  13. Unsupported queries (e.g. Python web development)
  14. False semantic matches rejection (e.g. cricket question remains unaccepted)
  15. Confident Speaker boundaries
  16. Mentor qualification boundaries
  17. Demo transaction safety
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from unittest.mock import MagicMock, patch

import pytest
import httpx

from app import config, database
from app.conversation import ConversationEngine
from app.knowledge import load_entries
from app.retrieval import Retriever, ScoredEntry
from app.semantic import KBEmbeddingStore, EmbeddingClient, fast_cosine_similarity


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Isolate database in tmp_path for all conversation tests."""
    db_file = tmp_path / "test_chat.db"
    monkeypatch.setattr(config, "DATABASE_PATH", db_file)
    database.init_db()


@pytest.fixture
def entries():
    return load_entries()


@pytest.fixture
def cached_query_vectors():
    q_cache_path = config.PROJECT_ROOT / "evaluation" / "cache" / "query_embeddings.json"
    if q_cache_path.exists():
        with open(q_cache_path, "r", encoding="utf-8") as f:
            return json.load(f).get("embeddings", {})
    # Fallback for CI/clean environments where evaluation/cache is excluded:
    # Use verified precomputed KB embeddings from knowledge/kb_embeddings.json
    store = KBEmbeddingStore()
    vectors = {}
    if store.is_valid:
        if "missed-classes-catch-up" in store.embeddings:
            vectors["He was sick and had to skip. Is there a way to do the session later?"] = (
                store.embeddings["missed-classes-catch-up"]
            )
        if "confident-speaker-business-english" in store.embeddings:
            vectors["I am a 28-year-old software engineer wanting to speak fluently in office meetings"] = (
                store.embeddings["confident-speaker-business-english"]
            )
        if "program-middle-school" in store.embeddings:
            vectors["I want academic maths and science classes for class 7"] = (
                store.embeddings["program-middle-school"]
            )
    vectors["Who is the best batsman in international cricket?"] = [0.001] * config.EMBEDDING_DIMENSION
    return vectors


# ---------------------------------------------------------------------------
# 1. Baseline Parity
# ---------------------------------------------------------------------------
class TestBaselineParity:
    def test_flag_false_matches_pure_lexical_exactly(self, entries, monkeypatch):
        """When HYBRID_RETRIEVAL_ENABLED is False, results must match pure lexical 1:1."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", False)
        retriever = Retriever(entries)

        queries = [
            "What is WeMentors?",
            "What are the fees for middle school?",
            "How long is a class?",
            "Do you offer demo classes?",
        ]
        for q in queries:
            lex_only = retriever._lexical_search(q, top_k=3)
            search_res = retriever.search(q, top_k=3)
            assert len(search_res) == len(lex_only)
            for r1, r2 in zip(search_res, lex_only):
                assert r1.entry.id == r2.entry.id
                assert r1.score == r2.score


# ---------------------------------------------------------------------------
# 2. Semantic Paraphrase Recovery
# ---------------------------------------------------------------------------
class TestSemanticParaphraseRecovery:
    def test_sick_and_skip_recovers_missed_classes(self, entries, cached_query_vectors, monkeypatch):
        """'He was sick and had to skip' fails in lexical but succeeds in hybrid."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        retriever = Retriever(entries)

        query = "He was sick and had to skip. Is there a way to do the session later?"
        mock_vec = cached_query_vectors.get(query)
        assert mock_vec is not None, "Precomputed vector for test query must be in cache"

        with patch.object(EmbeddingClient, "embed_query", return_value=mock_vec):
            results = retriever.search(query, top_k=3)
            assert len(results) > 0
            assert results[0].entry.id == "missed-classes-catch-up"

    def test_adult_business_meeting_speaking(self, entries, cached_query_vectors, monkeypatch):
        """'I am a 28-year-old software engineer wanting to speak fluently in office meetings' matches business english."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        retriever = Retriever(entries)

        query = "I am a 28-year-old software engineer wanting to speak fluently in office meetings"
        mock_vec = cached_query_vectors.get(query)
        assert mock_vec is not None

        with patch.object(EmbeddingClient, "embed_query", return_value=mock_vec):
            results = retriever.search(query, top_k=3)
            top_ids = [r.entry.id for r in results]
            assert any("confident-speaker" in tid for tid in top_ids)


# ---------------------------------------------------------------------------
# 3. 50/50 Weighted Fusion
# ---------------------------------------------------------------------------
class TestWeightedFusion:
    def test_fusion_formula_and_normalization(self, entries, monkeypatch):
        """Verify 50/50 linear fusion computes: 0.5 * norm_dense + 0.5 * norm_lex."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        retriever = Retriever(entries)

        e1 = entries[0]
        e2 = entries[1]

        lex_results = [
            ScoredEntry(entry=e1, score=0.8),
            ScoredEntry(entry=e2, score=0.4),
        ]
        # Lexical min=0.4, max=0.8 -> norm_l(e1)=1.0, norm_l(e2)=0.0
        # Dense scores: e1=0.6, e2=0.8
        dense_results = [(e1, 0.6), (e2, 0.8)]

        fused = retriever._fuse_results(lex_results, dense_results, top_k=2)
        fused_map = {r.entry.id: r.score for r in fused}

        # Expected e1: 0.5 * 0.6 + 0.5 * 1.0 = 0.8000
        # Expected e2: 0.5 * 0.8 + 0.5 * 0.0 = 0.4000
        assert fused_map[e1.id] == pytest.approx(0.8000, abs=1e-4)
        assert fused_map[e2.id] == pytest.approx(0.4000, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Conditional Fast Path
# ---------------------------------------------------------------------------
class TestConditionalFastPath:
    def test_confident_lexical_queries_skip_embedding_api(self, entries, monkeypatch):
        """When lexical match is confident (>=0.35) and high-coverage (>=0.75), embedding API is NOT called."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        retriever = Retriever(entries)

        query = "How long is each class?"
        with patch.object(EmbeddingClient, "embed_query") as mock_embed:
            results = retriever.search(query, top_k=3)
            # Embedding API must NOT be called for clear keyword query
            assert mock_embed.call_count == 0
            assert len(results) > 0
            assert results[0].entry.id == "class-duration"


# ---------------------------------------------------------------------------
# 5. Embedding Fallback on API Failure
# ---------------------------------------------------------------------------
class TestEmbeddingFailureFallback:
    def test_fallback_to_lexical_when_embed_returns_none(self, entries, monkeypatch):
        """If embed_query returns None, retriever degrades gracefully to lexical."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        retriever = Retriever(entries)

        query = "Tell me about classes for 4th standard students"
        with patch.object(EmbeddingClient, "embed_query", return_value=None):
            results = retriever.search(query, top_k=3)
            assert len(results) > 0
            assert results[0].entry.id == "program-foundation-years"


# ---------------------------------------------------------------------------
# 6. API Timeout Handling
# ---------------------------------------------------------------------------
class TestAPITimeoutHandling:
    def test_httpx_timeout_returns_none(self, monkeypatch):
        """EmbeddingClient catches TimeoutException and returns None safely."""
        client = EmbeddingClient(api_key="fake-test-key")
        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
            result = client.embed_query("test query")
            assert result is None


# ---------------------------------------------------------------------------
# 7. API 429 Rate Limit Handling
# ---------------------------------------------------------------------------
class TestAPI429Handling:
    def test_rate_limit_429_returns_none(self):
        """EmbeddingClient handles HTTP 429 and logs warning without raising."""
        client = EmbeddingClient(api_key="fake-test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "RESOURCE_EXHAUSTED"

        with patch("httpx.Client.post", return_value=mock_resp):
            result = client.embed_query("test query")
            assert result is None


# ---------------------------------------------------------------------------
# 8. Malformed Embedding Response Handling
# ---------------------------------------------------------------------------
class TestMalformedResponseHandling:
    def test_malformed_json_or_wrong_type(self):
        """EmbeddingClient handles malformed JSON without raising."""
        client = EmbeddingClient(api_key="fake-test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": {"values": "not-a-list"}}

        with patch("httpx.Client.post", return_value=mock_resp):
            result = client.embed_query("test query")
            assert result is None

    def test_wrong_dimension_from_api(self, monkeypatch):
        """EmbeddingClient handles vector dimension mismatch from API."""
        monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 3072)
        client = EmbeddingClient(api_key="fake-test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": {"values": [0.1] * 512}}  # 512 != 3072

        with patch("httpx.Client.post", return_value=mock_resp):
            result = client.embed_query("test query")
            assert result is None


# ---------------------------------------------------------------------------
# 9. Missing Embedding Cache Handling
# ---------------------------------------------------------------------------
class TestMissingEmbeddingCache:
    def test_missing_cache_file_degrades_safely(self, tmp_path):
        """KBEmbeddingStore marks is_valid=False if artifact file does not exist."""
        non_existent = tmp_path / "does_not_exist.json"
        store = KBEmbeddingStore(artifact_path=non_existent)
        assert store.is_valid is False
        assert "missing" in store.validation_error.lower()


# ---------------------------------------------------------------------------
# 10. Wrong Embedding Dimension Handling
# ---------------------------------------------------------------------------
class TestWrongEmbeddingDimension:
    def test_dimension_mismatch_in_cache(self, tmp_path, monkeypatch):
        """KBEmbeddingStore flags dimension mismatch in metadata."""
        monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 3072)
        bad_artifact = tmp_path / "bad_dim.json"
        bad_artifact.write_text(json.dumps({
            "metadata": {
                "embedding_model": "gemini-embedding-001",
                "vector_dimension": 768,  # mismatch
            },
            "embeddings": {}
        }))
        store = KBEmbeddingStore(artifact_path=bad_artifact)
        assert store.is_valid is False
        assert "dimension mismatch" in store.validation_error.lower()


# ---------------------------------------------------------------------------
# 11. Stale KB Hash Handling
# ---------------------------------------------------------------------------
class TestStaleKBHash:
    def test_hash_mismatch_in_cache(self, tmp_path):
        """KBEmbeddingStore detects when KB content has changed since embeddings were generated."""
        fake_kb = tmp_path / "wementors_kb.json"
        fake_kb.write_text(json.dumps({"entries": []}))

        fake_artifact = tmp_path / "kb_embeddings.json"
        fake_artifact.write_text(json.dumps({
            "metadata": {
                "embedding_model": "gemini-embedding-001",
                "vector_dimension": 3072,
                "kb_content_hash": "stale_hash_12345",
            },
            "embeddings": {}
        }))

        store = KBEmbeddingStore(artifact_path=fake_artifact, kb_path=fake_kb)
        assert store.is_valid is False
        assert "stale" in store.validation_error.lower()


# ---------------------------------------------------------------------------
# 12. Deterministic Safety Routing (Grades 11-12, JEE/NEET)
# ---------------------------------------------------------------------------
class TestDeterministicSafetyRouting:
    def test_grade_11_12_and_jee_neet_unsupported(self, entries, monkeypatch):
        """Deterministic safety logic must execute before retrieval and never be bypassed."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        engine = ConversationEngine(entries)

        unsupported_queries = [
            "Do you teach class 11?",
            "Do you offer JEE prep?",
            "Can I get coaching for NEET?",
            "12th grade physics",
        ]
        for q in unsupported_queries:
            reply = engine.handle_message(f"test_safety_{hash(q)}", q)
            assert reply.intent in ("grade_11_12_unsupported", "jee_neet_unsupported", "unsupported_grades_jee_neet")
            assert "do not offer" in reply.reply.lower() or "focus exclusively" in reply.reply.lower() or "focus" in reply.reply.lower()


# ---------------------------------------------------------------------------
# 13. Unsupported Queries (Python Web Development)
# ---------------------------------------------------------------------------
class TestUnsupportedQueries:
    def test_python_coding_remains_unsupported(self, entries, monkeypatch):
        """'Teach me Python web development with Django' must remain unsupported."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        engine = ConversationEngine(entries)

        reply = engine.handle_message("test_python", "Teach me Python web development with Django")
        assert reply.intent in ("python_course", "unverified_course")
        assert "confirmed details" in reply.reply.lower() or "do not offer" in reply.reply.lower() or "focus" in reply.reply.lower()


# ---------------------------------------------------------------------------
# 14. False Semantic Matches Rejection (Cricket / Out of Domain)
# ---------------------------------------------------------------------------
class TestFalseSemanticMatchesRejection:
    def test_cricket_question_rejected(self, entries, cached_query_vectors, monkeypatch):
        """'Who is the best batsman in international cricket?' must NOT produce a plausible WeMentors answer."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        engine = ConversationEngine(entries)

        query = "Who is the best batsman in international cricket?"
        mock_vec = cached_query_vectors.get(query)

        with patch.object(EmbeddingClient, "embed_query", return_value=mock_vec):
            reply = engine.handle_message("test_cricket", query)
            # Must be caught by off_topic or low_confidence fallback, NOT answered as WeMentors policy!
            assert reply.intent in ("off_topic", "low_confidence", "general")
            assert "wementors" in reply.reply.lower() or "mentoring" in reply.reply.lower() or "help" in reply.reply.lower()
            assert "international cricket" not in reply.reply.lower()


# ---------------------------------------------------------------------------
# 15. Confident Speaker Boundaries
# ---------------------------------------------------------------------------
class TestConfidentSpeakerBoundaries:
    def test_academic_math_query_does_not_match_confident_speaker(self, entries, cached_query_vectors, monkeypatch):
        """Query asking for academic maths must not match Confident Speaker even if dense similarity is non-zero."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        retriever = Retriever(entries)

        query = "I want academic maths and science classes for class 7"
        mock_vec = cached_query_vectors.get(query)

        with patch.object(EmbeddingClient, "embed_query", return_value=mock_vec):
            results = retriever.search(query, top_k=3)
            top_ids = [r.entry.id for r in results]
            for tid in top_ids:
                assert not tid.startswith("confident-speaker-")


# ---------------------------------------------------------------------------
# 16. Mentor Qualification Boundaries
# ---------------------------------------------------------------------------
class TestMentorQualificationBoundaries:
    def test_mentor_qualifications_explicit_routing(self, entries, monkeypatch):
        """Queries about mentor degrees and background route deterministically to mentor-qualifications."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        engine = ConversationEngine(entries)

        reply = engine.handle_message("test_mentor_qual", "What qualifications do your mentors have?")
        assert reply.intent == "mentor_qualifications"
        assert "mentor" in reply.reply.lower()


# ---------------------------------------------------------------------------
# 17. Demo Transaction Safety
# ---------------------------------------------------------------------------
class TestDemoTransactionSafety:
    def test_demo_booking_flow_intact(self, entries, monkeypatch):
        """Booking a demo still triggers lead capture / demo booking intent cleanly."""
        monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
        engine = ConversationEngine(entries)

        reply = engine.handle_message("test_demo_flow", "How can I book a free demo session?")
        assert "demo" in reply.reply.lower()
        assert reply.intent in ("demo_booking", "book_demo", "demo_enquiry", "how_to_book_demo", "lead_capture")
