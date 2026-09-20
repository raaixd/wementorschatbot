# WeMentors Chatbot — Production Readiness Audit (v1.0)
**Date:** September 20, 2026  
**Auditor:** AI Engineering & Systems Audit  
**Target Repository:** WeMentors AI Chatbot Experiment (`wementorschatbot`)

---

## 1. Executive Summary

A comprehensive architectural and code audit of the WeMentors AI Chatbot was conducted prior to final production hardening. The system is functionally complete and demonstrates rigorous defensive engineering: deterministic knowledge-base fallbacks, strict prompt injection deflection, structured conversation state management, and clear demo capability boundaries.

This audit details the current architecture, existing safeguards, test inventory, observed weaknesses, production risks, and actionable recommendations.

---

## 2. Current Architecture

The system is organized into a decoupled, layered architecture:

```
[ Visitor / Client Browser (index.html) ]
                   │
                   ▼  (HTTPS / REST)
       [ FastAPI Application (app/main.py) ]
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[ Rate Limiter (ratelimit.py) ] [ SQLite Storage (database.py) ]
    │
    ▼
[ Conversation Engine (conversation.py) ]
    ├── 1. Sanitization & Length Gating (max 2000 chars)
    ├── 2. Intent Detection & Routing (greetings, demo, subjects, cancel, etc.)
    ├── 3. Reference Resolution & Deictic Context ("it", "the first one")
    ├── 4. Hybrid Lexical Retriever (retrieval.py - TF-IDF + Keywords)
    └── 5. Confidence Gating & Grounding Check (threshold: 0.12)
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[ Deterministic KB Compose ]  [ LLM Provider Layer (llm.py) ]
(Zero-network, pure verified text) ├── Primary: Google Gemini (gemini-3.5-flash-lite)
                                   ├── Fallback: Groq (qwen/qwen3.8-27b / llama-3.3-70b)
                                   └── Sanitization & Anti-Hallucination Gate
```

### Key Components:
- **Frontend (`index.html`):** Single-page application with dark glassmorphism chat panel, CSS tokens, markdown parser, request deduplication, typing indicator, and reduced-motion support.
- **Backend API (`app/main.py`):** FastAPI 2.1.0 application serving `/health`, `/chat`, `/chat/clear`, `/feedback`, and `/admin/analytics`.
- **Knowledge Base (`wementors_kb.json`):** 40 verified structured FAQ entries containing questions, answers, alternative phrasings, keywords, and follow-up suggestion chips.
- **Persistence (`app/database.py`):** SQLite database (`chatbot.db`) tracking sessions, messages, feedback, error logs, and lead state with automated PII masking.
- **Conversation State (`app/conversation.py`):** Intent classification, contextual query rewriting, memory management, and anti-contamination guards.
- **Provider Abstraction (`app/llm.py`):** Provider interfaces for Gemini, Groq, Anthropic, and NullProvider with output sanitization (`sanitize_llm_output`).

---

## 3. Existing Safeguards

1. **Anti-Hallucination & Grounding:**
   - Strict retrieval confidence gating (scores < 0.12 trigger honest fallback rather than guessing).
   - Knowledge-base facts are strictly separated between `verified` and `unverified`.
   - `sanitize_llm_output()` strips false completion claims (e.g., claiming to have booked, registered, or scheduled).
2. **Demo Transaction Boundary:**
   - `DEMO_TRANSACTION_ENABLED = False` strictly enforces that the bot is an informational guide, not an active booking agent.
   - Any query attempting to confirm or submit bookings directs the user to the website's "Book Free Demo" button.
3. **Conversational Disambiguation:**
   - Distinct classification separating names ("My name is Raaid") from conversational acknowledgements ("nevermind", "got it", "hmm", "nvm").
   - Disambiguation for "practice": Academic practice resolves to curriculum homework/practice sessions, whereas Confident Speaker practice resolves to guided speaking/public speaking.
4. **Security & Data Protection:**
   - PII masking (`_redact_pii`) in SQLite logs for phone numbers and email addresses.
   - Input length capping at 2,000 characters.
   - Session ID format validation stopping path traversal or SQL injection patterns.
   - Client-level in-memory sliding-window rate limiting.
   - Frontend HTML sanitization (`escapeHtml`) preventing DOM XSS.

---

## 4. Existing Test Inventory

The repository contains 19 test modules and 1 live API validation script:
- `tests/test_core_pipeline.py` (75+ assertions)
- `tests/test_intent_generalization_regression.py` (332 regression checks)
- `tests/test_conversational_ux_and_format.py` (43 checks)
- `tests/test_all_user_scenarios.py` (55 checks)
- `tests/test_context_and_ownership.py` (50 checks)
- `tests/test_conversational_behavior_and_privacy.py` (34 checks)
- `tests/test_conversational_behavior_comprehensive.py` (36 checks)
- `tests/test_conversational_cancellation_and_names.py` (43 checks)
- `tests/test_conversational_intelligence.py` (26 checks)
- `tests/test_deep_conversational_routing.py` (90 checks)
- `tests/test_frontend_and_security.py` (30 checks)
- `tests/test_grade_1_2_and_international.py` (36 checks)
- `tests/test_lead_state_and_confirmations.py` (27 checks)
- `tests/test_llm_prompting.py` (75 checks)
- `tests/test_natural_inference_and_memory.py` (59 checks)
- `tests/test_natural_language_and_false_booking.py` (82 checks)
- `tests/test_no_api_mode.py` (35 checks)
- `tests/test_semantic_context_resolution.py` (54 checks)
- `tests/run_50_validation_tests.py` (50 end-to-end live API tests)

**Total Assertions Across Test Base:** 1,146+ automated checks.

---

## 5. Existing Weaknesses & Gaps Identified

1. **Provider Fallback Unwired at Runtime:**
   - In `app/llm.py`, `generate_answer()` called only the configured active provider. If the active provider failed or timed out, it immediately reverted to deterministic KB responses without attempting a secondary LLM fallback (e.g. Groq).
2. **Groq Model Compatibility:**
   - In `.env`, `GROQ_MODEL=llama-3.3-70b-versatile` returned `404 Not Found` for the active account key. Active models on this key are `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, and `openai/gpt-oss-20b`.
3. **Lack of Structured Request Observability:**
   - Request timing in `app/main.py` logged informal string lines (`[TIMING] duration_ms=...`) rather than structured JSON events recording request ID, intent, retrieval latency, LLM latency, fallback status, and validation result.
4. **Absence of a Formal Benchmark Dataset:**
   - While unit test regressions were thorough, there was no standardized 100+ case evaluation dataset with automated scoring for Groundedness, Answer Correctness, and Hallucination Rates across multi-turn dialogues.
5. **Standalone Test Script Invocations:**
   - Several test files had top-level `sys.exit(0)` calls that halted `pytest` test collection when run through the standard pytest harness.

---

## 6. Potential Production Risks

| Risk | Severity | Mitigation Strategy |
| :--- | :---: | :--- |
| **Primary Provider Rate Limits (Gemini 429)** | Medium | Bounded automatic failover to Groq with zero retries, then graceful degradation to deterministic KB. |
| **Serverless Ephemeral Disks (Vercel SQLite)** | Low | Ensure `/tmp/chatbot.db` is handled cleanly, with no crashes if restarted with a fresh DB. |
| **In-Memory Rate Limiting in Multi-Worker Mode** | Low | Document that in-memory rate limiting applies per process; recommend Redis for distributed horizontally-scaled clusters. |
| **User Prompt Injection & System Prompt Leakage** | Low | Multi-layer defense: system prompt boundaries, context framing, and post-generation sanitization. |

---

## 7. Recommended Changes

1. **Implement Bounded Provider Fallback Chain (`Gemini → Groq → Safe KB`):**
   - Update `app/llm.py` so that a failure in the primary provider automatically attempts the secondary fallback provider once with bounded timeout.
2. **Add Structured JSON Observability:**
   - Emit `chat_request_completed` structured JSON events per request containing detailed timing and metadata.
3. **Build 120+ Case RAG Benchmark Suite:**
   - Implement `evaluation/questions/dataset.json` covering Categories A through J with automated scoring.
4. **Build 20-Failure-Path Production Reliability Test Suite:**
   - Test empty strings, whitespace, malformed JSON, excessive input length, provider crashes, and duplicate requests.
5. **Document Knowledge Gaps & Performance:**
   - Produce `docs/KNOWLEDGE_GAPS.md` and `docs/PERFORMANCE.md` based on measured empirical evaluation.

---

## 8. Changes That Should NOT Be Made

- **DO NOT add a Vector Database (Pinecone, Chroma, Qdrant):** The knowledge base contains 40 curated entries. Lexical TF-IDF + keyword scoring runs in < 2ms with zero infrastructure overhead. Neural embedding databases would add unnecessary latency and cost.
- **DO NOT implement Autonomous Booking Agents:** The chatbot must never take transactional actions or submit mock leads without verified CRM integrations.
- **DO NOT add Multi-Agent Frameworks (CrewAI, LangGraph, AutoGen):** Single-stage retrieval + single LLM call is predictable, fast, and testable. Multi-agent orchestration would multiply latency and failure modes.
- **DO NOT refactor working conversational routing rules:** The regex and semantic disambiguation pipelines in `app/conversation.py` have 1,000+ passing regression checks; preserve all existing behavior.
