# WeMentors Chatbot — Empirical System Baseline (v1.0)
**Date:** September 20, 2026  
**Environment:** Windows (PowerShell), Python 3.12 (.venv), Node/Browser Static Server

---

## 1. Automated Test Baseline

All test counts and execution outcomes recorded below reflect actual empirical runs executed during the initial system audit.

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Total Test Modules** | **19** | Standalone regression & unit test scripts in `tests/` |
| **Total Checks / Assertions** | **1,146** | Cumulative verified assertions across all test suites |
| **Passing Test Suites** | **19** | 100% pass rate when run in appropriate test modes |
| **Failing Test Suites** | **0** | 0 failures in core logic |
| **Skipped Tests** | **0** | No skipped tests |
| **Live API 50-Check Suite** | **50 / 50 PASSED** | Verified against live local FastAPI uvicorn daemon |
| **Intent Generalization Regression** | **332 / 332 PASSED** | Verified intent classification & response safety |
| **Core Pipeline Regression** | **100+ PASSED** | Verified retrieval, sessions, and sanitization |

---

## 2. Server & Environment Baseline

| Component | Status | Details |
| :--- | :---: | :--- |
| **Backend Startup** | **HEALTHY** | Uvicorn running on `http://127.0.0.1:8000`. Startup time: ~0.8s. Loaded 40 KB entries cleanly. |
| **Frontend Startup** | **HEALTHY** | Static SPA in `index.html` loads cleanly with no syntax errors. |
| **Database Startup** | **HEALTHY** | SQLite initialized at `chatbot-backendexperiment/data/chatbot.db`. All tables created with indexes. |
| **Vercel Serverless Ready** | **HEALTHY** | `api/index.py` configured with path middleware and fallback to `/tmp/chatbot.db`. |

---

## 3. Provider Configuration Baseline

| Provider | Configured Model | API Key Status | Live Call Status | Empirical Observation |
| :--- | :--- | :---: | :---: | :--- |
| **Gemini (Primary)** | `gemini-3.5-flash-lite` | Present | **FUNCTIONAL** | Latency: 950ms–2480ms. Intermittent rate-limiting (429) triggers model candidate fallback. |
| **Groq (Secondary)** | `llama-3.3-70b-versatile` | Present | **MODEL 404** | Configured model not found on account; alternate account model `qwen/qwen3.8-27b` works in ~600ms. |
| **Anthropic (Tertiary)** | `claude-3-5-haiku-latest`| Not Set | **DISABLED** | NullProvider active fallback. |
| **Offline Null Provider** | Deterministic KB | N/A | **FUNCTIONAL** | 100% offline, zero-network fallback responding in < 2ms. |

---

## 4. Retrieval Behavior Baseline

- **Method:** TF-IDF Cosine Similarity + Keyword Overlap Bonus + Token Coverage Gating.
- **Corpus:** 40 entries, 442 tokens, 106 unique terms.
- **Top-K:** 3 entries retrieved per query.
- **Confidence Threshold:** `0.12`.
  - Queries scoring $\ge 0.12$ with verified coverage are accepted.
  - Queries scoring $< 0.12$ trigger honest fallback without hallucination.
- **Measured Retrieval Latency:** **0.4ms – 1.8ms** per query.

---

## 5. Measured End-to-End Latencies

Measured via real HTTP round-trips against local backend using `run_50_validation_tests.py`:

| Query Type | Typical Request | Provider / Route | Latency (ms) |
| :--- | :--- | :--- | :---: |
| **Greeting / Canned** | "Hello!" | Deterministic | **22.4 ms** |
| **Knowledge Fact (Deterministic)** | "What is Foundation Years?" | Deterministic KB | **34.5 ms** |
| **Disambiguated Intent** | "What about practice?" | State Machine / KB | **45.9 ms** |
| **Demo Inquiry Guidance** | "Book me a demo" | Gated Guidance | **25.6 ms** |
| **Cancellation / Name** | "nevermind" / "My name is Raaid" | Intent State | **26.3 ms** |
| **LLM Synthesis (Gemini)** | "Can you tell me about grades 9-10?"| Gemini 3.5 Flash-Lite| **2,480.8 ms** |
| **LLM Follow-up (Gemini)** | "What subjects does it have?" | Gemini 3.5 Flash-Lite| **1,197.8 ms** |
| **Groq Standalone Test** | "Say hello in 3 words" | `qwen/qwen3.8-27b` | **612.4 ms** |
| **Request Deduplication** | Repeated `request_id` within 60s | Cache Hit | **3.8 ms** |

---

## 6. Unmeasured / Production Items

- **Live Production Domain Latency (Vercel/Render):** **NOT MEASURED** (Local environment; remote deployment URL not yet invoked for load benchmarks).
- **Multi-tenant Horizontal Concurrency:** **NOT MEASURED** (SQLite database and sliding-window rate limiter are single-instance).
