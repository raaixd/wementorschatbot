# WeMentors AI Chatbot

> Production-oriented conversational RAG system for grounded, context-aware student and parent support.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://wementors.vercel.app)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)](https://docs.pytest.org/)
[![CI](https://github.com/raaixd/wementorschatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/raaixd/wementorschatbot/actions/workflows/ci.yml)

**[Live Demo](https://wementors.vercel.app)** · **[Repository](https://github.com/raaixd/wementorschatbot)**

WeMentors AI is the conversational assistant embedded in the [WeMentors](https://wementors.vercel.app) website, answering student and parent questions about programs, curricula, mentoring, class logistics, and demo bookings. It doesn't generate freely — every factual claim is grounded in a verified knowledge base, and the system is built to say "I don't know, here's who to ask" rather than guess.

Under the hood it's a conversational RAG pipeline: deterministic intent routing, contextual query rewriting for follow-ups ("what about the second one?"), hybrid lexical + semantic retrieval over a curated knowledge base, bounded LLM generation constrained to retrieved evidence, and a response-validation layer that strips hallucinated claims before anything reaches the user.

The project also treats evaluation as a first-class concern — retrieval changes are benchmarked against a fixed query set before they ship, not shipped on the assumption that "embeddings are better."

## At a Glance

| | |
|---|---|
| **Architecture** | Conversational RAG (retrieval → grounded generation → validation) |
| **Backend** | Python / FastAPI |
| **Database** | SQLite |
| **Retrieval** | Hybrid — lexical (TF-IDF) + semantic embeddings, conditionally fused |
| **Embeddings** | `gemini-embedding-001` (3072-dim) |
| **Primary LLM** | Gemini 3.5 Flash-Lite |
| **Fallback LLM** | Groq (Llama / GPT-OSS models) |
| **Final fail-safe** | Deterministic knowledge-base answer (no LLM call) |
| **Knowledge base** | 52 verified entries |
| **Deployment** | Vercel (serverless) |
| **CI** | GitHub Actions — pytest on every push/PR to `main` |

## Key Features

### Conversational Intelligence
- Multi-turn state read from persistent session history (correct across restarts/workers, not an in-process dict)
- Contextual query rewriting for follow-ups
- Pronoun and reference resolution ("its fees", "that one")
- Ordinal references — "the first one" / "the second program"
- Topic switching without losing prior context
- Deterministic intent routing for greetings, thanks, goodbyes, and demo requests

### Retrieval & RAG
- TF-IDF lexical retrieval with keyword-overlap boosting
- Token-coverage gating to reject low-evidence matches
- Conditional semantic embedding retrieval (only invoked when lexical confidence is weak)
- Weighted lexical–semantic score fusion
- Confidence thresholds with a safe "I'm not sure" fallback
- Category- and topic-specific negative gates to prevent cross-program false matches

### Reliability & Safety
- Grounding: generation is bounded to retrieved, verified KB content
- Honest fallback on unsupported questions instead of invented answers
- Prompt-injection detection and deflection
- Hard demo-transaction boundary — no fake bookings or leads
- LLM output sanitization (strips false completion claims, raw HTML)
- Client-side request deduplication and in-flight request cancellation
- Per-IP rate limiting
- Session isolation via validated session tokens

### Production Engineering
- LLM provider fallback chain (primary → secondary → deterministic KB)
- 25 automated regression/test suites
- Offline retrieval benchmarking with measured deltas, not assumptions
- GitHub Actions CI on every push and PR
- Vercel serverless deployment
- Production smoke test suite

## Architecture

```
                              User
                               │
                               ▼
                    Request Validation
                   (length, schema, rate limit)
                               │
                               ▼
                Intent + Context Resolution
             (greeting / demo / follow-up / FAQ)
                               │
                               ▼
                 Contextual Query Rewriting
                               │
                               ▼
                          Retrieval
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
         Lexical Retrieval           Semantic Retrieval
         (TF-IDF, instant)        (gemini-embedding-001,
                                    conditional only)
                 └─────────────┬─────────────┘
                               ▼
              Hybrid Score Fusion / Confidence Gate
                               │
                               ▼
                       Verified Evidence
                               │
                               ▼
                   Bounded LLM Generation
              (Gemini → Groq fallback → KB template)
                               │
                               ▼
              Response Validation / Safety Gates
                               │
                               ▼
                              User
```

Each stage is a plain function call, not an agent — the system is a single deterministic pipeline, which keeps it fast, testable, and easy to reason about.

## Hybrid Retrieval

Earlier versions of this system were purely lexical. The current implementation is a **conditional hybrid retriever**:

1. Every query runs through TF-IDF lexical retrieval first — sub-10ms, zero external calls.
2. If the top lexical match is confident *and* has high keyword coverage, that result is returned immediately (the fast path — no embedding call).
3. Weaker or ambiguous matches trigger semantic retrieval via `gemini-embedding-001`.
4. The same domain/safety gates used for lexical retrieval (fee gating, program-boundary gating, etc.) are applied to semantic candidates too.
5. Lexical and semantic scores are min-max normalized and linearly fused (50/50 weighting).
6. A final acceptance threshold decides whether the fused result is returned at all.
7. If the embedding API times out, rate-limits, or returns malformed data, the system falls back to the lexical result — semantic retrieval never blocks or breaks a reply.

**Why this shape:** lexical retrieval is fast, interpretable, and reliable for direct FAQ-style questions, which make up most real traffic. Semantic retrieval earns its latency and API cost only on the harder slice — paraphrased or conversational queries with little literal keyword overlap. The conditional design means the embedding API is called only when it's actually needed, not on every request.

## Retrieval Evaluation

Hybrid retrieval wasn't adopted on the assumption that embeddings are strictly better — it was benchmarked against the existing lexical baseline on a **156-query evaluation set** before being enabled.

| Metric | Lexical Baseline | Hybrid | Change |
|---|---:|---:|---:|
| Recall@1 | 72.00% | 76.00% | +4.00 pp |
| Recall@3 | 84.00% | 84.00% | — |
| MRR | 0.7733 | 0.8000 | +0.0267 |
| NDCG@3 | 0.7905 | 0.8105 | +0.0200 |
| Answer correctness | 78.85% | 82.69% | +3.84 pp |
| Groundedness | 96.79% | 97.44% | +0.65 pp |
| Hallucination | 3.21% | 2.56% | −0.65 pp |
| Context resolution | 91.03% | 93.59% | +2.56 pp |

Lexical retrieval was kept as the baseline and fast path rather than replaced. Dense retrieval was tested standalone, then swept across fusion strategies (weighted linear, reciprocal rank fusion) and gating configurations; weighted fusion at the fast-path thresholds above performed best on this set. The clearest gains were in top-1 retrieval accuracy and answer correctness, driven by paraphrased queries the lexical matcher had no shared vocabulary with. In this same run, the fast path resolved the majority of queries without ever calling the embedding API — semantic retrieval was invoked only for the harder minority.

## Example: Semantic Recovery

> "He was sick and had to skip. Is there a way to do the session later?"

The lexical retriever finds almost no shared vocabulary between this query and the knowledge base's missed-class/catch-up entry — "sick," "skip," and "later" don't appear in the entry's question or phrasings. Semantic retrieval recovers the correct entry by matching on meaning rather than keywords. This is one of the regression cases in the test suite, covering the exact scenario the hybrid path exists for.

## LLM & Generation Architecture

Generation is bounded, not open-ended: the model only receives retrieved, verified knowledge-base content as context and is not free to answer from general world knowledge about WeMentors.

- **Provider abstraction** — a common interface over Gemini, Groq, and Anthropic providers, plus a null provider for template-only mode
- **Primary:** Gemini (`gemini-3.5-flash-lite`)
- **Fallback:** Groq, attempted automatically if the primary provider fails, times out, or is rate-limited
- **Final fail-safe:** a deterministic, template-based answer built directly from the KB entry — no network call, works fully offline
- API credentials are read from environment variables server-side and never returned to the client
- Provider selection and any misconfiguration are surfaced through `/health`, not hidden behind a silent fallback

## Safety & Reliability

The chatbot answers questions — it does not take actions. `DEMO_TRANSACTION_ENABLED = False` is a hard constant: the system cannot submit a booking, create a lead, or claim to have completed a transaction on the user's behalf. It always directs booking intent to the site's official demo/contact flow. This is a capability boundary by design, not a missing feature.

| Control | Implementation |
|---|---|
| Prompt injection | Regex-based deflection + generation bounded to retrieved context |
| Secrets | Environment variables only, never logged or echoed |
| Output safety | LLM output sanitization strips false action claims and raw HTML |
| Demo safety | No transactional capability; `DEMO_TRANSACTION_ENABLED = False` |
| Rate limiting | Per-client sliding-window limiter (in-memory) |
| Session isolation | Session IDs, validated by format before use |
| Input validation | Pydantic schemas, length-capped requests |
| HTML safety | Frontend escapes all rendered text before DOM insertion |

**Known limitation:** session IDs are bearer values — anyone holding a session ID can act as that session. There is no additional authentication layer, which is an accepted trade-off for an anonymous, informational assistant with no PII or transactional capability. See [`docs/SECURITY_REVIEW.md`](docs/SECURITY_REVIEW.md) for the full OWASP-mapped review.

## Testing & Reliability

The guiding rule: **observed failures were converted into regression tests.** The suite isn't a checklist written in advance — most of it exists because something broke once and got locked down.

25 automated test suites cover:
- Retrieval (lexical, semantic, and hybrid fusion)
- Conversation state and contextual follow-ups
- Unsupported/out-of-scope questions
- Prompt injection
- LLM provider failure and fallback
- Demo-transaction safety
- Message ownership and request deduplication
- Frontend and security behavior
- Production smoke testing

GitHub Actions runs the full pytest suite on every push and pull request to `main`.

One example worth calling out as production engineering rather than a bug: the semantic embedding cache is validated against a SHA-256 hash of the knowledge-base file, and that hash needs to match across Windows and Linux. A `.env`/KB file with Windows line endings produced a different hash than the same content on the Linux CI runner, which invalidated the embedding cache on every CI run. The fix normalizes newlines before hashing, and CI has been stable since.

## Project Structure

```
wementorschatbot/
├── chatbot-backendexperiment/
│   ├── app/
│   │   ├── main.py           # FastAPI app, routes, request validation
│   │   ├── conversation.py   # Conversation engine / RAG pipeline
│   │   ├── retrieval.py      # Lexical retrieval + hybrid fusion
│   │   ├── semantic.py       # Embedding client + KB embedding store
│   │   ├── llm.py            # Provider abstraction (Gemini/Groq/Anthropic)
│   │   ├── knowledge.py      # Knowledge base loading
│   │   ├── leads.py          # Demo-transaction boundary
│   │   ├── database.py       # SQLite persistence
│   │   ├── ratelimit.py      # Per-client rate limiter
│   │   └── config.py         # Environment-driven configuration
│   └── tests/                # 25 pytest suites + release/smoke runners
├── knowledge/
│   ├── wementors_kb.json     # Verified knowledge base (52 entries)
│   └── kb_embeddings.json    # Precomputed KB embeddings
├── scripts/
│   └── generate_kb_embeddings.py
├── evaluation/
│   ├── questions/            # Benchmark query sets
│   ├── runners/               # Benchmark + experiment runners
│   └── reports/               # Measured benchmark reports
├── docs/                     # Engineering documentation (see below)
└── .github/workflows/ci.yml  # CI pipeline
```

## Quick Start

```powershell
git clone https://github.com/raaixd/wementorschatbot.git
cd wementorschatbot

cd chatbot-backendexperiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload
```

In another terminal, serve the frontend:

```powershell
cd C:\path\to\wementorschatbot
python -m http.server 3000
```

| | |
|---|---|
| Frontend | http://127.0.0.1:3000 |
| Backend | http://127.0.0.1:8000 |
| Health | http://127.0.0.1:8000/health |
| Docs | http://127.0.0.1:8000/docs |

### Configuration

The chatbot runs fully offline with no keys set — it answers from the knowledge base using deterministic templates. Set `GEMINI_API_KEY` (and optionally `GROQ_API_KEY` for fallback) in `.env` for LLM-generated phrasing. Set `HYBRID_RETRIEVAL_ENABLED=true` to enable semantic retrieval; it defaults to lexical-only. See `.env.example` for the full list of variables.

### Run Tests

```powershell
pytest                                  # full suite
python tests/run_release_test_suite.py  # aggregated release scorecard
python tests/run_50_validation_tests.py # live-API end-to-end checks
```

## Documentation

| Document | Purpose |
|---|---|
| [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) | Full engineering case study |
| [`docs/RAG_EVALUATION.md`](docs/RAG_EVALUATION.md) | Retrieval methodology and benchmark results |
| [`docs/SECURITY_REVIEW.md`](docs/SECURITY_REVIEW.md) | OWASP-mapped security analysis |
| [`docs/PRODUCTION_AUDIT.md`](docs/PRODUCTION_AUDIT.md) | Architecture / codebase audit |
| [`docs/KNOWLEDGE_GAPS.md`](docs/KNOWLEDGE_GAPS.md) | Verified knowledge gaps |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Performance measurements |
| [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) | Release validation gates |

## Current Limitations

- Knowledge base is small and curated (52 entries) — coverage, not scale, is the design goal
- Semantic retrieval depends on an external embedding API; it degrades gracefully to lexical-only on failure, but adds latency when invoked
- SQLite is single-instance; horizontal scaling would need a shared database
- Rate limiting is in-memory and per-process — a multi-worker or multi-instance deployment would need a shared store (e.g. Redis)
- No real CRM or booking-system integration — demo requests are guided, not submitted
- Session identity is a bearer token with no additional authentication layer
- The retrieval benchmark set (156 queries) is useful but modest in size for statistical confidence

## Future Work

- Larger, more diverse evaluation datasets
- Human preference evaluation alongside automated metrics
- Structured production observability (request-level tracing, not log lines)
- Real CRM / booking-system integration
- Knowledge-base versioning and an approval workflow for edits
- A shared-state rate limiter and database if traffic outgrows a single instance

## License

This project is a portfolio / engineering case study. See the repository for license details.
