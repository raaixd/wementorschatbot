# WeMentors Chatbot

A RAG-powered chatbot embedded in the WeMentors website. A FastAPI backend
retrieves relevant answers from a structured, verified WeMentors knowledge
base, resolves follow-up questions using real conversation history, and
replies in a consistent, warm, and honest tone — without ever inventing
fees, policies, or promises the knowledge base doesn't support.

## Features

- Premium, dark glassmorphism chat widget (frosted-glass surfaces, subtle
  purple/indigo/blue gradients, a liquid-glass logo mark in the launcher
  and header) that opens over the existing light WeMentors site without
  changing it
- Smooth, restrained micro-interactions throughout: launcher hover/press,
  panel open/close, message fade-and-rise, typing indicator, suggestion
  chip hover/press, send-button feedback, focus states — all disabled
  under `prefers-reduced-motion`
- Structured JSON knowledge base (`knowledge/wementors_kb.json`) with
  categories, alternative phrasings, keywords, and follow-up suggestions
- Hybrid lexical retrieval (TF-IDF + keyword overlap) with a confidence
  threshold and safe fallback — no external ML service required
- Responses formatted to fit the question: short answers for simple
  facts, bullet lists for "what programs/subjects", numbered steps for
  "how do I apply", and a side-by-side layout for "compare X and Y"
- Intent detection: greetings, thanks, goodbyes, demo-booking requests,
  program comparisons, frustrated/confused users, prompt-injection
  deflection, and off-topic handling — when a provider is active, an
  unmatched question that is genuinely general or conversational (not a
  WeMentors-specific fact) gets a brief, honest answer from the model
  instead of a canned redirect; with no provider, or when the model
  declines, the canned redirect is used
- Context-aware follow-up handling: "tell me more", "what about the first
  one?", "the second program", pronoun references like "it"/"that" —
  without misfiring on ordinary phrases like "one-on-one classes"
- Server-side session + conversation history stored in SQLite, so context
  survives page refreshes within the same browser session and backend
  restarts
- Optional LLM answer rewriting via **Groq** or **Anthropic**, fully
  **off by default** — with no API key the chatbot works completely
  offline using deterministic answers built from verified knowledge-base
  text, and makes no outbound network call at all
- Rate limiting, input validation, consistent JSON responses, and no
  leaked stack traces / internal paths in visitor-facing errors
- `/health`, `/docs` (interactive API docs, built into FastAPI), and a
  key-protected `/admin/analytics` endpoint for operability

## Project Structure

```text
wementorschatbotdev/
├── index.html                       # website + chatbot widget (HTML/CSS/JS)
├── knowledge/
│   └── wementors_kb.json            # structured knowledge base (source of truth)
├── chatbot-backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, routes, error handling
│   │   ├── config.py                # environment-variable configuration
│   │   ├── database.py              # SQLite: sessions, messages, feedback, error logs
│   │   ├── knowledge.py             # knowledge base loader/parser
│   │   ├── retrieval.py             # TF-IDF + keyword hybrid retriever
│   │   ├── conversation.py          # intent detection, reference resolution, RAG orchestration
│   │   ├── personality.py           # persona spec + canned response templates
│   │   ├── ratelimit.py             # per-client rate limiting (no web-framework dep, unit-tested)
│   │   └── llm.py                   # prompt construction + Groq/Anthropic providers + output validation
│   ├── tests/
│   │   └── test_core_pipeline.py    # stdlib-only tests for the RAG pipeline
│   ├── data/                        # SQLite database file (git-ignored, created on first run)
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

## How It Works (RAG Pipeline)

```
user message
  → validation (length, empty-message checks)
  → intent detection (greeting / thanks / goodbye / demo booking /
      comparison / frustrated / confused / off-topic / faq / injection)
  → reference resolution (uses the last turn stored in SQLite:
      "the first one" / "the second program" → ordinal lookup into the
      previously shown list; "it" / "that" / "tell me more" → falls back
      to the previous answer only when direct retrieval is weak)
  → knowledge-base retrieval (TF-IDF + keyword overlap + a token-coverage
      gate, top-k=3 — see "Retrieval precision" below)
  → confidence threshold check: no match at all → if a provider is
      active, ask it for a brief general answer (never a WeMentors fact
      it wasn't given); below the confidence threshold → honest fallback,
      never a guess
  → answer generation (template composition by default, rendering
      "bullets"/"steps" entries as real lists; optional LLM rewrite when a
      provider is configured, using ONLY the retrieved text as source
      material, then validated before use)
  → response returned to the frontend, and the turn is logged to SQLite
```

Only the single best-matching entry (or, for an explicit multi-part
reference like the programs list or a two-program comparison) is ever
sent back — the full knowledge base is never dumped into one answer.

### Why lexical retrieval instead of embeddings?

The knowledge base is a few dozen short FAQ entries. A pure-Python TF-IDF +
keyword-overlap retriever (see `retrieval.py`) is fast, has zero extra
dependencies, needs no vector database, and is easy for a student
developer to read and extend. `retrieval.py` isolates the
"turn text into a ranked list of candidate entries" step behind a single
`Retriever.search()` call — if the knowledge base grows much larger or
needs true semantic matching, that method is the only place that would
need to change to call an embeddings API instead.

### Retrieval precision: the token-coverage gate

Plain TF-IDF cosine similarity has a real failure mode on a small corpus:
one rare shared word can make an unrelated multi-word question look like a
confident match. For example, "Do you offer scholarships or financial
aid?" (a topic genuinely absent from the knowledge base) shared only the
word "offer" with the programs-overview entry's phrasing "what courses do
you offer" — but because "offer" is a rare (high-IDF) word, the cosine
score alone came out above the confidence threshold, and the chatbot would
have confidently returned the programs list instead of honestly saying it
didn't know. `Retriever.search()` now scales the score by what fraction of
the query's words actually appear anywhere in the candidate entry (for
queries of 3+ words); that single case drops from a false-confident 0.33
to 0.08, correctly falling back to the honest "I'm not certain about that"
response, while every genuine multi-word match in the test suite is
unaffected (verified in `test_core_pipeline.py`).

### LLM providers (all optional)

Three modes are supported. **The default is `none`** — the chatbot is
fully functional with no API key and makes no external request.

| Provider | Env vars required | Package | Default model |
|---|---|---|---|
| `none` (default) | *(none)* | *(none)* | n/a — deterministic templates |
| `gemini` | `GEMINI_API_KEY` | `pip install openai` | `gemini-3.5-flash-lite` |
| `groq` | `GROQ_API_KEY` | `pip install openai` | `llama-3.3-70b-versatile` |
| `anthropic` | `ANTHROPIC_API_KEY` | `pip install anthropic` | `claude-3-5-haiku-latest` |

Gemini and Groq are reached through their **OpenAI-compatible** endpoints
(`GEMINI_BASE_URL`, default `https://generativelanguage.googleapis.com/v1beta/openai/`;
`GROQ_BASE_URL`, default `https://api.groq.com/openai/v1`), which is why
the `openai` package is what you install.

**How selection works**

1. If `LLM_PROVIDER` is set, that provider is used — **but only if its API
   key is present**. Requesting a provider without its key is treated as a
   misconfiguration: the app logs a warning at startup, falls back to
   `none`, and `/health` reports `llm_enabled: false` with an
   `llm_config_warning`. It never claims an LLM is active when it isn't.
2. If `LLM_PROVIDER` is unset, the provider is auto-detected:
   `GEMINI_API_KEY` first, then `GROQ_API_KEY`, then `ANTHROPIC_API_KEY`.
3. If no key is set, the provider is `none`.
4. An unrecognised `LLM_PROVIDER` value fails safe to `none` with a
   warning naming the valid options.

An empty or whitespace-only key counts as absent. `LLM_PROVIDER=none`
disables the LLM even when keys are present.

**No API call happens when the provider is disabled.** With `none`
active, `NullProvider` declines every request before any client is built,
and no provider SDK is even imported. This is enforced by
`tests/test_no_api_mode.py`, which blocks outbound sockets for the whole
run and asserts that zero connection attempts were made.

**Running with no API key**

```powershell
# nothing to configure - this is the default
python -m uvicorn app.main:app --reload
```

**Running with Groq**

```powershell
pip install openai
# in chatbot-backend\.env:
#   GROQ_API_KEY=<your key>
#   GROQ_MODEL=openai/gpt-oss-120b
python -m uvicorn app.main:app --reload
```

**Running with Anthropic**

```powershell
pip install anthropic
# in chatbot-backend\.env:
#   ANTHROPIC_API_KEY=<your key>
#   ANTHROPIC_MODEL=claude-3-5-haiku-latest
python -m uvicorn app.main:app --reload
```

**Three kinds of answer, kept separate**

The persona prompt (`personality.py`) instructs the model to distinguish:
1. **WeMentors-specific facts** (programs, fees, schedules, contact,
   eligibility) — stated only from the retrieved context, never invented.
2. **General conversational/educational content** that needs no
   WeMentors-specific fact (e.g. reassurance about choosing a program, a
   plain general-knowledge question) — the model may answer this briefly
   from its own knowledge, but must never present it as WeMentors policy.
3. **A WeMentors-specific fact the context doesn't contain** — stated
   honestly as unknown, with a pointer to the WeMentors team or a demo.

This is why an unmatched question doesn't always get the same canned
"I'm here to help with WeMentors..." redirect: a genuinely off-topic or
inappropriate message still does, but an ordinary conversational or
general question gets a real answer when a provider is active (see
`ConversationEngine._generate_general_answer`). Injection attempts and
demo-booking requests are still classified before this path is ever
reached, so they're unaffected by it.

**What the provider does (and does not) do**

When a provider is active it only *rephrases* the retrieved
knowledge-base text more conversationally. It is given that text as its
only source of facts and the visitor's message explicitly labelled as
data, never instructions. Every response is validated by
`sanitize_llm_output()` before display: HTML stripped, length capped, and
any output that leaks internal instructions or falsely claims a completed
action (e.g. "I've booked your demo") is discarded in favour of the
deterministic answer. Any failure — missing package, bad key, timeout,
HTTP error, empty response — falls back silently to the same
deterministic answer, so the chatbot never appears broken.

## Conversation Memory

Conversation state lives in SQLite (`chatbot-backend/data/chatbot.db`),
keyed by a `session_id` that the frontend generates once per browser tab
(`sessionStorage`, cleared when the tab closes) and sends with every
request. This means:

- Follow-up questions work correctly because the backend — not the
  browser — is the source of truth for "what was just discussed."
- A backend restart doesn't erase conversation context (it's on disk).
- Different browser tabs/visitors never share or contaminate each other's
  context (separate session ids).
- "Clear Chat" both resets the visible widget and deletes that session's
  stored messages server-side.

## Database

SQLite was chosen deliberately over a heavier database: this is a
single-instance, low-write-volume application, and SQLite needs no extra
infrastructure while still giving real persistence across restarts (see
`chatbot-backend/app/database.py` for the schema and rationale in the
module docstring). Tables:

- `sessions` — one row per chat session
- `messages` — every user/assistant turn, with detected intent, matched
  knowledge-base entry id(s), and retrieval confidence (used for
  follow-up resolution and analytics)
- `feedback` — optional thumbs up/down on a specific reply
  (`POST /feedback`)
- `error_logs` — server-side error details, so visitors only ever see a
  safe generic message while the real cause is still recoverable for
  debugging

If this project ever needs multi-instance deployment, the rate limiter
(currently in-memory, see the note in `main.py`) and the SQLite file would
both need to move to shared infrastructure (e.g. Redis + a networked
database) — everything else is already isolated behind small modules so
that swap wouldn't touch retrieval, conversation, or persona logic.

## Requirements

- Python 3.10 or newer (Windows: install from [python.org](https://www.python.org/downloads/) and check "Add python.exe to PATH" during install)
- Git
- A modern web browser

## Local Setup — Windows (PowerShell)

These are the exact commands, in order, for Windows PowerShell. Use two
separate PowerShell windows: **Terminal A runs the backend**, **Terminal B
runs the frontend**. Leave both open while you test.

### 1. Extract the ZIP and open the project

Right-click the downloaded ZIP → **Extract All...**, then open the
extracted folder in PowerShell:

```powershell
cd C:\path\to\wementorschatbotdev
```

### 2. Terminal A — set up and start the backend

```powershell
cd chatbot-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script with an execution-policy error,
run this once (in the same window) and try activating again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Continue in the same (now-activated) terminal:

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload
```

You should see `Uvicorn running on http://127.0.0.1:8000`. Leave this
window open — **this is the backend, at `http://127.0.0.1:8000`.**

### 3. Terminal B — start the frontend

Open a **new** PowerShell window (don't close Terminal A):

```powershell
cd C:\path\to\wementorschatbotdev
python -m http.server 5500
```

Leave this window open too — **this is the frontend, at
`http://127.0.0.1:5500`.**

### 4. Test the backend

In a third PowerShell window (or a browser tab):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

You should get back `status: healthy`. You can also open
`http://127.0.0.1:8000/docs` in a browser for the interactive API
documentation (provided automatically by FastAPI).

### 5. Open the website

Open `http://127.0.0.1:5500` in your browser. The chatbot launcher button
appears in the bottom-right corner.

### 6. Stop both processes

Click into each PowerShell window and press `Ctrl+C`. To leave the Python
virtual environment afterward, run `deactivate` in Terminal A.

### Troubleshooting (Windows)

| Problem | Fix |
|---|---|
| `python` is not recognized | Reinstall Python from python.org with "Add to PATH" checked, then open a new PowerShell window |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in that window, then retry |
| Chatbot shows a connection error in the browser | Confirm Terminal A still shows `Uvicorn running...` and that `chatbot-backend/.env` has `ALLOWED_ORIGINS` including `http://127.0.0.1:5500` |
| `Address already in use` / port 8000 or 5500 busy | Another process is using that port — stop it, or run uvicorn on a different port with `--port 8001` (and update `window.CHATBOT_API_URL` in `index.html` to match) |
| `ModuleNotFoundError: No module named 'fastapi'` | The virtual environment isn't activated, or `pip install -r requirements.txt` wasn't run inside it — re-activate and reinstall |
| Browser console shows a CORS error | Add the exact frontend origin you're using (e.g. `http://127.0.0.1:5500`) to `ALLOWED_ORIGINS` in `chatbot-backend/.env` and restart the backend |

## Local Setup — macOS / Linux

```bash
git clone https://github.com/raaixd/wementorschatbot.git
cd wementorschatbot/chatbot-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app.main:app --reload
```

In a second terminal, from the project root:

```bash
python3 -m http.server 5500
```

Backend: `http://127.0.0.1:8000` (health check: `http://127.0.0.1:8000/health`, docs: `http://127.0.0.1:8000/docs`)
Frontend: `http://127.0.0.1:5500`
Stop either with `Ctrl+C` in its terminal.

## Environment Variables

See `chatbot-backend/.env.example` for the full, commented list. Key ones:

| Variable | Purpose | Default |
|---|---|---|
| `ALLOWED_ORIGINS` | Comma-separated frontend origins allowed by CORS | `http://127.0.0.1:5500,http://localhost:5500` |
| `DATABASE_PATH` | SQLite file location | `chatbot-backend/data/chatbot.db` |
| `KNOWLEDGE_FILE` | Path to the JSON knowledge base | `knowledge/wementors_kb.json` |
| `RATE_LIMIT_MAX_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | Per-IP request limit | 20 requests / 60s |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | Minimum score before the bot admits it doesn't know | `0.12` |
| `LLM_PROVIDER` | Force a provider: `none`, `gemini`, `groq`, `anthropic`. Unset = auto-detect from whichever key is present | *(unset)* |
| `GEMINI_API_KEY` | Enables Google Gemini (needs `pip install openai`). Blank = provider not selectable | *(blank)* |
| `GEMINI_MODEL` | Gemini model id | `gemini-3.5-flash-lite` |
| `GEMINI_BASE_URL` | Gemini's OpenAI-compatible endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `GROQ_API_KEY` | Enables Groq (needs `pip install openai`). Blank = provider not selectable | *(blank)* |
| `GROQ_MODEL` | Groq model id | `llama-3.3-70b-versatile` |
| `GROQ_BASE_URL` | Groq's OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `ANTHROPIC_API_KEY` | Enables Anthropic (needs `pip install anthropic`) | *(blank)* |
| `ANTHROPIC_MODEL` | Anthropic model id | `claude-3-5-haiku-latest` |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | Generation limits | `15` / `600` / `0.3` |
| `LLM_HISTORY_TURNS` | Prior turns replayed so follow-ups resolve | `6` |
| `ADMIN_API_KEY` | Protects `GET /admin/analytics`; blank disables the endpoint | *(blank)* |

Never commit a real `.env` file — it's already listed in `.gitignore`.

## API Endpoints

### `GET /health`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

With no API key configured (the default):

```json
{
  "status": "healthy",
  "service": "wementors-chatbot",
  "knowledge_base_loaded": true,
  "knowledge_base_entries": 27,
  "llm_enabled": false,
  "llm_provider": "none",
  "llm_provider_configured": "none"
}
```

`llm_enabled` reflects the **actually active** provider, not what
configuration asked for. If a provider was requested but could not be
built (missing key, missing package, bad client), `llm_enabled` is
`false`, `llm_provider` is `"none"`, and an `llm_config_warning` field
explains why. Responses are sent as
`application/json; charset=utf-8` so Windows PowerShell decodes
punctuation such as em dashes correctly.

### `POST /chat`

Request:

```json
{"message": "Which grades are supported?", "session_id": "optional-existing-session-id"}
```

Response:

```json
{"reply": "...", "session_id": "the session id to reuse on the next call", "intent": "faq"}
```

### `POST /chat/clear`

Clears stored history for a session. Body: `{"session_id": "..."}`.

### `POST /feedback`

Optional thumbs up/down on a reply: `{"session_id": "...", "message_id": 42, "rating": "up"}`.

### `GET /admin/analytics`

Requires header `X-Admin-Key` matching `ADMIN_API_KEY`. Returns total
sessions, total user messages, top intents, and low-confidence reply
count. Returns 404 (not just 401) when disabled, so the endpoint's
existence isn't revealed.

## Updating the Knowledge Base

Edit `knowledge/wementors_kb.json`. Each entry follows the schema
documented in that file's `_meta.schema` key:

```json
{
  "id": "unique-slug",
  "category": "programs",
  "question": "The canonical question",
  "phrasings": ["alternative ways to ask this"],
  "keywords": ["extra retrieval keywords"],
  "answer": "The verified, direct answer.",
  "follow_ups": ["A related question to suggest next"],
  "confidence": "verified"
}
```

An entry doesn't have to restate a single fact from your team to be
`"verified"` — it can also be a direct logical consequence of facts
already verified elsewhere in the file. For example,
`college-students-eligibility` states that college students aren't
eligible for the academic programs and are welcome in Confident Speaker;
that isn't a new claim, it follows directly from `which-grades-supported`
(academic programs are grade-banded 3-10) and `who-can-use-wementors`
(Confident Speaker already covers "students, professionals, and
homemakers"). If a fact doesn't follow that directly from something
already verified, set `"confidence": "unverified"` for anything not yet confirmed (see the
`fees-and-pricing` entry for an example) — the persona and fallback
wording are built around never stating unverified information as fact.
Restart the backend after editing (or rely on `--reload` in dev) and test
the updated question through the chatbot.

## Testing

### Automated

Two **stdlib-only** test scripts (no `pip install` required to run them):

**`chatbot-backend/tests/test_core_pipeline.py`** — knowledge base, database
layer, retrieval (including the coverage-gate fix above), intent detection,
reference resolution, structured (bullet/numbered) formatting, program
comparison, frustrated/confused handling, session isolation, LLM output
sanitization, and the exact multi-turn scenario from the project spec
("what programs do you offer" → "tell me more about the first one" →
"what are its fees" → "how can I apply" → "what about the second program").

**`chatbot-backend/tests/test_frontend_and_security.py`** — `index.html`
parses cleanly, all required chatbot elements exist, the shared liquid-glass
logo SVG def is defined once and referenced by both the launcher and header,
the close button is actually wired, `prefers-reduced-motion` is present,
inline JS syntax is checked with Node if available, message rendering is
confirmed to use `.textContent` (never `.innerHTML` with dynamic content) —
scoped to the chatbot's own script so the pre-existing site's course/FAQ
rendering (unrelated, out of scope) isn't flagged — the duplicate-submit
guard is present, and a source-tree scan confirms no hardcoded API keys,
no real `.env` file, and that `.gitignore`/`.env.example` are safe.

```powershell
cd chatbot-backend
python -m compileall app tests
python tests\test_core_pipeline.py
python tests\test_frontend_and_security.py
python tests\test_llm_prompting.py
python tests\test_no_api_mode.py
```

All four suites are plain scripts using only the standard library — no
`pytest`, no API key, and no network. (There are no pytest-style tests in
this project, so `python -m pytest -q` collects nothing.)

**Initializing the database** — this happens automatically on startup, but
to create it explicitly:

```powershell
cd chatbot-backend
python -c "from app import database; database.init_db(); print('Database initialized')"
```

**Last run in this environment: 105/105 in `test_core_pipeline.py`, 38/38
in `test_frontend_and_security.py`, 71/71 in `test_llm_prompting.py`,
43/43 in `test_no_api_mode.py`** (257 checks total, 0 failures). All were
actually executed, not just written.

`test_llm_prompting.py` and `test_no_api_mode.py` use fake providers and a
hard socket block, respectively — neither needs a real API key, network
access, or `openai`/`anthropic` to be installed.

A `pytest`-based FastAPI `TestClient` suite can be added the same way once
`fastapi`/`httpx` are installed in your environment (already listed in
`requirements.txt`); this project was developed in a sandbox with **no
network access**, so `pip install` could not run here. As a result:

- **Verified in this environment:** all pipeline/retrieval/conversation
  logic above, `python -m py_compile app/*.py` (no syntax errors), JS
  syntax via `node --check` (Node.js was available in this sandbox),
  static HTML/element/security checks.
- **Not verified in this environment (blocked by no network access):**
  the live `uvicorn` server, real HTTP requests to `/health` or `/chat`,
  CORS behavior against a real browser, rate-limit behavior under load,
  the actual rendered dark-glass UI, the liquid-glass logo animation, and
  any keyboard/mobile/screen-reader behavior. These require a real
  browser and are listed explicitly so they aren't mistaken for having
  been tested.

Please run the manual checklist below after starting both servers locally
(see the Windows PowerShell or macOS/Linux setup sections above):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType "application/json" -Body '{"message": "Which grades are supported?"}'
```

### Manual checklist (requires a real browser — not yet performed)

- Greetings, thanks, goodbyes
- Grades, subjects, programs, fees (should say "not published", never a
  number), admissions, contact, teaching approach, scheduling questions
- Program comparison ("compare Foundation Years and Middle School")
- Follow-ups: "tell me more", "what about the first one", "the second
  program", pronoun references, and confirming "one-on-one classes"
  doesn't get misread as an ordinal reference
- Frustrated ("this is useless") and confused ("I don't understand")
  messages get an empathetic / clarifying reply, not a generic fallback
- Topic switching (asking about batch size, then working hours, and
  confirming the second answer isn't contaminated by the first topic)
- Clear Chat (resets both the visible widget and server-side history)
- Backend offline (frontend shows a friendly connection error, not a
  crash); optional LLM key missing or invalid (falls back to template
  answers automatically)
- Empty message, very long message (2000+ characters), rapid repeated
  clicks on Send (input/button disable during a pending request, so no
  duplicate requests)
- Prompt-injection attempts ("ignore previous instructions...")
- Desktop, tablet, and mobile widths (320px, 375px, 430px) — no
  horizontal overflow, no clipped text, touch targets large enough
- Keyboard navigation (Tab to reach the launcher, Enter/Space to open,
  Tab through header/messages/suggestions/input/send, Escape to close);
  visible focus rings; `prefers-reduced-motion` disables animations
- Browser DevTools: no console errors, `/chat` requests visible and
  succeeding in the Network tab, correct ports on both requests

## Security Review

Verified by actually running `test_frontend_and_security.py` (source-tree
scan) and by reading the relevant code, not assumed from having a section
titled "Security":

- `.env` is git-ignored (`chatbot-backend/.env` in `.gitignore`, confirmed
  present in `.gitignore`'s text); only `.env.example` (blank secret
  fields, confirmed by regex) ships in the project/ZIP; a source-tree scan
  confirms no real `chatbot-backend/.env` file exists in the delivered
  project
- A source-tree scan (`test_frontend_and_security.py`) checked every
  `.py`/`.html`/`.js`/`.json`/`.md` file for common secret-key shapes
  (Anthropic/OpenAI-style key prefixes, AWS access key IDs, a populated
  `ANTHROPIC_API_KEY=`) — none found
- `ANTHROPIC_API_KEY` and `ADMIN_API_KEY` are read from environment
  variables only (`config.py`), never hardcoded, never logged, never
  echoed back in any API response, and never referenced anywhere in
  `index.html` or its JavaScript (the frontend only ever calls `/chat`,
  `/chat/clear`, and `/feedback` — it holds no key of any kind)
- `/admin/analytics` requires `X-Admin-Key` to match `ADMIN_API_KEY` and
  returns a plain 404 (not 401) when the key is unset or wrong, so the
  endpoint's existence isn't revealed to an unauthenticated caller
- The chatbot never reveals its system prompt/instructions: injection
  attempts ("ignore previous instructions", "reveal your system prompt",
  etc.) are pattern-matched and deflected before reaching retrieval or the
  optional LLM (verified by test); in the default (no-LLM) mode there is
  no model to prompt-inject in the first place — replies are always a
  fixed template built from knowledge-base text. When the optional LLM is
  active, the user's message is always sent labeled as "data, not
  instructions", and the persona system prompt explicitly forbids
  revealing instructions or inventing facts
- If the optional LLM is active, every response is passed through
  `sanitize_llm_output()` before use: HTML tags and control characters are
  stripped (defense-in-depth — the frontend already renders all message
  text via `.textContent`, verified by static test, never `.innerHTML`, so
  injected HTML/script text cannot execute either way), length is capped,
  and text matching a false-completion pattern ("I've booked...", "I have
  contacted...") is discarded rather than shown, so the assistant can
  never falsely claim to have performed a real action
- Only the single best-matching knowledge-base entry (or, for an explicit
  comparison, the specific two asked about) is ever returned — the full
  knowledge base is never dumped into a response; the retrieval
  token-coverage gate (see the RAG section above) additionally prevents
  an unrelated entry from leaking through as a false-confident match
- All request bodies are validated with Pydantic (length limits on
  `message`, `session_id`, `comment`, etc.); malformed JSON or missing
  fields return FastAPI's standard 422 response, not a stack trace
- The generic `Exception` handler in `main.py` returns a fixed, safe JSON
  error (`{"error": "An unexpected error occurred..."}`) and logs the real
  exception server-side (`error_logs` table) — no stack trace, file path,
  or internal detail reaches the visitor
- Per-IP rate limiting (`RATE_LIMIT_MAX_REQUESTS` per
  `RATE_LIMIT_WINDOW_SECONDS`, default 20/60s) applies to `/chat`,
  `/chat/clear`, and `/feedback`
- `ALLOWED_ORIGINS` defaults to explicit local dev origins, not `*` —
  update it to your real deployed domain(s) before going to production
- User-provided content is only ever stored as text (SQLite parameterized
  queries throughout `database.py`) or matched against the knowledge base
  lexically — never evaluated, executed, or interpolated into a shell
  command or template
- Session ids are validated against `[A-Za-z0-9_-]{8,100}` before use;
  trivially guessable ("1"), oversized, path-traversal (`../../etc/passwd`)
  and SQL-injection-shaped values are rejected and replaced with a freshly
  generated UUID (covered by tests)
- **Residual risk, stated plainly:** session ids are unauthenticated bearer
  values — anyone who obtains one can continue that conversation's
  follow-up context. This is acceptable here because the frontend mints a
  random UUID per browser tab and no personal or sensitive data is stored
  against a session, but it is *not* suitable for storing anything private.
  If sensitive data is ever added, real authentication must come first.
- The rate limiter prunes idle clients (verified by test), so a public
  deployment cannot be made to leak memory by cycling source IPs

## Production Deployment (handing this to a website developer)

**The key point:** this project has two halves that deploy to two
different places. Uploading the whole folder to the WeMentors web host
will *not* work unless that host can run Python.

| Half | What it is | Where it goes |
|---|---|---|
| `chatbot-backend/` + `knowledge/` | A Python/FastAPI service | A host that runs Python (Render, Railway, Fly.io, a VPS, etc.) — **not** typical shared/static web hosting |
| The chatbot widget inside `index.html` | Plain HTML/CSS/JS | Pasted into the existing WeMentors website |

### Step 1 — Deploy the backend

On a Python-capable host, from the project root:

```bash
pip install -r chatbot-backend/requirements.txt
cd chatbot-backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Do **not** use `--reload` in production. Run it under a process manager
(systemd, Docker, or your platform's built-in process runner) so it
restarts automatically after a crash or reboot.

### Step 2 — Configure environment variables

Create `chatbot-backend/.env` **on the server** (never commit it):

```bash
ALLOWED_ORIGINS=https://wementors.co,https://www.wementors.co
ANTHROPIC_API_KEY=            # optional; blank = deterministic KB mode
ADMIN_API_KEY=                # optional; blank disables /admin/analytics
LOG_LEVEL=INFO
```

`ALLOWED_ORIGINS` **must** list the real website origin(s), exactly as the
browser sends them (scheme + domain, no trailing slash). If this is wrong,
the widget will load but every request will fail with a CORS error in the
browser console.

### Step 3 — Get the public backend URL

After deploying you will have a URL such as
`https://wementors-chatbot.onrender.com`. Confirm it works before going
further:

```bash
curl https://YOUR-BACKEND-URL/health
```

You should get `{"status":"healthy", ...}`. Serve it over **HTTPS** — a
browser on an `https://` website will refuse to call an `http://` backend.

### Step 4 — Give the website developer the widget

The chatbot in `index.html` is self-contained in three blocks, all marked
with the comment `WeMentors chatbot`:

1. the `<style id="chatbot-styles">` block,
2. the chatbot markup (the shared `<svg>` logo defs, `#chatbotToggle`, and
   `#chatbotPanel`), and
3. the chatbot `<script>` block (the `(function () { ... })();` IIFE).

Copy those three blocks into the existing website's HTML, just before
`</body>`. They are namespaced under `chatbot*` ids/classes and their own
CSS variables, so they should not collide with existing site styles.

### Step 5 — Point the widget at the deployed backend

The widget defaults to `http://127.0.0.1:8000`. Override it by adding this
line **before** the chatbot script block:

```html
<script>window.CHATBOT_API_URL = "https://YOUR-BACKEND-URL";</script>
```

### Step 6 — Test the live website

Open the site and check, with DevTools open:

- the launcher appears and the panel opens
- a real question returns a real answer
- the Network tab shows `POST /chat` returning 200 (not a CORS error)
- no console errors
- it works on a phone, not just desktop

### Step 7 — Updating the knowledge base later

Edit `knowledge/wementors_kb.json` (see *Updating the Knowledge Base*
above), redeploy the backend, and confirm `/health` reports the new
`knowledge_base_entries` count. The website HTML does **not** need to
change — content updates are backend-only.

### Backup and rollback

- **Backup:** `chatbot-backend/data/chatbot.db` holds conversation history.
  Back it up on whatever schedule suits; it is not required for the bot to
  function (a fresh database is created automatically if missing).
- **Rollback:** the backend is stateless apart from that file, so rolling
  back means redeploying the previous commit. Keep `wementors_kb.json`
  under version control so a bad content edit can be reverted on its own.
- **Persistence warning:** some platforms (including Render's free tier)
  use ephemeral disks, so the SQLite file is wiped on each redeploy. The
  chatbot still works — only stored history is lost. Use a persistent
  volume if you want history to survive deploys.

## Deployment Considerations

This project is set up for local development; before deploying it
publicly:

- Set `ALLOWED_ORIGINS` to your real production domain(s) only — never `*`
- Set a strong, random `ADMIN_API_KEY` if you want `/admin/analytics`
  enabled in production (leave it blank to disable the endpoint entirely)
- Put the backend behind HTTPS (a reverse proxy such as nginx/Caddy, or
  your hosting platform's TLS termination) — plain `uvicorn --reload` is
  for local development only
- Run without `--reload` in production (`python -m uvicorn app.main:app
  --host 0.0.0.0 --port 8000`), and consider a process manager
  (systemd, Docker, a platform-as-a-service) instead of a bare terminal
- The in-memory rate limiter and SQLite file are both per-process/single
  file — fine for one backend instance, but a multi-instance/multi-worker
  deployment needs a shared rate-limit store (e.g. Redis) and either a
  shared database or a networked one instead of local SQLite
- Update `window.CHATBOT_API_URL` (or set it via a `<script>` tag before
  the chatbot script in `index.html`) to point at the deployed backend's
  real URL instead of `http://127.0.0.1:8000`
- Back up or rotate `chatbot-backend/data/chatbot.db` per your own data
  retention policy — it accumulates conversation history over time

## Current Limitations

- Retrieval is lexical (TF-IDF + keywords + a token-coverage gate), not
  semantic embeddings — works well for this FAQ's size but won't catch
  paraphrases with no shared words at all.
- **Known residual retrieval case:** "What program is offered in summer?"
  still scores 0.2355 (above the 0.12 threshold) and returns the subjects
  list, because it shares two real words with that entry while the
  discriminating word ("summer") is simply absent from the corpus. It does
  not fabricate a summer program, but it answers an adjacent question
  rather than saying "I don't have that." IDF-weighted coverage was
  evaluated as a fix and rejected — it moved the score only to 0.2202 while
  slightly degrading a legitimate query — so this is documented rather than
  papered over. Adding an explicit "we don't offer summer/holiday
  programmes" knowledge-base entry (once WeMentors confirms the fact) would
  resolve it cleanly.
- The in-memory rate limiter is per-process; a multi-worker or
  multi-instance deployment needs a shared store (e.g. Redis) instead.
- No authentication/user accounts (not required for this use case).
- `sanitize_llm_output()`'s validation logic (HTML stripping, length cap,
  false-completion-claim detection) has automated test coverage, but the
  the providers' actual network calls have not been exercised against a
  live Groq or Anthropic API key in this environment (no network access) —
  test it with a real key before relying on it in production.
- Demo-class enquiry details are logged as a conversation message, not
  yet forwarded anywhere (e.g. email/CRM) — see Future Improvements.
- The premium dark-glass UI, liquid-glass logo animation, and all
  responsive/keyboard/reduced-motion behavior were built and verified with
  static checks (HTML parsing, required elements, JS syntax) but have
  **not** been visually confirmed in a real browser from this sandboxed
  environment (no display/browser/network available here) — please do a
  visual pass after starting the servers locally, using the manual
  checklist above.
- The live FastAPI server itself (`uvicorn`) has not been started in this
  environment, so real HTTP request/response behavior, CORS preflight
  handling, and live rate-limiting have been verified by code review and
  by exercising the same underlying Python objects `main.py` calls, but
  not by an actual HTTP round-trip. Run the `Invoke-RestMethod` commands
  above locally to confirm.

## Future Improvements

- Forward demo-class enquiries to an email address or CRM instead of just
  logging them
- Swap `retrieval.py`'s lexical search for embeddings if the knowledge
  base grows significantly
- Admin UI for editing `wementors_kb.json` without touching JSON by hand
- Automated FastAPI `TestClient` suite (`pytest`) once dependencies can be
  installed in CI
- Deploy frontend and backend, move `ALLOWED_ORIGINS` to the real domain

## Git Workflow

```bash
git checkout chatbot-development
git add .
git commit -m "Describe your changes"
git push origin chatbot-development
```

After testing, merge into `main` through GitHub or Git.

## License

This project is intended for the WeMentors website. Add an appropriate
license here if the project will be distributed publicly.
