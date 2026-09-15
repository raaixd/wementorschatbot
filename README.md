# WeMentors Chatbot

A website assistant for WeMentors Academy. Visitors ask questions in a floating chat on the existing site. A FastAPI backend retrieves only the most relevant knowledge-base entries and writes a concise, grounded reply.

The chatbot is designed to sound like a friendly academic guidance assistant. It answers from verified WeMentors information and does not invent fees, schedules, or policies.

## Features

- Hybrid FAQ retrieval (keywords, aliases, categories, and BM25-style scoring)
- Intent detection and follow-up handling (“the first program”, “its fees”, “explain that simply”)
- Session memory in the browser plus SQLite on the backend
- Optional OpenAI phrasing layer; local answers still work without an API key
- Rate limiting, input validation, prompt-injection filtering, and health checks
- Chat UI matched to the WeMentors purple/gold visual identity

## Project structure

```text
wementorschatbotdev/
├── index.html                 # Existing WeMentors website + chatbot widget
├── knowledge/
│   ├── wementors_kb.json      # Structured knowledge base (source of truth)
│   └── wementors_faq.md       # Human-readable copy of the same facts
├── chatbot-backend/
│   ├── app/
│   │   ├── main.py            # FastAPI routes and security middleware
│   │   ├── pipeline.py        # RAG orchestration
│   │   ├── retrieve.py        # Ranking / top-k retrieval
│   │   ├── intent.py          # Intent + reference resolution
│   │   ├── generate.py        # Answer composition
│   │   ├── knowledge.py       # Knowledge-base loader
│   │   ├── memory.py          # Conversation state
│   │   ├── db.py              # SQLite persistence
│   │   ├── safety.py          # Sanitization and injection checks
│   │   └── personality.py     # Tone and behaviour rules
│   ├── tests/
│   ├── data/                  # Local SQLite file (not committed)
│   ├── requirements.txt
│   └── .env.example
├── .env.example
└── README.md
```

## Architecture

```text
User message
  → sanitize / injection check
  → session memory (history + last program/topic)
  → intent detection and follow-up rewrite
  → retrieve top-k knowledge entries
  → relevance filter
  → compose answer from retrieved facts only
  → optional LLM polish (if OPENAI_API_KEY is set)
  → validate (no invented prices)
  → JSON response + SQLite log
```

The full knowledge base is never sent as context. Only the highest-scoring entries are used.

## Local setup

### Requirements

- Python 3.10 or newer
- A modern browser

### 1. Create a virtual environment

```bash
python -m venv chatbot-backend/.venv
```

Windows PowerShell:

```powershell
.\chatbot-backend\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r chatbot-backend/requirements.txt
```

### 3. Environment variables

Copy `chatbot-backend/.env.example` to `chatbot-backend/.env` if you need local overrides.

The chatbot runs without any paid API. To optionally polish answers with OpenAI:

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Never put API keys in `index.html`.

### 4. Start the backend

From `chatbot-backend`:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check: http://127.0.0.1:8000/health

### 5. Start the website

From the project root:

```bash
python -m http.server 5500
```

Open http://127.0.0.1:5500

Keep both processes running while you test the chat widget.

To point the widget at another API URL, set this before the chatbot script (or in the browser console):

```javascript
window.WEMENTORS_CHAT_API = "http://127.0.0.1:8000";
```

## API

### `GET /health`

```json
{
  "status": "healthy",
  "service": "wementors-chatbot",
  "knowledge_entries": 27,
  "database": "ok"
}
```

### `POST /chat`

```json
{
  "message": "What programs do you offer?",
  "history": [],
  "session_id": "optional-browser-session-id"
}
```

Response:

```json
{
  "reply": "...",
  "session_id": "...",
  "intent": "programs",
  "sources": ["programs-overview"]
}
```

### `POST /chat/clear`

```json
{ "session_id": "optional-browser-session-id" }
```

## Knowledge base

Edit `knowledge/wementors_kb.json`. Each entry should include:

- `id`, `category`, `question`, `answer`
- `aliases`, `keywords`, `follow_ups`
- `confidence` (`verified` or `missing`)
- `source`

Then update `knowledge/wementors_faq.md` so the readable copy stays aligned. Restart the backend after JSON changes (or rely on `--reload`).

Do not add unverified fees, discounts, schedules, or policies. Use `confidence: "missing"` and a confirmation placeholder instead.

## Database

SQLite is used locally for:

- Session IDs and compact conversation state
- Recent messages (follow-ups after refresh/restart)
- Lightweight event logs (intent + truncated question text)

It is not a user-account system. Avoid putting extra personal data into the chat if you do not need it stored. The database file lives at `chatbot-backend/data/chatbot.db` and is gitignored.

## Tests

From the project root, with the virtual environment active:

```bash
python -m pytest chatbot-backend/tests/test_chat.py -q
```

## Deployment notes

1. Host `index.html` (and assets) on your web server or static host.
2. Host the FastAPI app behind HTTPS (for example Uvicorn + Nginx, or a PaaS).
3. Set `ENVIRONMENT=production`.
4. Set `ALLOWED_ORIGINS` to the real website origin(s). Development currently also allows `*` so local file/live-server testing works.
5. Set `WEMENTORS_CHAT_API` in the frontend to the public API URL.
6. Keep `.env` and SQLite off public repositories.
7. Confirm WeMentors fee/policy facts before publishing those answers.

## License

This project is intended for the WeMentors website.
