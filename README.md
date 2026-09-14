# WeMentors Chatbot

An FAQ-powered chatbot for the WeMentors website. The project combines a lightweight website chat interface with a FastAPI backend that retrieves relevant answers from a structured WeMentors knowledge base.

## Features

- Floating chatbot interface embedded into the website
- FAQ-based responses using the WeMentors knowledge base
- Follow-up questions using recent conversation history
- Clear Chat functionality
- FastAPI backend with `/health` and `/chat` endpoints
- Input validation for chat messages and conversation history
- Friendly fallback responses when a relevant answer cannot be found
- Simple local development setup with separate frontend and backend servers

## Project Structure

```text
wementorschatbotdev/
├── index.html
├── knowledge/
│   └── wementors_faq.md
├── chatbot-backend/
│   └── app/
│       └── main.py
└── README.md
```

## How It Works

1. A visitor opens the WeMentors website and uses the floating chatbot.
2. The frontend sends the user's message and recent conversation history to the FastAPI backend.
3. The backend searches the structured FAQ knowledge base for the most relevant section.
4. A response is returned to the frontend and displayed in the chat window.
5. The frontend stores recent messages locally in the current chat session so follow-up questions can be answered with context.
6. The Clear Chat button removes the current conversation history and resets the chat interface.

## Technologies Used

- **HTML, CSS, and JavaScript** — chatbot interface and frontend behavior
- **Python** — backend logic
- **FastAPI** — API framework
- **Uvicorn** — local ASGI server
- **Pydantic** — request validation
- **Markdown** — knowledge-base format
- **python-dotenv** — environment configuration support

## Requirements

- Python 3.10 or newer
- Git
- A modern web browser

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/raaixd/wementorschatbot.git
cd wementorschatbot
```

### 2. Create a virtual environment

From the project root:

```bash
python -m venv chatbot-backend/.venv
```

Activate it on Windows Command Prompt:

```cmd
chatbot-backend\.venv\Scripts\activate
```

Activate it on Windows PowerShell:

```powershell
.\chatbot-backend\.venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```bash
pip install fastapi uvicorn python-dotenv
```

### 4. Start the backend

Open a terminal in the project root and run:

```bash
cd chatbot-backend
python -m uvicorn app.main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

Health-check endpoint:

```text
http://127.0.0.1:8000/health
```

### 5. Start the frontend

Open a second terminal in the project root:

```bash
python -m http.server 5500
```

Open the website at:

```text
http://127.0.0.1:5500
```

Keep both terminals running while testing the chatbot.

## API Endpoints

### `GET /health`

Checks whether the backend is running.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /chat`

Accepts a user message and optional recent conversation history.

Example request:

```json
{
  "message": "Which grades are supported?",
  "history": []
}
```

Example response:

```json
{
  "reply": "..."
}
```

## Updating the Knowledge Base

Edit:

```text
knowledge/wementors_faq.md
```

Use the following structure:

```markdown
## Category Name

### Question

Answer to the question.

### Another Question

Answer to the second question.
```

Each `###` heading represents a separate FAQ entry. After editing the knowledge base, restart the backend if necessary and test the updated questions through the chatbot.

## Testing Checklist

Test the chatbot with:

- Greetings
- Questions about supported grades
- Questions about available subjects
- Questions about programs
- Questions about fees and admissions
- Questions about contact information
- Follow-up questions such as “Tell me more about that”
- Unrelated questions
- Empty or excessively long messages
- Clear Chat functionality
- Backend unavailable or disconnected scenarios

## Current Limitations

- Retrieval is based on FAQ parsing and keyword matching rather than a full semantic-search system.
- The current conversation history is maintained in the browser session.
- Authentication and user accounts are not included.
- No persistent conversation database is included.
- Production deployment, monitoring, rate limiting, and multi-user isolation still need to be configured before public release.
- The backend currently runs locally and requires a separate deployment for production use.

## Future Improvements

- Add semantic retrieval using embeddings and a vector database
- Add an LLM response layer for more natural answers
- Add source references for answers
- Improve intent detection and irrelevant-question handling
- Add automated backend tests
- Add rate limiting and request logging
- Add environment-based configuration
- Deploy the frontend and backend
- Add analytics for common visitor questions
- Add administrator tools for updating the knowledge base

## Git Workflow

The repository uses a development branch for chatbot changes.

```bash
git checkout chatbot-development
git add .
git commit -m "Describe your changes"
git push origin chatbot-development
```

After testing, merge the development branch into `main` through GitHub or Git:

```bash
git checkout main
git merge chatbot-development
git push origin main
```

## License

This project is intended for the WeMentors website. Add an appropriate license here if the project will be distributed publicly.
