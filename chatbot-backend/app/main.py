from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="WeMentors Chatbot API",
    version="1.0.0",
    description="FAQ chatbot backend for the WeMentors website.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    reply: str


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_FILE = PROJECT_ROOT / "knowledge" / "wementors_faq.md"

GENERAL_FALLBACK = (
    "I can help you with WeMentors subjects, grades, programs, online classes, "
    "demo classes, mentoring, and contact details. What would you like to know?"
)

UNKNOWN_ANSWER = (
    "I couldn't find a precise answer to that in the WeMentors knowledge base. "
    "Please contact the WeMentors team for confirmation."
)

KNOWLEDGE_ERROR = (
    "The WeMentors knowledge base is currently unavailable. "
    "Please contact the WeMentors team directly for assistance."
)


class FAQSection:
    def __init__(self, heading: str, content: str) -> None:
        self.heading = heading.strip()
        self.content = content.strip()


def load_knowledge() -> str:
    try:
        return KNOWLEDGE_FILE.read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError, UnicodeError):
        return ""


def parse_faq_sections(knowledge: str) -> List[FAQSection]:
    """Parse each ### question and its answer as a separate FAQ section."""
    sections: List[FAQSection] = []
    current_heading: Optional[str] = None
    current_content: List[str] = []

    def save_current_section() -> None:
        if current_heading is not None:
            content = "\n".join(current_content).strip()
            if content:
                sections.append(FAQSection(current_heading, content))

    for raw_line in knowledge.splitlines():
        line = raw_line.strip()

        # Ignore the document title and category headings.
        if line.startswith("# ") or line.startswith("## "):
            save_current_section()
            current_heading = None
            current_content = []
            continue

        # Every level-three heading starts one independent FAQ answer.
        if line.startswith("### "):
            save_current_section()
            current_heading = line[4:].strip()
            current_content = []
            continue

        if current_heading is not None:
            current_content.append(raw_line)

    save_current_section()
    return sections


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def contains_any(text: str, phrases: List[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def find_section(sections: List[FAQSection], keywords: List[str]) -> Optional[FAQSection]:
    for section in sections:
        heading = normalize(section.heading)
        if all(keyword in heading for keyword in keywords):
            return section
    return None


def select_sections(message: str, sections: List[FAQSection]) -> List[FAQSection]:
    text = normalize(message)

    exact_rules = [
        (["what is wementors", "who can use wementors"], ["what is wementors"]),
        (["which grades", "what grades", "grades supported", "classes supported"], ["which grades"]),
        (["curriculum", "curricula", "cbse", "icse", "igcse", "ib"], ["curricula"]),
        (["what subjects", "which subjects", "subjects offered", "what do you teach"], ["what subjects"]),
        (["online classes", "live classes", "virtual classes"], ["are classes online"]),
        (["one-on-one", "one on one"], ["one-on-one classes"]),
        (["batch size", "how many students"], ["batch size"]),
        (["mentoring process", "how does mentoring"], ["how does the mentoring process"]),
        (["progress updates", "progress tracked"], ["progress updates"]),
        (["doubt-solving", "doubt solving"], ["doubt-solving sessions"]),
        (["teaching approach", "how do you teach"], ["what teaching approach"]),
        (["foundation", "grade 3", "grade 4", "grade 5"], ["foundation years"]),
        (["middle school", "grade 6", "grade 7", "grade 8"], ["middle school"]),
        (["senior school", "grade 9", "grade 10", "board preparation"], ["senior school"]),
        (["confident speaker", "spoken english", "public speaking", "interview skills"], ["confident speaker"]),
        (["demo", "trial class", "book a demo"], ["demo class"]),
        (["contact", "phone number", "whatsapp", "email address"], ["contact"]),
        (["working hours", "hours"], ["working hours"]),
        (["response time", "reach out within"], ["expected response time"]),
    ]

    for triggers, heading_keywords in exact_rules:
        if contains_any(text, triggers):
            section = find_section(sections, heading_keywords)
            if section:
                return [section]

    # Generic keyword fallback, returning only the most relevant section.
    keywords = [word for word in text.split() if len(word) > 3]
    scored = []
    for section in sections:
        searchable = normalize(section.heading + " " + section.content)
        score = sum(1 for keyword in keywords if keyword in searchable)
        if score:
            scored.append((score, section))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return [scored[0][1]]

    return []


def clean_answer(content: str) -> str:
    lines = content.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    result: List[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        result.append(line.rstrip())
        previous_blank = blank
    return "\n".join(result).strip()


def format_reply(sections: List[FAQSection]) -> str:
    answers = []
    for section in sections:
        answer = clean_answer(section.content)
        if answer and answer not in answers:
            answers.append(answer)
    return "\n\n".join(answers) if answers else UNKNOWN_ANSWER


def generate_reply(message: str, history: Optional[List[ChatMessage]] = None) -> str:
    knowledge = load_knowledge()
    if not knowledge:
        return KNOWLEDGE_ERROR

    sections = parse_faq_sections(knowledge)

    # Include recent conversation context so follow-up questions work.
    context_parts: List[str] = []
    for item in (history or [])[-10:]:
        if item.role in {"user", "assistant"}:
            context_parts.append(item.content)

    selection_message = " ".join(context_parts + [message])
    selected = select_sections(message, sections)

    if not selected:
        return GENERAL_FALLBACK

    return format_reply(selected)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "wementors-chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        return ChatResponse(reply="Please enter a question so I can help you.")
    return ChatResponse(reply=generate_reply(message))
