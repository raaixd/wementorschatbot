from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .knowledge import KnowledgeBase


ORDINALS = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "second": 1,
    "2nd": 1,
    "two": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
    "fourth": 3,
    "4th": 3,
    "four": 3,
    "last": -1,
}

FOLLOWUP_HINTS = (
    "tell me more",
    "more detail",
    "more details",
    "explain that",
    "simpler",
    "what about",
    "and the",
    "the first",
    "the second",
    "the third",
    "the fourth",
    "that one",
    "this one",
    "its fees",
    "their fees",
    "how much",
    "how do i apply",
    "how can i apply",
    "how do i enroll",
)

UNRELATED_HINTS = (
    "weather",
    "stock market",
    "movie",
    "recipe",
    "who won",
    "capital of",
    "write python",
    "homework",
    "crypto",
    "bitcoin",
    "medical advice",
    "legal advice",
)

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|namaste)\b", re.I)
THANKS_RE = re.compile(r"\b(thanks|thank you|thx|appreciate it)\b", re.I)
BYE_RE = re.compile(r"^\s*(bye|goodbye|see you|that'?s all|ok thanks bye)\b", re.I)


@dataclass
class IntentResult:
    name: str
    rewritten_query: str
    program_id: Optional[str] = None
    wants_more_detail: bool = False
    wants_simpler: bool = False
    is_followup: bool = False


def _contains(text: str, phrases: List[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def detect_intent(message: str, state: Dict, kb: KnowledgeBase) -> IntentResult:
    text = message.lower().strip()
    wants_more = _contains(text, ["tell me more", "more detail", "more details", "explain more"])
    wants_simpler = _contains(text, ["simpler", "simple words", "in simple"])
    followup = wants_more or wants_simpler or _contains(text, list(FOLLOWUP_HINTS))
    if re.search(r"\b(it|that|they|them|this|those)\b", text) and len(text.split()) <= 12:
        followup = True

    if GREETING_RE.search(text) and len(text.split()) <= 6:
        return IntentResult("greeting", message)
    if BYE_RE.search(text):
        return IntentResult("goodbye", message)
    if THANKS_RE.search(text) and len(text.split()) <= 8:
        return IntentResult("thanks", message)

    if _contains(text, ["fee", "fees", "cost", "price", "pricing", "how much", "charge", "tuition"]):
        return IntentResult("fees", message, is_followup=followup)
    if _contains(text, ["refund", "cancellation", "guarantee", "admission policy"]):
        return IntentResult("policies", message, is_followup=followup)
    if _contains(text, ["enroll", "enrol", "apply", "join", "admission", "book a demo", "demo", "trial"]):
        return IntentResult("admissions", message, is_followup=followup)
    if _contains(text, ["contact", "phone", "whatsapp", "email", "working hours", "office hours"]):
        return IntentResult("contact", message)
    if _contains(text, ["schedule", "timing", "timings", "when are classes", "availability"]):
        return IntentResult("scheduling", message)
    if _contains(text, ["curriculum", "curricula", "cbse", "icse", "igcse", " ib", "board"]):
        return IntentResult("curricula", message)
    if _contains(text, ["subject", "math", "science", "english", "what do you teach"]):
        return IntentResult("subjects", message)
    if _contains(text, ["grade", "class", "classes", "which classes"]):
        return IntentResult("grades", message)
    if _contains(text, ["teaching", "how do you teach", "approach", "one-on-one", "one on one", "batch"]):
        return IntentResult("teaching", message)
    if _contains(text, ["mentor", "progress", "doubt"]):
        return IntentResult("mentorship", message)
    if _contains(text, ["program", "course", "offer"]):
        program_id = resolve_program_reference(text, state, kb)
        if program_id:
            return IntentResult("program_detail", message, program_id=program_id, is_followup=True)
        return IntentResult("programs", message, is_followup=followup)

    program_id = resolve_program_reference(text, state, kb)
    if program_id:
        return IntentResult(
            "program_detail",
            message,
            program_id=program_id,
            wants_more_detail=wants_more,
            wants_simpler=wants_simpler,
            is_followup=True,
        )

    if followup:
        last_intent = state.get("last_intent")
        last_program = state.get("last_program_id")
        rewritten = expand_followup(message, state)
        return IntentResult(
            last_intent or "followup",
            rewritten,
            program_id=last_program,
            wants_more_detail=wants_more,
            wants_simpler=wants_simpler,
            is_followup=True,
        )

    if any(hint in text for hint in UNRELATED_HINTS) and "wementors" not in text:
        return IntentResult("unrelated", message)

    if len(text.split()) <= 2 and text not in {"fees", "programs", "subjects", "grades", "contact", "demo"}:
        return IntentResult("clarify", message)

    return IntentResult("question", message, wants_more_detail=wants_more, wants_simpler=wants_simpler)


def resolve_program_reference(text: str, state: Dict, kb: KnowledgeBase) -> Optional[str]:
    order: List[str] = state.get("listed_program_ids") or kb.programs_order
    if not order:
        return None

    for word, index in ORDINALS.items():
        if re.search(rf"\b{word}\b", text):
            if index == -1:
                return order[-1]
            if 0 <= index < len(order):
                return order[index]

    names = {
        "foundation": "foundation",
        "middle": "middle",
        "senior": "senior",
        "confident speaker": "confident_speaker",
        "spoken english": "confident_speaker",
        "public speaking": "confident_speaker",
    }
    for needle, program_id in names.items():
        if needle in text:
            return program_id
    return None


def expand_followup(message: str, state: Dict) -> str:
    last_topic = state.get("last_topic") or ""
    last_program = state.get("last_program_id") or ""
    last_question = state.get("last_user_message") or ""
    pieces = [message]
    if last_program:
        pieces.append(last_program.replace("_", " "))
    if last_topic:
        pieces.append(last_topic)
    if last_question:
        pieces.append(last_question)
    return " ".join(pieces)
