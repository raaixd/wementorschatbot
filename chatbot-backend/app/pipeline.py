from __future__ import annotations

import uuid
from typing import Dict, List, Tuple

from . import db
from .config import settings
from .generate import compose_answer, optional_llm_polish, source_blob
from .intent import detect_intent
from .knowledge import KnowledgeBase, load_knowledge
from .memory import compress_if_needed, merge_history, new_state, update_state
from .personality import KNOWLEDGE_ERROR, PERSONALITY, UNKNOWN_REPLY
from .retrieve import retrieve
from .safety import sanitize_user_text, split_intent_and_data, validate_reply
from .schemas import ChatMessage


def _session_id(raw: str | None) -> str:
    value = (raw or "").strip()
    if 8 <= len(value) <= 64 and all(ch.isalnum() or ch in "-_" for ch in value):
        return value
    return str(uuid.uuid4())


def run_pipeline(
    message: str,
    history: List[ChatMessage],
    session_id: str | None,
) -> Tuple[str, str, str, List[str]]:
    sid = _session_id(session_id)
    db.ensure_session(sid)
    state = db.load_state(sid) or new_state()
    stored_history = merge_history(sid, history)
    stored_history = compress_if_needed(state, stored_history)

    cleaned = sanitize_user_text(message, settings.max_message_length)
    if not cleaned:
        return "Please enter a question so I can help you.", sid, "empty", []

    injected, cleaned = split_intent_and_data(cleaned)

    try:
        kb: KnowledgeBase = load_knowledge()
        if not kb.entries:
            return KNOWLEDGE_ERROR, sid, "error", []
    except (OSError, ValueError):
        return KNOWLEDGE_ERROR, sid, "error", []

    intent = detect_intent(cleaned, state, kb)
    if injected and intent.name not in {"greeting", "thanks", "goodbye"}:
        db.log_event(sid, "injection_blocked", intent.name, cleaned[:80])
        reply = (
            "I can only help with WeMentors programs, classes, demo bookings, and contact details. "
            "What would you like to know about WeMentors?"
        )
        _persist(sid, cleaned, reply, state)
        return reply, sid, "injection", []

    category_hint = intent.name if intent.name in {
        "fees", "admissions", "contact", "scheduling", "subjects", "grades",
        "teaching", "mentorship", "programs", "policies", "curricula",
    } else ""

    query = intent.rewritten_query
    scored = retrieve(
        query,
        kb,
        top_k=settings.retrieve_top_k,
        min_score=settings.min_relevance_score,
        category_hint=category_hint,
    )

    if intent.name in {"greeting", "thanks", "goodbye", "unrelated", "clarify"}:
        scored = []

    if intent.program_id and not any(item.entry.id == intent.program_id for item in scored):
        program = kb.by_id(intent.program_id)
        if program:
            from .retrieve import ScoredEntry

            scored = [ScoredEntry(entry=program, score=99.0)] + list(scored)

    draft = compose_answer(intent, scored, kb, state, greeting_index=int(state.get("turn_count") or 0))
    entries = [item.entry for item in scored]
    context = source_blob(entries)
    history_text = "\n".join(f"{item['role']}: {item['content']}" for item in stored_history[-6:])
    polished = optional_llm_polish(cleaned, draft, PERSONALITY + "\n\n" + context, history_text)
    reply = polished or draft
    reply = validate_reply(reply, context + " " + draft)

    if not reply.strip():
        reply = UNKNOWN_REPLY

    state = update_state(state, cleaned, intent.name, list(scored), kb, intent.program_id or "")
    db.save_state(sid, state)
    _persist(sid, cleaned, reply, state)
    db.log_event(sid, "chat", intent.name, cleaned[:80])
    sources = [entry.id for entry in entries[:3]]
    if intent.program_id:
        sources = [intent.program_id] + [sid for sid in sources if sid != intent.program_id]
    return reply, sid, intent.name, sources[:3]


def _persist(session_id: str, user_message: str, reply: str, state: Dict) -> None:
    db.add_message(session_id, "user", user_message)
    db.add_message(session_id, "assistant", reply)
    db.save_state(session_id, state)
