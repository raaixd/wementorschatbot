from __future__ import annotations

from typing import Dict, List

from .config import settings
from . import db
from .knowledge import KnowledgeBase
from .retrieve import ScoredEntry


def new_state() -> Dict:
    return {
        "last_intent": "",
        "last_topic": "",
        "last_program_id": "",
        "listed_program_ids": [],
        "last_entry_ids": [],
        "last_user_message": "",
        "summary": "",
        "turn_count": 0,
    }


def merge_history(session_id: str, request_history: List) -> List[Dict[str, str]]:
    stored = db.recent_messages(session_id, settings.history_limit)
    if stored:
        return stored
    cleaned = []
    for item in request_history[-settings.history_limit :]:
        role = getattr(item, "role", None) or item.get("role")
        content = getattr(item, "content", None) or item.get("content")
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def compress_if_needed(state: Dict, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(history) <= settings.history_limit:
        return history
    older = history[: -settings.history_limit // 2]
    topics = []
    for item in older:
        if item["role"] == "user":
            topics.append(item["content"][:80])
    state["summary"] = "Earlier topics: " + "; ".join(topics[-6:])
    return history[-settings.history_limit // 2 :]


def update_state(
    state: Dict,
    message: str,
    intent_name: str,
    scored: List[ScoredEntry],
    kb: KnowledgeBase,
    program_id: str = "",
) -> Dict:
    state["last_intent"] = intent_name
    state["last_user_message"] = message
    state["turn_count"] = int(state.get("turn_count") or 0) + 1
    if scored:
        state["last_topic"] = scored[0].entry.category
        state["last_entry_ids"] = [item.entry.id for item in scored]
        if scored[0].entry.category == "programs" or scored[0].entry.id in kb.programs_order:
            if scored[0].entry.id == "programs-overview":
                state["listed_program_ids"] = list(kb.programs_order)
            elif scored[0].entry.id in kb.programs_order:
                state["last_program_id"] = scored[0].entry.id
    if program_id:
        state["last_program_id"] = program_id
        state["listed_program_ids"] = list(kb.programs_order)
        state["last_topic"] = "programs"
    return state
