"""
Conversation engine: the actual RAG pipeline.

    user message
      -> sanitize / validate
      -> intent detection (greeting / thanks / goodbye / demo enquiry / faq / off-topic)
      -> reference resolution (uses the previous turn's retrieved entries)
      -> knowledge-base retrieval (retrieval.py)
      -> confidence check
      -> answer generation (llm.py if enabled, else template-based)
      -> safe fallback if nothing confident was found

Session state (recent messages, what was last discussed) is read from the
database rather than kept in a process-wide dict, so behavior is correct
across backend restarts and multiple workers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from . import config, database, llm, personality
from .knowledge import KBEntry
from .retrieval import Retriever, ScoredEntry

_ORDINAL_WORDS = {
    # Deliberately only explicit ordinal words ("first", "1st"), not bare
    # number words ("one", "two") — those are too common in ordinary
    # questions (e.g. "one-on-one classes") and would cause false-positive
    # reference resolution right after a list-type answer.
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "last": -1,
}

_GREETING_START_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s+(morning|afternoon|evening)|greetings|yo|namaste)\b",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^\s*(thank(s| you)?(\s+(a lot|so much|very much))?|many thanks|thx|much appreciated|appreciate it)\s*[!.]*\s*$",
    re.IGNORECASE,
)
_GOODBYE_RE = re.compile(
    r"^\s*(bye|goodbye|see you|take care|that('?s| is) all|no,?\s*that('?s| is) (it|all))\s*[!.]*\s*$",
    re.IGNORECASE,
)

_HELP_RE = re.compile(
    r"^\s*(help|help me|can you help( me)?|i need help|please help|support|assist(ance)?)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_ADVISING_RE = re.compile(
    r"\b(help (me )?choose|recommend( a)? (class|course|program)|which (class|program|course) should (i|my child)|which (class|program|course) is (best|right|suitable)|not sure which (class|program|course)|choose a class)\b",
    re.IGNORECASE,
)

_VAGUE_MORE_RE = re.compile(
    r"^\s*(i want to know more|tell me more|know more|more info|more details|learn more)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_VAGUE_COST_RE = re.compile(
    r"^\s*(how much\??|how much is it\??|cost\??|what is the cost\??|price\??|pricing\??)\s*$",
    re.IGNORECASE,
)

_DEMO_BOOKING_RE = re.compile(
    r"\b(book|schedule|sign( me)? up|register|enroll( me)?|start|get|have|try|take|request|arrange|attend)\b.*\bdemo\b"
    r"|\b(want|need|like|interested in)\b.*\bdemo\b"
    r"|\bdemo class booking\b|\bbook( a)? demo\b"
    r"|\bhow (can|do) i (book|get|attend|schedule) a demo\b"
    r"|\b(how to join|how do i join|want to join|enquire about joining|interested in joining)\b"
    r"|\b(how do i enroll|how to enroll|admissions? process|admission enquiry|enquire about classes)\b",
    re.IGNORECASE,
)
_MORE_RE = re.compile(r"\b(tell me more|more (details|info)|explain (that|more)|elaborate|go on)\b", re.IGNORECASE)
_REFERENCE_WORD_RE = re.compile(r"\b(it|that|this|they|those|these|the\s+program|the\s+course|the\s+one)\b", re.IGNORECASE)

_INJECTION_MARKERS_RE = re.compile(
    r"\b(ignore (all |any )?(previous|prior|above) instructions|system prompt|you are now|"
    r"disregard (all|previous) instructions|reveal your (prompt|instructions))\b",
    re.IGNORECASE,
)

_FRUSTRATED_RE = re.compile(
    r"\b(this (isn'?t|is not) working|useless|terrible|awful|ridiculous|frustrat\w*|"
    r"annoying|angry|fed up|waste of time|not helpful|stupid (bot|chatbot|assistant))\b",
    re.IGNORECASE,
)
_CONFUSED_RE = re.compile(
    r"\b(i don'?t understand|confus\w*|what do you mean|that makes no sense|huh\??$|i'?m lost)\b",
    re.IGNORECASE,
)
_RELATIVE_REF_RE = re.compile(r"\b(next|following|after that|previous|prior|one before)\b", re.IGNORECASE)

_COMPARISON_RE = re.compile(r"\b(compare|comparison|difference between|vs\.?|versus)\b", re.IGNORECASE)

_PROGRAM_NAME_HINTS = {
    "program-foundation-years": ["foundation"],
    "program-middle-school": ["middle"],
    "program-senior-school": ["senior", "board"],
    "program-confident-speaker": ["confident", "speaker", "spoken", "speaking", "interview"],
}


@dataclass
class ReplyResult:
    reply: str
    intent: str
    matched_entry_ids: List[str]
    confidence: Optional[float]


class ConversationEngine:
    def __init__(self, entries: List[KBEntry]):
        self.entries = entries
        self.entries_by_id = {e.id: e for e in entries}
        self.retriever = Retriever(entries)

    # ---- intent detection -------------------------------------------------
    def detect_intent(self, text: str) -> str:
        stripped = text.strip()
        word_count = len(stripped.split())
        if _HELP_RE.match(stripped):
            return "help"
        if _ADVISING_RE.search(stripped):
            return "advising"
        if _GREETING_START_RE.match(stripped) and word_count <= 4:
            return "greeting"
        if _GOODBYE_RE.match(stripped):
            return "goodbye"
        if _INJECTION_MARKERS_RE.search(stripped):
            return "injection_attempt"
        if _FRUSTRATED_RE.search(stripped):
            return "frustrated"
        if _DEMO_BOOKING_RE.search(stripped):
            return "demo_booking"
        if _THANKS_RE.search(stripped) and len(stripped.split()) <= 6:
            return "thanks"
        if _COMPARISON_RE.search(stripped):
            return "comparison"
        if _CONFUSED_RE.search(stripped):
            return "confused"
        return "faq"

    def _extract_ordinal_index(self, text: str) -> Optional[int]:
        lowered = text.lower()
        for word, index in _ORDINAL_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                return index
        return None

    def _is_reference_query(self, text: str) -> bool:
        return bool(_MORE_RE.search(text) or _REFERENCE_WORD_RE.search(text))

    # ---- reference resolution ----------------------------------------------
    def _resolve_ordinal_reference(
        self, text: str, last_list_ids: List[str], last_matched_ids: Optional[List[str]] = None
    ) -> Optional[KBEntry]:
        """Resolve positional references against the remembered list.

        Absolute ("the first one", "the second program") and relative
        ("what about the next one?", "the previous one") are both
        supported. Relative references step from whichever list item was
        last discussed, which is why last_matched_ids is needed."""
        if not last_list_ids:
            return None

        ordinal = self._extract_ordinal_index(text)
        if ordinal is not None:
            index = ordinal if ordinal >= 0 else len(last_list_ids) - 1
            if 0 <= index < len(last_list_ids):
                return self.entries_by_id.get(last_list_ids[index])

        relative = _RELATIVE_REF_RE.search(text)
        if relative:
            current = next((i for i in (last_matched_ids or []) if i in last_list_ids), None)
            if current is None:
                return None
            step = -1 if relative.group(1).lower() in {"previous", "prior", "one before"} else 1
            index = last_list_ids.index(current) + step
            if 0 <= index < len(last_list_ids):
                return self.entries_by_id.get(last_list_ids[index])
        return None

    def _resolve_pronoun_reference(self, text: str, last_matched_ids: List[str]) -> Optional[KBEntry]:
        """Pronoun references ("it", "that", "tell me more") are only used
        as a fallback when direct retrieval doesn't already find a strong,
        specific match — a phrase like "how much does it cost" should still
        be answered as a fees question, not silently reinterpreted as
        "the previous topic's price"."""
        if self._is_reference_query(text) and last_matched_ids:
            return self.entries_by_id.get(last_matched_ids[0])
        return None

    def _get_last_turn_context(self, session_id: str) -> tuple[List[str], List[str]]:
        """Return (list_ids, last_matched_entry_ids) for reference resolution.

        These are tracked separately on purpose. `last_matched_entry_ids`
        is the most recent answer, used for pronouns ("it", "that").
        `list_ids` is the most recently *enumerated list* — which may be
        several turns back, because a visitor can ask "what programs do
        you offer?", follow up on one of them, ask about fees and
        admissions, and only then say "what about the second program?".
        Looking only at the immediately previous answer lost the list at
        that point and fell through to keyword search, which picked an
        arbitrary program."""
        rows = database.get_recent_messages(session_id, limit=config.MAX_HISTORY_MESSAGES)
        last_matched: List[str] = []
        list_ids: List[str] = []
        for row in reversed(rows):
            if row["role"] != "assistant" or not row["matched_entry_ids"]:
                continue
            ids = [i for i in row["matched_entry_ids"].split(",") if i]
            if not ids:
                continue
            if not last_matched:
                last_matched = ids
            if not list_ids:
                for entry_id in ids:
                    entry = self.entries_by_id.get(entry_id)
                    if entry and entry.list_ids:
                        list_ids = entry.list_ids
                        break
            if last_matched and list_ids:
                break
        return list_ids, last_matched

    def _last_assistant_intent(self, session_id: str) -> Optional[str]:
        rows = database.get_recent_messages(session_id, limit=config.MAX_HISTORY_MESSAGES)
        for row in reversed(rows):
            if row["role"] == "assistant":
                return row["intent"]
        return None

    # ---- answer generation --------------------------------------------------
    def _format_entry_text(self, entry: KBEntry) -> str:
        """Render a single entry, using bullets/numbered steps when the
        entry is flagged for it so list-like answers ('what programs do
        you offer', 'how do I apply') read as structured points rather
        than a run-on sentence."""
        answer = entry.answer.strip()
        if entry.format == "bullets" and entry.items:
            bullets = "\n".join(f"- {item}" for item in entry.items)
            return f"{answer}\n{bullets}" if answer else bullets
        if entry.format == "steps" and entry.items:
            steps = "\n".join(f"{i}. {item}" for i, item in enumerate(entry.items, start=1))
            return f"{answer}\n{steps}" if answer else steps
        return answer

    def _format_template_answer(self, scored: List[ScoredEntry]) -> str:
        parts = []
        for item in scored:
            entry = item.entry
            text = self._format_entry_text(entry)
            if text and text not in parts:
                parts.append(text)
        return "\n\n".join(parts)

    def _recent_turns(self, session_id: str) -> List[dict]:
        """Recent turns as {role, content}, oldest first, for the model to
        resolve follow-ups against. Trimmed to LLM_HISTORY_TURNS so long
        conversations don't grow the request without bound."""
        rows = database.get_recent_messages(session_id, limit=config.MAX_HISTORY_MESSAGES)
        turns = [{"role": r["role"], "content": r["content"]} for r in rows]
        return turns[-config.LLM_HISTORY_TURNS:] if config.LLM_HISTORY_TURNS > 0 else []

    def _generate_answer(self, user_message: str, scored: List[ScoredEntry], session_id: str) -> str:
        if config.LLM_ENABLED:
            answer = llm.generate_answer(user_message, scored, self._recent_turns(session_id))
            if answer:
                return answer
            # LLM failed or timed out -> fall through to the safe template path.
        return self._format_template_answer(scored)

    def _match_programs_mentioned(self, text: str) -> List[KBEntry]:
        lowered = text.lower()
        matches: List[KBEntry] = []
        for entry_id, hints in _PROGRAM_NAME_HINTS.items():
            if any(hint in lowered for hint in hints):
                entry = self.entries_by_id.get(entry_id)
                if entry:
                    matches.append(entry)
        return matches

    def _handle_comparison(self, text: str) -> ReplyResult:
        matched = self._match_programs_mentioned(text)
        if len(matched) < 2:
            overview = self.entries_by_id.get("programs-overview")
            items = ", ".join(overview.items) if overview else ""
            prompt = (
                "I can compare our programs for you — which two would you like to see side by side? "
                "We offer Foundation Years, Middle School, Senior School Focus, and Confident Speaker."
            )
            return ReplyResult(prompt, "clarify", [], None)

        a, b = matched[0], matched[1]
        text_out = (
            f"{a.question.replace('What is the ', '').rstrip('?')}:\n{a.answer}\n\n"
            f"{b.question.replace('What is the ', '').rstrip('?')}:\n{b.answer}\n\n"
            "Let me know if you would like help deciding which fits better for the student."
        )
        return ReplyResult(text_out, "comparison", [a.id, b.id], None)

    def _generate_general_answer(self, user_message: str, session_id: str) -> Optional[str]:
        """For messages that don't match anything in the knowledge base:
        ask the LLM for a brief, honest, non-WeMentors-specific reply
        (persona rule "answer type 2") rather than the canned redirect.
        Returns None if the LLM is disabled or fails, so the caller can
        fall back to OFF_TOPIC_RESPONSE — general questions should get a
        real answer when possible, never a fabricated WeMentors fact."""
        if not config.LLM_ENABLED:
            return None
        return llm.generate_answer(user_message, [], self._recent_turns(session_id))

    # ---- main entry point ---------------------------------------------------
    def handle_message(self, session_id: str, message: str) -> ReplyResult:
        message = message.strip()

        if len(message) > 2000:
            return ReplyResult(personality.TOO_LONG_MESSAGE_RESPONSE, "too_long", [], None)

        intent = self.detect_intent(message)

        last_intent = self._last_assistant_intent(session_id)

        if intent == "help":
            return ReplyResult(personality.HELP_RESPONSE, intent, [], None)
        if intent == "advising":
            return ReplyResult(personality.ADVISING_RESPONSE, intent, [], None)
        if intent == "greeting":
            return ReplyResult(personality.pick(personality.GREETINGS), intent, [], None)
        if intent == "goodbye":
            return ReplyResult(personality.pick(personality.GOODBYE_RESPONSES), intent, [], None)
        if intent == "thanks":
            return ReplyResult(personality.pick(personality.THANKS_RESPONSES), intent, [], None)
        if intent == "injection_attempt":
            return ReplyResult(personality.INJECTION_DEFLECTION, intent, [], None)
        if intent == "frustrated":
            return ReplyResult(personality.pick(personality.FRUSTRATED_RESPONSES), intent, [], None)
        if intent == "confused":
            return ReplyResult(personality.pick(personality.CONFUSED_RESPONSES), intent, [], None)
        if intent == "comparison":
            return self._handle_comparison(message)
        if intent == "demo_booking":
            return ReplyResult(
                personality.DEMO_BOOKING_RESPONSE,
                "demo_booking",
                ["contact-info", "how-to-book-demo"],
                1.0,
            )

        last_list_ids, last_matched_ids = self._get_last_turn_context(session_id)

        # Handle vague follow-up or vague questions when there is no prior context
        if _VAGUE_MORE_RE.match(message) and not last_matched_ids and not last_list_ids:
            return ReplyResult(personality.MORE_INFO_CLARIFICATION, "clarify", [], None)
        if _VAGUE_COST_RE.match(message) and not last_matched_ids:
            return ReplyResult(personality.HOW_MUCH_CLARIFICATION, "clarify", [], None)

        # Ordinal references ("the first one") are unambiguous and always win.
        referenced_entry = self._resolve_ordinal_reference(message, last_list_ids, last_matched_ids)

        direct_scored = self.retriever.search(message, top_k=config.RETRIEVAL_TOP_K)
        direct_top_score = direct_scored[0].score if direct_scored else 0.0

        # Pronoun references ("it", "that", "tell me more") only take over
        # when direct retrieval didn't already find something specific.
        if referenced_entry is None and direct_top_score < 0.3:
            referenced_entry = self._resolve_pronoun_reference(message, last_matched_ids)

        if referenced_entry is not None:
            scored = [ScoredEntry(entry=referenced_entry, score=1.0)]
        else:
            scored = direct_scored

        if not scored:
            if self._is_reference_query(message):
                return ReplyResult(personality.CLARIFY_NO_PRIOR_CONTEXT, "clarify", [], None)
            general_answer = self._generate_general_answer(message, session_id)
            if general_answer:
                return ReplyResult(general_answer, "general", [], None)
            return ReplyResult(personality.OFF_TOPIC_RESPONSE, "off_topic", [], None)

        top_score = scored[0].score
        if referenced_entry is None and top_score < config.RETRIEVAL_CONFIDENCE_THRESHOLD:
            # Low-confidence match: do not force-feed weak matches into the LLM as facts.
            general_answer = self._generate_general_answer(message, session_id)
            if general_answer:
                return ReplyResult(general_answer, "general", [], None)
            return ReplyResult(personality.FALLBACK_RESPONSE, "low_confidence", [], top_score)

        # Keep only the single best match once we're past the threshold —
        # never dump the whole knowledge base into one reply.
        best = scored[:1] if referenced_entry is None else scored
        answer = self._generate_answer(message, best, session_id)
        matched_ids = [item.entry.id for item in best]
        return ReplyResult(answer, "faq", matched_ids, top_score)
