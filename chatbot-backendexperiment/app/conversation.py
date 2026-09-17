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

from . import config, database, leads, llm, personality
from .knowledge import KBEntry
from .retrieval import Retriever, ScoredEntry, tokenize, _FEE_TRIGGER_WORDS

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

_SHORT_CONFUSION_RE = re.compile(
    r"^\s*(what\??|huh\??|pardon\??|what do you mean\??|i don'?t understand\??|sorry\??)\s*$",
    re.IGNORECASE,
)
_BEGINNER_RE = re.compile(
    r"\b(?:what|which)\s+(?:course|class|program)\s+is\s+good\s+for\s+beginners?\b"
    r"|\bbeginner\s+(?:course|class|program)s?\b"
    r"|\b(?:courses?|classes?|programs?)\s+for\s+beginners?\b"
    r"|\bwhat\s+should\s+a\s+beginner\s+(?:take|learn|start\s+with)\b",
    re.IGNORECASE,
)
_DEMO_CLASS_EXACT_RE = re.compile(
    r"^\s*(?:free\s+)?demo\s+class(?:es)?\s*[?!.]*$",
    re.IGNORECASE,
)
_BOOK_ENROLL_RE = re.compile(
    r"^\s*(?:book\s+enroll|enroll\s+book|book\s+and\s+enroll|how\s+to\s+book\s+and\s+enroll|book\s+or\s+enroll|enroll\s+or\s+book)\s*[?!.]*$",
    re.IGNORECASE,
)
_PYTHON_COURSE_RE = re.compile(
    r"\b(python|coding|programming|java|c\+\+|robotics|artificial intelligence|machine learning|web development|app development)\b",
    re.IGNORECASE,
)
_DEMO_OFFER_AFFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|sure|yes\s+please|please|definitely|i\s+would|i'?d\s+love\s+to)\s*[!.]*$",
    re.IGNORECASE,
)
_OUT_OF_SCOPE_RE = re.compile(
    r"\b(football|cricket|sports|weather|temperature|poem|poetry|song|lyrics|recipe|cook|pizza|burger|"
    r"movie|cinema|actor|president|prime minister|politics|election|joke)\b",
    re.IGNORECASE,
)

_HELP_RE = re.compile(
    r"^\s*(help|help me|can you help( me)?|i need help|please help|support|assist(ance)?)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_CAPABILITY_RE = re.compile(
    r"^\s*(what\s+(?:can\s+)?(?:you|u)\s+help\s+(?:me\s+)?with\??|"
    r"what\s+do\s+(?:you|u)\s+know\??|"
    r"what\s+(?:information|info)\s+do\s+(?:you|u)\s+have\??|"
    r"tell\s+me\s+about\s+wementors\??|"
    r"what\s+is\s+wementors\??|"
    r"how\s+can\s+(?:you|u)\s+help\??|"
    r"what\s+can\s+i\s+ask\s+(?:you|u)\??|"
    r"can\s+(?:you|u)\s+help\s+me\??|"
    r"can\s+(?:you|u)\s+help\??|"
    r"i\s+need\s+information\??|"
    r"what\s+are\s+your\s+capabilities\??|"
    r"what\s+can\s+(?:you|u)\s+do\??)\s*$",
    re.IGNORECASE,
)

_VAGUE_INFO_RE = re.compile(
    r"^\s*(information|info|details)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_HOW_TO_BOOK_RE = re.compile(
    r"^\s*(how\s+(?:can|do)\s+i\s+book(?:\s+(?:a\s+)?demo)?\??|how\s+to\s+book(?:\s+(?:a\s+)?demo)?\??)\s*$",
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
    r"\b(book|schedule|sign( me)? up|register|enroll( me)?|start|get|have|try|take|request|arrange|attend)\b.*\b(demo|trial)\b"
    r"|\b(want|need|like|interested in)\b.*\b(demo|trial)\b"
    r"|\b(demo|trial)\s+class\s+booking\b|\bbook( a)? (demo|trial)\b"
    r"|\bhow (can|do) i (book|get|attend|schedule) a (demo|trial)\b"
    r"|\b(can|could) i (get|have|book|attend) a (free\s+)?(demo|trial)\b"
    r"|\b(how to join|how do i join|want to join|enquire about joining|interested in joining)\b"
    r"|\b(how do i enroll|how to enroll|admissions? process|admission enquiry|enquire about classes)\b",
    re.IGNORECASE,
)
_WHERE_DETAILS_RE = re.compile(
    r"\b(where\s+(?:do|can)\s+i\s+(?:enter|fill|put|submit|register|type)\s+(?:my\s+)?(?:details|info|information|name|form)|where\s+to\s+(?:enter|fill|put|submit|register)\s+(?:my\s+)?(?:details|info|information)|where\s+can\s+i\s+register(?:\s+for\s+(?:a\s+)?demo)?|where\s+do\s+i\s+register(?:\s+for\s+(?:a\s+)?demo)?|where\s+is\s+the\s+(?:book\s+free\s+demo\s+)?(?:form|button|link|option))\b",
    re.IGNORECASE,
)
_CONTACT_REQUEST_RE = re.compile(
    r"\b((?:want|can|could|would\s+like)\s+(?:someone|somebody|the\s+team)\s+(?:from\s+wementors\s+)?(?:to\s+)?contact\s+me|call\s+me\s+back|have\s+someone\s+call\s+me)\b",
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

_PLAYFUL_RE = re.compile(
    r"\b(cat|cats|dog|dogs|pet|pets|kitten|puppy|alien|aliens|spaceship|spaceships|toaster|purple|banana|quantum toaster)\b",
    re.IGNORECASE,
)


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
        if _SHORT_CONFUSION_RE.match(stripped):
            return "confused"
        if _BOOK_ENROLL_RE.match(stripped):
            return "book_enroll"
        if _CAPABILITY_RE.match(stripped):
            return "capability"
        if _VAGUE_INFO_RE.match(stripped):
            return "vague_info"
        if _HOW_TO_BOOK_RE.match(stripped):
            return "demo_booking"
        if _DEMO_CLASS_EXACT_RE.match(stripped):
            return "demo_inquiry"
        if _BEGINNER_RE.search(stripped):
            return "beginner_recommendation"
        if _PYTHON_COURSE_RE.search(stripped):
            return "unverified_course"
        if _OUT_OF_SCOPE_RE.search(stripped) and not any(k in stripped.lower() for k in ["class", "mentor", "course", "subject", "demo", "wementors"]):
            return "off_topic"
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
        if _WHERE_DETAILS_RE.search(stripped):
            return "where_details"
        if _CONTACT_REQUEST_RE.search(stripped):
            return "contact_request"
        if _DEMO_BOOKING_RE.search(stripped):
            remainder = re.sub(r"\b(free\s+)?(demo|trial)\s+class(es)?\b", "", stripped, flags=re.IGNORECASE)
            remainder = re.sub(r"\b(demo|trial)\b", "", remainder, flags=re.IGNORECASE)
            if re.search(r"\b(subject|course|program|curriculum|grade|teach|learn|batch|online)\w*\b", remainder, re.IGNORECASE):
                return "multi_intent"
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

    def _last_assistant_message(self, session_id: str) -> Optional[str]:
        rows = database.get_recent_messages(session_id, limit=config.MAX_HISTORY_MESSAGES)
        for row in reversed(rows):
            if row["role"] == "assistant":
                return row["content"]
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
            turns = self._recent_turns(session_id)
            has_fee_word = bool(
                set(tokenize(user_message)) & _FEE_TRIGGER_WORDS
                or re.search(r"\b(fee|fees|cost|costs|price|prices|pricing|charge|rate|how much)\b", user_message, re.IGNORECASE)
            )
            is_course_query = any(item.entry.id.startswith(("program-", "curriculum-", "programs-overview")) for item in scored)
            if is_course_query and not has_fee_word:
                turns = [
                    t for t in turns
                    if not re.search(r"\b(fee|fees|pricing|cost|costs|price|prices|charge|scholarship)\b", t.get("content", ""), re.IGNORECASE)
                ]
            answer = llm.generate_answer(user_message, scored, turns)
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

        # Greetings always take priority: never treated as names or demo trigger
        if _GREETING_START_RE.match(message) and len(message.split()) <= 4:
            return ReplyResult(personality.pick(personality.GREETINGS), "greeting", [], None)

        # Short confusion ("What?", "huh?", "pardon?")
        if _SHORT_CONFUSION_RE.match(message):
            return ReplyResult(personality.CONFUSION_CLARIFICATION_RESPONSE, "confused", [], None)

        # Exact "Book enroll"
        if _BOOK_ENROLL_RE.match(message):
            return ReplyResult(personality.BOOK_ENROLL_RESPONSE, "book_enroll", ["how-to-book-demo", "how-to-apply"], 1.0)

        # General chatbot capability & broad scope questions ("what can u help me with", "what do you know", etc.)
        if _CAPABILITY_RE.match(message):
            return ReplyResult(personality.CAPABILITY_RESPONSE, "capability", ["programs-overview", "how-to-book-demo"], 1.0)

        # Single-word vague information request ("information", "info", "details")
        if _VAGUE_INFO_RE.match(message):
            return ReplyResult(personality.VAGUE_INFO_CLARIFICATION, "vague_info", ["programs-overview", "how-to-book-demo"], 1.0)

        # How to book a demo ("how do i book", "how to book a demo")
        if _HOW_TO_BOOK_RE.match(message):
            return ReplyResult(personality.HOW_TO_BOOK_RESPONSE, "demo_booking", ["how-to-book-demo"], 1.0)

        # Exact "Demo class"
        if _DEMO_CLASS_EXACT_RE.match(message):
            return ReplyResult(personality.DEMO_CLASS_RESPONSE, "demo_inquiry", ["demo-class-available", "how-to-book-demo"], 1.0)

        # Beginner recommendations ("What course is good for beginners?")
        if _BEGINNER_RE.search(message):
            if not re.search(r"\b(grade\s*\d+|class\s*\d+|\d+th\s*(grade|class|standard)?|math|science|english|speaker)\b", message, re.IGNORECASE):
                return ReplyResult(personality.BEGINNER_RECOMMENDATION_RESPONSE, "beginner_recommendation", ["programs-overview"], 1.0)

        # Unverified courses (Python, Coding, Programming, etc.)
        if _PYTHON_COURSE_RE.search(message):
            has_fee_word = bool(
                set(tokenize(message)) & _FEE_TRIGGER_WORDS
                or re.search(r"\b(fee|fees|cost|costs|price|prices|pricing|charge|rate|how much)\b", message, re.IGNORECASE)
            )
            if has_fee_word:
                return ReplyResult(personality.UNVERIFIED_PYTHON_FEES_RESPONSE, "unverified_course_fees", ["fees-and-pricing", "how-to-book-demo"], 1.0)
            return ReplyResult(personality.UNVERIFIED_PYTHON_COURSE_RESPONSE, "unverified_course", ["programs-overview", "how-to-book-demo"], 1.0)

        # Context-aware follow-up: User replies "Yes" to an offer to book a demo
        last_assistant_msg = self._last_assistant_message(session_id)
        if _DEMO_OFFER_AFFIRM_RE.match(message) and last_assistant_msg:
            if re.search(r"\b(would you like to book|want to book|arrange a (free )?demo|book a (free )?demo)\b", last_assistant_msg, re.IGNORECASE):
                return ReplyResult(
                    personality.DEMO_OFFER_YES_RESPONSE,
                    "demo_booking",
                    ["how-to-book-demo"],
                    1.0,
                )

        # Context-aware follow-up: Assistant asked for name and user replied "Okay" / "Sure"
        if leads.is_pure_acknowledgement(message) and last_assistant_msg:
            if re.search(r"\b(what name|your name|student or parent name|name should i use)\b", last_assistant_msg, re.IGNORECASE):
                return ReplyResult(
                    personality.ASK_NAME_AGAIN_RESPONSE,
                    "demo_acknowledgement",
                    ["contact-info"],
                    None,
                )

        # Out-of-scope questions
        if _OUT_OF_SCOPE_RE.search(message) and not any(k in message.lower() for k in ["class", "mentor", "course", "subject", "demo", "wementors"]):
            return ReplyResult(personality.OFF_TOPIC_RESPONSE, "off_topic", [], None)

        # Handle casual 'never mind' / cancellation when not in active demo flow
        if re.search(r"^\s*(never\s*mind|nevermind|no\s*worries|no\s*thanks|forget\s*it)\s*[!.]*\s*$", message, re.IGNORECASE):
            existing_lead_data = database.get_demo_lead(session_id)
            if existing_lead_data and existing_lead_data.get("stage") in ("collecting", "confirming"):
                reply, resolved_intent, updated_lead = leads.process_demo_flow(session_id, message, leads.DemoLead.from_dict(existing_lead_data))
                return ReplyResult(reply, resolved_intent, ["contact-info"], None)
            return ReplyResult("No problem at all! Feel free to ask anytime if you have questions about WeMentors courses, curriculum, or scheduling a demo.", "never_mind", [], None)

        # Handle playful / humor queries (e.g. cat calculus)
        if _PLAYFUL_RE.search(message):
            if config.LLM_ENABLED:
                gen = self._generate_general_answer(message, session_id)
                if gen:
                    return ReplyResult(gen, "general", [], None)
            return ReplyResult(
                "While our mentors would love to help, we specialize in mentoring students for school subjects and confident speaking! Let me know if you'd like to learn about our courses for Grades 3–10.",
                "general",
                [],
                None,
            )

        # Check existing demo lead state for this session
        existing_lead_data = database.get_demo_lead(session_id)
        current_lead = leads.DemoLead.from_dict(existing_lead_data) if existing_lead_data else None

        # Check if user is asking whether their demo has been booked
        if leads.is_asking_if_booked(message):
            reply, resolved_intent, updated_lead = leads.process_demo_flow(session_id, message, current_lead)
            return ReplyResult(reply, resolved_intent, ["contact-info", "how-to-book-demo"], None)

        # If user is in an active demo collection or confirmation flow
        if current_lead and current_lead.stage in ("collecting", "confirming"):
            if leads.is_cancellation(message):
                reply, resolved_intent, updated_lead = leads.process_demo_flow(session_id, message, current_lead)
                return ReplyResult(reply, resolved_intent, ["contact-info"], None)

            if leads.is_pure_acknowledgement(message):
                reply, resolved_intent, updated_lead = leads.process_demo_flow(session_id, message, current_lead)
                return ReplyResult(reply, resolved_intent, [], None)

            extracted_lead, found_fields = leads.extract_lead_fields(message, current_lead)
            if found_fields or current_lead.stage == "confirming":
                reply, resolved_intent, updated_lead = leads.process_demo_flow(session_id, message, current_lead)
                return ReplyResult(reply, resolved_intent, ["contact-info", "how-to-book-demo"], None)

        intent = self.detect_intent(message)

        last_intent = self._last_assistant_intent(session_id)

        if intent == "help":
            return ReplyResult(personality.HELP_RESPONSE, intent, [], None)
        if intent == "advising":
            if re.search(r"\b(grade\s*\d+|class\s*\d+|\d+th\s*(grade|class|standard)?|math|science|english|speaker|middle|senior|foundation)\b", message, re.IGNORECASE):
                pass  # Fall through to conversational RAG advising with specific grade context
            else:
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
            if config.LLM_ENABLED:
                gen = self._generate_general_answer(message, session_id)
                if gen:
                    return ReplyResult(gen, "frustrated", [], None)
            return ReplyResult(personality.pick(personality.FRUSTRATED_RESPONSES), intent, [], None)
        if intent == "confused":
            if config.LLM_ENABLED:
                gen = self._generate_general_answer(message, session_id)
                if gen:
                    return ReplyResult(gen, "confused", [], None)
            return ReplyResult(personality.pick(personality.CONFUSED_RESPONSES), intent, [], None)
        if intent == "comparison":
            return self._handle_comparison(message)
        if intent == "where_details":
            return ReplyResult(
                personality.WHERE_DETAILS_RESPONSE,
                "where_details",
                ["how-to-book-demo", "contact-info"],
                1.0,
            )
        if intent == "contact_request":
            return ReplyResult(
                personality.CONTACT_REQUEST_RESPONSE,
                "contact_request",
                ["contact-info", "how-to-book-demo"],
                1.0,
            )
        if intent == "demo_booking":
            extracted_lead, found_fields = leads.extract_lead_fields(message)
            if found_fields:
                reply, resolved_intent, updated_lead = leads.process_demo_flow(session_id, message, extracted_lead)
                return ReplyResult(reply, resolved_intent, ["contact-info", "how-to-book-demo"], 1.0)

            # Start collecting stage in database and provide official contact details
            database.save_demo_lead(session_id, stage="collecting")
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

        # Check for multi-clause or multi-intent questions (e.g. subjects and how to join)
        clauses = [c.strip() for c in re.split(r"\band\b|[?!;]|\balso\b", message) if len(c.strip().split()) >= 2]
        multi_scored: List[ScoredEntry] = []
        if len(clauses) > 1 and referenced_entry is None:
            seen_ids = set()
            for clause in clauses:
                sub_res = self.retriever.search(clause, top_k=2)
                for item in sub_res:
                    if item.entry.id not in seen_ids and item.score >= config.RETRIEVAL_CONFIDENCE_THRESHOLD:
                        seen_ids.add(item.entry.id)
                        multi_scored.append(item)

        if referenced_entry is not None:
            scored = [ScoredEntry(entry=referenced_entry, score=1.0)]
        elif len(multi_scored) >= 2:
            scored = multi_scored[:3]
        else:
            scored = direct_scored

        if not scored:
            if self._is_reference_query(message):
                if not self._recent_turns(session_id):
                    return ReplyResult(personality.CLARIFY_NO_PRIOR_CONTEXT, "clarify", [], None)
                general_answer = self._generate_general_answer(message, session_id)
                if general_answer:
                    return ReplyResult(general_answer, "general", [], None)
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
            return ReplyResult(personality.LOW_CONFIDENCE_FALLBACK, "low_confidence", [], top_score)

        # For multi-clause questions, keep the multiple matched entries; otherwise keep the best single match.
        if referenced_entry is not None:
            best = scored
        elif len(multi_scored) >= 2:
            best = scored
        else:
            best = scored[:1]

        answer = self._generate_answer(message, best, session_id)
        matched_ids = [item.entry.id for item in best]
        resolved_intent = "multi_intent" if len(multi_scored) >= 2 else ("faq" if intent == "faq" else intent)
        return ReplyResult(answer, resolved_intent, matched_ids, top_score)
