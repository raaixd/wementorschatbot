"""Last-step wording and fallback helpers for visitor-facing replies."""

from __future__ import annotations

import re
from typing import Optional

_NOT_PUBLISHED_RE = re.compile(
    r"\bnot published in (?:our|my) (?:verified )?(?:records|knowledge base)\b", re.IGNORECASE
)
_INTERNAL_SOURCE_RE = re.compile(
    r"\bin (?:our|my) (?:verified )?(?:records|knowledge base)\b", re.IGNORECASE
)


def hide_internal_terms(reply: str) -> str:
    """Visitors should not see how answers are stored. Replacing, not just
    deleting, keeps "no entrance exam in our knowledge base" from turning
    into a flat claim that there is no exam."""
    reply = _NOT_PUBLISHED_RE.sub("not published", reply)
    return _INTERNAL_SOURCE_RE.sub("in the information we have published", reply)


_FORMERLY_RE = re.compile(r"\s*\((?:formerly\s+)?Confident Speaker\)")
_OLD_NAME_RE = re.compile(r"\bConfident Speaker\b")


def use_current_program_name(question: str, reply: str) -> str:
    """The website calls the course Communicative English. Parents who use the
    old name still see it, so they can tell it is the same course."""
    if "confident speaker" in question.lower():
        return reply
    return _OLD_NAME_RE.sub("Communicative English", _FORMERLY_RE.sub("", reply))


# A grade counts only when it is clearly a school grade, so that "on the 7th"
# or "5 classes" are ignored.
_GRADE_RE = re.compile(
    r"\b(?:grade|class|clas|std|standard)\.?\s*(?P<after>10|[3-9])\b(?!\s*(?:days?|times|per|a\s+week))"
    r"|\b(?P<before>10|[3-9])\s*(?:th|rd)\s*(?:grade|class|clas|std|standard|grader)\b"
    r"|\b(?:kid|child|son|daughter|he|she)(?:\s+(?:is|studies|studying))?\s+in\s+(?P<child>10|[3-9])\s*(?:th|rd)\b",
    re.IGNORECASE,
)


def stated_grade(message: str) -> Optional[int]:
    """The single Grade 3-10 the visitor states, or None if absent or unclear."""
    found = {
        int(m.group("after") or m.group("before") or m.group("child"))
        for m in _GRADE_RE.finditer(message)
    }
    return found.pop() if len(found) == 1 else None


_PROGRAMS = (
    (range(3, 6), "Foundation Years (Grades 3–5)",
     "curiosity-first teaching, concept games and visual learning, and weekly progress notes for parents"),
    (range(6, 9), "Middle School (Grades 6–8)",
     "concept-first, practical problem sets, fortnightly doubt clinics, and a progress dashboard"),
    (range(9, 11), "Senior School Focus (Grades 9–10)",
     "board-exam coaching in Maths and Science, weekly mock tests with review, and priority doubt-clearing"),
)


def grade_fallback(grade: int) -> str:
    """What we can truthfully say when only the grade is understood."""
    name, features = next((n, f) for grades, n, f in _PROGRAMS if grade in grades)
    return (
        f"For Grade {grade}, our program is **{name}**, which includes {features}.\n\n"
        "I may not have answered your exact question. For that, contact the team at "
        "**admin@wementors.co** or **+91 76111 92227**, or use **Book Free Demo**."
    )
