"""Hard rules applied before routing or any language model.

These are topics where a guessed answer can hurt a family: the model has told
a parent the academic mentor is a woman, and implied support for ADHD, with
nothing published to back either. They get a fixed reply instead.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_CONTACT = "Please ask the team directly: **admin@wementors.co** or **+91 76111 92227** (call or WhatsApp)."

_ADDITIONAL_NEEDS_RE = re.compile(
    r"\b(adhd|autis\w*|asperger\w*|dyslexi\w*|dyscalculia|dysgraphia|dyspraxia|down'?s?\s+syndrome|"
    r"special\s+(?:needs|educational)|disabilit\w*|disabled|learning\s+(?:difficult\w*|disorder\w*)|"
    r"hearing\s+(?:impair\w*|loss)|visual(?:ly)?\s+impair\w*|deaf|speech\s+(?:delay|disorder)|stammer\w*|stutter\w*)\b",
    re.IGNORECASE,
)
_CHILD_SAFETY_RE = re.compile(
    r"\b(background[\s-]*(?:check\w*|verif\w*)|police\s+(?:check\w*|verif\w*)|vetted|vetting|safeguard\w*|"
    r"child(?:'s|ren'?s?)?\s+safety|safety\s+of\s+(?:my\s+|the\s+)?(?:child|kid|children|son|daughter)|"
    r"safe\s+for\s+(?:my\s+|the\s+)?(?:child|kid|children|son|daughter))\b",
    re.IGNORECASE,
)
_PRIVACY_RE = re.compile(
    r"\bprivacy\s+(?:policy|of)\b|"
    r"\b(?:data|personal\s+(?:info\w*|details))\b[^.?!]{0,30}\b(?:safe|secure|private|shared?|sold|protected?)\b|"
    r"\b(?:share|sell|protect)\b[^.?!]{0,20}\b(?:my|our|child'?s)\s+(?:data|personal)\b",
    re.IGNORECASE,
)
_PERSON = r"(?:mentor|teacher|tutor|faculty)s?"
_GENDER = r"(?:female|male|woman|women|man|men|lady|ladies|gents?)"
_MENTOR_ATTRIBUTE_RE = re.compile(
    rf"\b{_GENDER}\s+{_PERSON}\b|\b{_PERSON}\b[^.?!]{{0,20}}\b(?:a\s+)?{_GENDER}\b|"
    rf"\bgender\b[^.?!]{{0,20}}\b{_PERSON}\b|\b{_PERSON}'?s?\s+(?:gender|religion|age)\b",
    re.IGNORECASE,
)
_PROMISE_RE = re.compile(r"\b(guarantee[sd]?|promise[sd]?|assure[sd]?|assurance)\b", re.IGNORECASE)
_OUTCOME_RE = re.compile(
    r"\b(marks?|grades?|ranks?|scores?|results?|band|pass(?:ing)?|admissions?|selection|toppers?|fluen\w+)\b|\d\s*%",
    re.IGNORECASE,
)

NEEDS_REPLY = (
    "I don’t have confirmed information about support for specific learning, developmental or physical "
    "needs, and this is too important to guess. " + _CONTACT
)
SAFETY_REPLY = (
    "I don’t have confirmed information about mentor background checks or child-safety procedures, "
    "and this is too important to guess. " + _CONTACT
)
PRIVACY_REPLY = (
    "I don’t have confirmed information about how personal data is stored or shared. " + _CONTACT
)
ATTRIBUTE_REPLY = "I can’t confirm personal details about which mentor would be assigned. " + _CONTACT
NO_GUARANTEE_REPLY = (
    "WeMentors does not guarantee marks, ranks, exam scores, band scores or admissions. Mentors focus on "
    "concept clarity, regular practice and individual progress tracking.\n\n"
    "You can click **Book Free Demo** at the top-right of the website to explore the mentoring approach."
)


def policy_reply(message: str) -> Optional[Tuple[str, str]]:
    """Return (intent, reply) when a fixed rule applies, else None."""
    if _ADDITIONAL_NEEDS_RE.search(message):
        return "additional_needs", NEEDS_REPLY
    if _CHILD_SAFETY_RE.search(message):
        return "child_safety", SAFETY_REPLY
    if _PRIVACY_RE.search(message):
        return "data_privacy", PRIVACY_REPLY
    if _MENTOR_ATTRIBUTE_RE.search(message):
        return "mentor_attribute", ATTRIBUTE_REPLY
    if _PROMISE_RE.search(message) and _OUTCOME_RE.search(message):
        return "marks_guarantee", NO_GUARANTEE_REPLY
    return None


# "I don't want a demo, just tell me the price" is a price question. Without
# this, keyword routing sees "demo" and answers the thing just turned down.
_ITEM = r"[\w' -]{1,30}?"
_DECLINED_RES = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"^\s*(?:(?:i|we)\s+)?(?:don[’']?t|do\s+not)\s+(?:want|need)\s+{_ITEM}[,.;]\s*",
        rf"^\s*no\s+need\s+(?:for|of)\s+{_ITEM}[,.;]\s*",
        rf"^\s*{_ITEM}\s+is\s+not\s+what\s+(?:i|we)(?:[’']m|\s+am|[’']re|\s+are)?\s+(?:want|need|looking\s+for)\s*[,.;]\s*",
    )
)
_FILLER_RE = re.compile(r"^(?:just|but|only)\s+", re.IGNORECASE)


def without_declined_clause(message: str) -> str:
    """Drop a leading "I don't want X," when a real question follows."""
    for pattern in _DECLINED_RES:
        match = pattern.match(message)
        if match:
            rest = _FILLER_RE.sub("", message[match.end():].strip())
            if len(rest.split()) >= 3:
                return rest
    return message
