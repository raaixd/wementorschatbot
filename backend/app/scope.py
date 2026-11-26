"""Decides whether a question may be answered with a "yes".

This is an allowlist: a request is in scope only when every word is either
ordinary phrasing or something WeMentors publishes. Unknown words (another
subject, a city, a language, a mentor attribute, a time) mean the bot must
not confirm. A wrong "can't confirm" is acceptable; a false "yes" is not.
"""

from __future__ import annotations

import re

# How people ask, with no claim about what is being asked for.
_PHRASING = """
a about again also am an and any are as at available avail be both can could
dedicated do does each find for get give has have having hello help her here hey
hi his how i i'd i'm if in interested is it kid kids like live looking match
allocate arrange assign book may me my need needed needing needs new of offer on
one online our personal personalised personalized please pls plz provide
provided provides want wanted wants we will with would you your u ur there some
the their them to someone child children son daughter student students learner
learners mentor mentors mentoring mentorship tutor tutors tutoring teacher
teachers coach coaching class classes teaching teach session sessions support
study studies studying learn learning weak struggling improve extra private
individual
"""

# What WeMentors publishes: grades 3-10, subjects, boards, programs.
_OFFERED = """
grade grades std standard th rd st school primary foundation years middle senior
board boards curriculum curricula syllabus subject subjects academic academics
cbse icse igcse cambridge ib state
math maths mathematics algebra geometry science physics chemistry biology
english grammar literature evs environmental social studies history geography
civics spoken speaking public speaker confident communication communicative
conversation business interview skills ielts exam exams preparation prep
"""

# Wider wording for general questions (delivery, joining, programs, demo).
_GENERAL = """
all also anyone anywhere abroad adult adults college professional professionals
homemaker homemakers working business people person age ages from outside
country countries international globally worldwide india saudi arabia uae dubai
us usa uk join joining enroll enrol apply take taken attend offer offers offered
teach teaches taught supported include includes included cover covers covered supports
program programs programme course courses lesson lesson's lessons mode conducted
held happen happens daily weekly week many often how much long what which when
where does is are was be been or not no only just if so then that this it's
there's that's what's demo trial book free test tests mock practice general
academic 1:1 one-to-one group groups small batch batches size mentor's child's
kid's today tell more info information details know wondering whether
flexible flexibility timings timing schedule schedules choose google meet lms own use used virtual
other accept wementors prepare according following switch back really
"""

_ALLOWED = frozenset((_PHRASING + _OFFERED).split())
_ALLOWED_GENERAL = _ALLOWED | frozenset(_GENERAL.split())
# 3-10 are the supported grades; 1 covers "1:1" and "1 on 1".
_ALLOWED_NUMBERS = frozenset(["1", "3", "4", "5", "6", "7", "8", "9", "10"])
_TOKEN_RE = re.compile(r"\d+|[a-z]+(?:'[a-z]+)?")


def _covered(message: str, allowed: frozenset) -> bool:
    text = message.lower().replace("’", "'")
    # "7th" -> "7 th"; any non-Latin letters cannot be checked.
    if any(ch.isalpha() and not ch.isascii() for ch in text):
        return False
    return all(token in _ALLOWED_NUMBERS if token.isdigit() else token in allowed for token in _TOKEN_RE.findall(text))


def mentor_request_in_scope(message: str) -> bool:
    return _covered(message, _ALLOWED)


_INTRO_RE = re.compile(
    r"^\s*(?:my\s+name\s+is|name\s+is|i\s+am|i'm|call\s+me)\s+[a-z]{2,20}\s*[,.:;-]\s*", re.IGNORECASE
)
_LEADING_YES_RE = re.compile(r"^(?P<greeting>Nice to meet you, [^!]+! )?\**yes\b[.!,]?\**[,.!]?\s*", re.IGNORECASE)
CAVEAT = "I can’t confirm every detail of your question. Related published information:"


_ADULT_WORDS = frozenset("adult adults professional professionals homemaker homemakers businessperson".split())
_SCHOOL_SUBJECTS = frozenset(
    "math maths mathematics science physics chemistry biology evs history geography civics".split()
)


def _misleading_combination(question: str) -> bool:
    """Known words that are only true apart: the demo is free but classes are
    not published as free, and adults can join English but not school maths."""
    words = set(_TOKEN_RE.findall(question.lower()))
    if "free" in words and not words & {"demo", "trial"}:
        return True
    return bool(words & _ADULT_WORDS and words & _SCHOOL_SUBJECTS)


# Answers that read as a "yes" without saying it: a subject list in reply to
# "do you teach French?", or degrees in reply to "are mentors background checked?"
_IMPLIED_YES_ENTRIES = frozenset(["what-subjects-offered", "mentor-qualifications"])
_LISTING_WORDS = frozenset(
    """
qualification qualifications qualified credentials degree degrees education educated
experience experienced certified who which what are your tefl list kind kinds type types
university degrees hold holds background certification certifications guys
""".split()
)


def qualify_listing(question: str, reply: str, matched_ids: list) -> str:
    """Add the caveat when such a list answers a question it may not cover."""
    if not matched_ids or matched_ids[0] not in _IMPLIED_YES_ENTRIES or reply.startswith(CAVEAT):
        return reply
    if _covered(_INTRO_RE.sub("", question), _ALLOWED_GENERAL | _LISTING_WORDS):
        return reply
    return f"{CAVEAT}\n\n{reply}"


_EITHER_MODE_RE = re.compile(r"\b(?:online\s+or\s+offline|offline\s+or\s+online)\b", re.IGNORECASE)


def needs_caveat(question: str, answer: str, entry_ids: list) -> bool:
    """True when this answer would be shown under the caveat."""
    return qualify_affirmation(question, answer) != answer or qualify_listing(question, answer, entry_ids) != answer


def qualify_affirmation(question: str, reply: str) -> str:
    """Withdraw a leading "Yes" when the question mentions something unknown.

    Replies are chosen by keyword, so "do you offer online French classes"
    can reach the online-classes answer. The facts in it are still true; the
    "Yes" is not."""
    match = _LEADING_YES_RE.match(reply)
    if not match:
        return reply
    if _EITHER_MODE_RE.search(question):
        # "Online or offline?" is not a yes/no question; the facts answer it.
        rest = reply[match.end() :]
        return f"{match.group('greeting') or ''}{rest[:1].upper()}{rest[1:]}"
    asked = _INTRO_RE.sub("", question)
    if _covered(asked, _ALLOWED_GENERAL) and not _misleading_combination(asked):
        return reply
    rest = reply[match.end() :]
    return f"{match.group('greeting') or ''}{CAVEAT}\n\n{rest[:1].upper()}{rest[1:]}"
