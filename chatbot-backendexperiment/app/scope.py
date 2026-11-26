"""Decides whether a mentor request may be answered with a "yes".

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

_ALLOWED = frozenset((_PHRASING + _OFFERED).split())
# 3-10 are the supported grades; 1 covers "1:1" and "1 on 1".
_ALLOWED_NUMBERS = frozenset(["1", "3", "4", "5", "6", "7", "8", "9", "10"])
_TOKEN_RE = re.compile(r"\d+|[a-z]+(?:'[a-z]+)?")


def mentor_request_in_scope(message: str) -> bool:
    text = message.lower().replace("’", "'")
    # "7th" -> "7 th"; any non-Latin letters cannot be checked.
    if any(ch.isalpha() and not ch.isascii() for ch in text):
        return False
    return all(
        token in _ALLOWED_NUMBERS if token.isdigit() else token in _ALLOWED
        for token in _TOKEN_RE.findall(text)
    )
