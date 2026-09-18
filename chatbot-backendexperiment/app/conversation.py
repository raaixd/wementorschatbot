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

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from . import config, database, leads, llm, personality
from .knowledge import KBEntry

logger = logging.getLogger(__name__)
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
    r"movie|cinema|actor|president|prime minister|politics|election|joke|medical\s+advice|medicine|doctor|"
    r"write\s+(?:my\s+)?(?:assignment|essay|homework|code|paper)|do\s+my\s+homework)\b",
    re.IGNORECASE,
)

_HELP_RE = re.compile(
    r"^\s*(help|help me|can you help( me)?|i need help|please help|support|assist(ance)?)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_FORMAT_RE = re.compile(
    r"\b(what(?:'?s|\s+is)\s+(?:the\s+)?format|"
    r"program\s+format|"
    r"session\s+format|"
    r"how\s+does\s+(?:it|the\s+program|confident\s+speaker)\s+work|"
    r"how\s+are\s+(?:the\s+)?sessions\s+conducted|"
    r"what\s+happens\s+in\s+the\s+program|"
    r"what\s+is\s+the\s+structure(?:\s+of\s+the\s+program)?|"
    r"can\s+you\s+explain\s+the\s+program\s+format|"
    r"explain\s+the\s+program\s+format)\b"
    r"|^\s*format\s*[?!.]*$",
    re.IGNORECASE,
)

_AUDIENCE_RE = re.compile(
    r"\b(who\s+is\s+(?:it|this|the\s+program|confident\s+speaker)\s+for|"
    r"who\s+can\s+join|"
    r"is\s+(?:it|the\s+program|confident\s+speaker)\s+for\s+(?:adults|professionals|homemakers|students|kids|children)|"
    r"target\s+audience)\b",
    re.IGNORECASE,
)

_SCOPE_RE = re.compile(
    r"\b(what\s+does\s+(?:it|the\s+program|confident\s+speaker)\s+cover|"
    r"what\s+do\s+(?:we|students|learners)\s+learn|"
    r"what\s+is\s+covered(?:\s+in\s+the\s+program)?|"
    r"what\s+will\s+(?:my\s+child|i)\s+learn|"
    r"scope\s+of\s+(?:the\s+program|confident\s+speaker))\b",
    re.IGNORECASE,
)

_MENTORING_RE = re.compile(
    r"\b(is\s+(?:it|the\s+teaching|the\s+class|the\s+program)\s+(?:one[- ]on[- ]one|1[- ]on[- ]1|1:1)|"
    r"does\s+every\s+student\s+get\s+a\s+personal\s+mentor|"
    r"do\s+(?:you|they)\s+have\s+personal\s+mentors?|"
    r"how\s+does\s+(?:personalized\s+)?mentoring\s+work|"
    r"individual\s+attention|"
    r"personalized\s+mentoring|"
    r"personal\s+mentoring|"
    r"personal\s+mentors?|"
    r"mentoring)\b",
    re.IGNORECASE,
)

_SESSION_ACTIVITIES_RE = re.compile(
    r"\b("
    r"what\s+happens\s+(?:during|in)\s+(?:the\s+|practice\s+|class\s+|academic\s+)?sessions?|"
    r"what\s+about\s+(?:the\s+|academic\s+|confident\s+speaker\s+)?practice\s+sessions?|"
    r"what\s+about\s+(?:the\s+)?(?:sessions?|classes)|"
    r"what\s+do\s+(?:we|students|learners)\s+do\s+during\s+(?:the\s+|practice\s+|academic\s+)?sessions?|"
    r"what\s+will\s+my\s+child\s+do\s+in\s+(?:the\s+program|sessions?)|"
    r"session\s+activities|"
    r"practice\s+sessions?|"
    r"speaking\s+practice|"
    r"how\s+are\s+(?:the\s+)?(?:classes|sessions)\s+conducted|"
    r"how\s+are\s+(?:academic\s+)?subjects\s+taught|"
    r"what\s+happens\s+in\s+academic\s+sessions"
    r")\b",
    re.IGNORECASE,
)
_ACTIVITIES_RE = _SESSION_ACTIVITIES_RE

_ACADEMIC_COURSES_OVERVIEW_RE = re.compile(
    r"\b("
    r"academic\s+(?:courses?|classes|programs?|subjects?|mentoring|learning)|"
    r"classes\s+for\s+school\s+students|"
    r"courses\s+for\s+school\s+students|"
    r"what\s+about\s+academic\s+(?:courses?|classes|programs?|subjects?)|"
    r"tell\s+me\s+about\s+academic\s+(?:courses?|classes|programs?)|"
    r"academic\s+offerings?"
    r")\b",
    re.IGNORECASE,
)

_GRADE7_MATHS_SESSIONS_RE = re.compile(
    r"\b("
    r"grade\s*7\s+maths?\s+(?:practice\s+)?sessions?|"
    r"how\s+are\s+maths?\s+classes\s+conducted|"
    r"maths?\s+classes\s+conducted|"
    r"what\s+about\s+grade\s*7\s+maths?\s+sessions?|"
    r"practice\s+for\s+grade\s*7"
    r")\b",
    re.IGNORECASE,
)


_SPEAKING_PRACTICE_RE = re.compile(
    r"\b("
    r"speaking\s+practice|"
    r"public\s+speaking\s+sessions?|"
    r"how\s+does\s+speaking\s+practice\s+work|"
    r"speaking\s+sessions?"
    r")\b",
    re.IGNORECASE,
)


_PARENT_UPDATES_RE = re.compile(
    r"\b(do\s+parents\s+(?:receive|get)\s+(?:progress\s+)?updates|"
    r"how\s+do\s+parents\s+track\s+progress|"
    r"progress\s+updates\s+for\s+parents|"
    r"parent\s+updates)\b",
    re.IGNORECASE,
)

_MARKS_GUARANTEE_RE = re.compile(
    r"\b(can\s+you\s+guarantee\s+(?:marks|grades|ranks|scores|results)|"
    r"do\s+you\s+guarantee\s+(?:marks|grades|ranks|scores|results)|"
    r"guarantee\s+(?:marks|ranks|results|scores))\b",
    re.IGNORECASE,
)

_CONFIRMATION_RE = re.compile(
    r"^\s*(okay|ok|sure|yes|yeah|yup|got it|understood|alright|all right)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_ENROLLMENT_ACTION_RE = re.compile(
    r"\b("
    r"how\s+(?:can|do)\s+i\s+(?:sign\s+up|enroll|join|register|apply)|"
    r"how\s+to\s+(?:sign\s+up|enroll|join|register|apply)|"
    r"i\s+want\s+to\s+(?:join|enroll|sign\s+up|register|apply)|"
    r"how\s+do\s+i\s+register\s+my\s+child|"
    r"how\s+can\s+my\s+child\s+(?:start|join|enroll)|"
    r"what\s+is\s+the\s+(?:enrollment|admission|registration)\s+process|"
    r"where\s+do\s+i\s+apply|"
    r"can\s+i\s+register(?:\s+for\s+(?:this|the)\s+program)?|"
    r"i\s+want\s+to\s+book\s+a\s+place|"
    r"how\s+do\s+i\s+get\s+started"
    r")\b",
    re.IGNORECASE,
)

_AFFIRMATION_RE = re.compile(
    r"^\s*(yes|yeah|yep|correct|exactly|that\s+one|this\s+one|right|definitely|sure)\s*[!.]*$",
    re.IGNORECASE,
)

_NEGATION_RE = re.compile(
    r"^\s*(no|nope|not\s+that(?:\s+one)?|neither|wrong)\s*[!.]*$",
    re.IGNORECASE,
)

_THE_OTHER_ONE_RE = re.compile(
    r"^\s*(?:what\s+about\s+)?the\s+other(?:\s+one)?\s*[?!.]*$",
    re.IGNORECASE,
)

_ACKNOWLEDGEMENT_RE = re.compile(
    r"^\s*(okay|ok|okay\s+thanks|ok\s+thanks|got\s+it|understood|alright|all\s+right|makes\s+sense|that'?s\s+helpful|helpful|i\s+see)\s*[!.]*$",
    re.IGNORECASE,
)

_MENTORING_APPROACH_RE = re.compile(
    r"\b(tell\s+me\s+about\s+(?:the\s+)?mentoring\s+approach|"
    r"what\s+is\s+(?:the\s+)?mentoring\s+approach|"
    r"how\s+does\s+mentoring\s+work|"
    r"how\s+does\s+personalized\s+mentoring\s+work|"
    r"mentoring\s+approach)\b",
    re.IGNORECASE,
)

_MENTORING_ONE_ON_ONE_RE = re.compile(
    r"\b(is\s+mentoring\s+(?:one[- ]on[- ]one|1[- ]on[- ]1|1:1)|"
    r"is\s+it\s+(?:one[- ]on[- ]one|1[- ]on[- ]1|1:1)|"
    r"do\s+students\s+get\s+individual\s+attention|"
    r"individual\s+attention(?:\s+for\s+students)?|"
    r"is\s+it\s+(?:one[- ]to[- ]one|1[- ]to[- ]1))\b",
    re.IGNORECASE,
)

_FOUNDATION_SUBJECTS_RE = re.compile(
    r"\b(what\s+subjects\s+(?:are\s+included|do\s+you\s+teach|are\s+taught|are\s+there)|"
    r"subjects?\s+(?:included|offered|taught)|"
    r"what\s+are\s+the\s+subjects)\b",
    re.IGNORECASE,
)

_FOUNDATION_APPROACH_RE = re.compile(
    r"\b(how\s+does\s+it\s+work|teaching\s+approach|how\s+is\s+it\s+taught|how\s+do\s+you\s+teach\s+them|learning\s+activities)\b",
    re.IGNORECASE,
)

_FOUNDATION_PROGRESS_RE = re.compile(
    r"\b(progress\s+tracking|weekly\s+progress\s+notes|how\s+do\s+parents\s+track\s+progress|parent\s+notes)\b",
    re.IGNORECASE,
)

ACTION_INTENTS = {
    "enrollment",
    "sign_up",
    "registration",
    "demo_booking",
    "contact",
    "fees",
    "duration",
    "foundation_enrollment",
    "middle_enrollment",
    "senior_enrollment",
    "confident_speaker_enrollment",
}



def normalize_query(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"\bwanna\b", "want to", cleaned)
    cleaned = re.sub(r"\bgonna\b", "going to", cleaned)
    cleaned = re.sub(r"\bu\b", "you", cleaned)
    cleaned = re.sub(r"\bur\b", "your", cleaned)
    cleaned = re.sub(r"\br\b", "are", cleaned)
    cleaned = re.sub(r"\bi'm\b", "i am", cleaned)
    cleaned = re.sub(r"\bim\b", "i am", cleaned)
    cleaned = re.sub(r"\bdon't\b", "do not", cleaned)
    cleaned = re.sub(r"\bdont\b", "do not", cleaned)
    cleaned = re.sub(r"\bcan't\b", "cannot", cleaned)
    cleaned = re.sub(r"\bcant\b", "cannot", cleaned)
    cleaned = re.sub(r"\bwhat's\b", "what is", cleaned)
    cleaned = re.sub(r"\bwhats\b", "what is", cleaned)
    cleaned = re.sub(r"\bprep\b", "prepare", cleaned)
    cleaned = re.sub(r"\bprepping\b", "preparing", cleaned)
    cleaned = re.sub(r"\bboards\b", "board", cleaned)
    cleaned = re.sub(r"\bexams\b", "exam", cleaned)
    cleaned = re.sub(r"\b(foundating|doundation|foudation|foundaton)\b", "foundation", cleaned)
    cleaned = re.sub(r"\b(middel|midle|mddle)\b", "middle", cleaned)
    cleaned = re.sub(r"\bgrades?\s*(\d+)\s*[-–to]\s*(\d+)\b", r"grades \1-\2", cleaned)
    cleaned = re.sub(r"\bclasses?\s*(\d+)\s*[-–to]\s*(\d+)\b", r"grades \1-\2", cleaned)
    cleaned = re.sub(r"\bstandard\s*(\d+)\b", r"grade \1", cleaned)
    cleaned = re.sub(r"\bclass\s*(\d+)\b", r"grade \1", cleaned)
    return cleaned


_CAPABILITY_RE = re.compile(
    r"^\s*(what\s+(?:can\s+)?(?:you|u)\s+help\s+(?:me\s+)?with\??|"
    r"what\s+do\s+(?:you|u)\s+know\??|"
    r"what\s+(?:information|info)\s+do\s+(?:you|u)\s+have\??|"
    r"how\s+can\s+(?:you|u)\s+help\??|"
    r"what\s+can\s+i\s+ask\s+(?:you|u)\??|"
    r"can\s+(?:you|u)\s+help\s+me\??|"
    r"can\s+(?:you|u)\s+help\??|"
    r"i\s+need\s+information\??|"
    r"what\s+are\s+your\s+capabilities\??|"
    r"what\s+can\s+(?:you|u)\s+do\??)\s*$",
    re.IGNORECASE,
)

_GENERAL_INFO_RE = re.compile(
    r"^\s*(what\s+(?:is|are)\s+wementors|"
    r"who\s+(?:is|are)\s+wementors|"
    r"about\s+wementors|"
    r"tell\s+me\s+about\s+wementors|"
    r"tell\s+me\s+about\s+(?:this\s+|your\s+|the\s+)?(?:academy|organization|organisation|institution|company|school)|"
    r"what\s+is\s+(?:this\s+|your\s+|the\s+)?(?:academy|organization|organisation|institution|company|school)\s+about|"
    r"what\s+do\s+you\s+(?:guys\s+)?do|"
    r"what\s+does\s+wementors\s+do|"
    r"tell\s+me\s+what\s+you\s+(?:guys\s+)?do|"
    r"tell\s+me\s+what\s+you\s+do|"
    r"can\s+you\s+explain\s+wementors|"
    r"explain\s+wementors|"
    r"can\s+you\s+explain\s+(?:your\s+)?platform|"
    r"what\s+is\s+this\s+website\s+about|"
    r"give\s+me\s+(?:some\s+)?information\s+about\s+wementors|"
    r"give\s+me\s+(?:some\s+)?information\s+about\s+(?:your\s+)?(?:academy|organization|organisation)|"
    r"i\s+(?:want|wanna)\s+to\s+know\s+more\s+about\s+wementors|"
    r"i\s+(?:want|wanna)\s+to\s+know\s+more\s+about\s+you|"
    r"tell\s+me\s+more\s+about\s+(?:your\s+)?classes|"
    r"how\s+does\s+wementors\s+work|"
    r"how\s+do\s+you\s+(?:guys\s+)?work|"
    r"what\s+does\s+wementors\s+offer|"
    r"can\s+you\s+introduce\s+(?:me\s+to\s+)?wementors|"
    r"introduce\s+wementors|"
    r"can\s+you\s+introduce\s+(?:your\s+)?services|"
    r"introduce\s+(?:your\s+)?services|"
    r"how\s+does\s+(?:your\s+)?academy\s+help(?:\s+students)?|"
    r"how\s+do\s+you\s+help\s+students)\s*[?!.]*$",
    re.IGNORECASE,
)

_BOARD_EXAM_RE = re.compile(
    r"\b(how\s+will\s+you\s+(?:prep|prepare)\s+(?:my\s+)?(?:child|student|son|daughter|kid)\s+for\s+board(?:\s+exam)?s?|"
    r"how\s+do\s+you\s+prepare\s+students\s+for\s+board(?:\s+exam)?s?|"
    r"can\s+you\s+help\s+(?:(?:my\s+)?(?:child|student|son|daughter|kid)|me)?\s*with\s+board(?:\s+exam)?s?|"
    r"what\s+support\s+do\s+you\s+provide\s+for\s+(?:class|grade)\s+(?:9|10|9\s+(?:and|&)\s+10)|"
    r"how\s+will\s+you\s+help\s+(?:my\s+)?(?:child|student|son|daughter|kid)\s+score\s+well\s+in\s+board(?:\s+exam)?s?|"
    r"do\s+you\s+have\s+a\s+board(?:\s+exam)?\s+preparation\s+course|"
    r"how\s+do\s+you\s+prepare\s+(?:class|grade)\s+(?:9|10|9\s+(?:and|&)\s+10)\s+students|"
    r"how\s+will\s+you\s+prep\s+(?:my\s+)?(?:child|student|son|daughter|kid)\s+for\s+board(?:\s+exam)?s?|"
    r"how\s+do\s+you\s+prepare\s+(?:class|grade)\s+10\s+students|"
    r"how\s+do\s+you\s+prepare\s+(?:class|grade)\s+9\s+students|"
    r"can\s+you\s+help\s+(?:my\s+)?(?:child|student|son|daughter|kid)\s+with\s+boards?|"
    r"what\s+support\s+do\s+you\s+provide\s+for\s+(?:grade|class)\s+9|"
    r"what\s+support\s+do\s+you\s+provide\s+for\s+(?:grade|class)\s+10|"
    r"how\s+will\s+the\s+mentor\s+work\s+with\s+my\s+child|"
    r"my\s+(?:son|daughter|child|kid)\s+is\s+in\s+(?:10th|9th)|"
    r"can\s+(?:your\s+)?mentors\s+help\s+with\s+board\s+preparation|"
    r"academic\s+support\s+for\s+.*board\s+exams?|"
    r"board\s+exam\s+prep(?:aration)?|"
    r"preparation\s+for\s+board(?:\s+exam)?s?|"
    r"board\s+preparation\s+course|"
    r"what\s+about\s+board\s+exams?|"
    r"do\s+you\s+offer\s+board\s+exam\s+support|"
    r"do\s+you\s+have\s+board\s+exam\s+classes|"
    r"support\s+for\s+(?:class|grade)\s+(?:9|10))\b",
    re.IGNORECASE,
)

_CONFIDENT_SPEAKER_RE = re.compile(
    r"\b(tell\s+me\s+about\s+(?:the\s+)?confident\s+speaker(?:\s+program)?|"
    r"what\s+is\s+(?:your\s+)?confident\s+speaker(?:\s+(?:course|program))?|"
    r"how\s+does\s+(?:the\s+)?confident\s+speaker(?:\s+program)?\s+work|"
    r"can\s+you\s+help\s+(?:my\s+)?(?:child|student|son|daughter|kid|me)\s+become\s+(?:a\s+)?confident\s+speaker|"
    r"what\s+do\s+students\s+learn\s+in\s+confident\s+speaker|"
    r"is\s+there\s+personalized\s+mentoring\s+for\s+confident\s+speaking|"
    r"how\s+will\s+you\s+work\s+on\s+(?:my\s+)?(?:child'?s\s+|kid'?s\s+)?speaking\s+confidence|"
    r"how\s+can\s+(?:my\s+)?(?:child|student|son|daughter|kid|me)\s+become\s+better\s+at\s+speaking|"
    r"tell\s+me\s+about\s+your\s+speaking\s+confidence\s+program|"
    r"do\s+you\s+have\s+something\s+for\s+public\s+speaking\s+confidence|"
    r"confident\s+speaker|"
    r"speaking\s+confidence|"
    r"public\s+speaking\s+confidence|"
    r"speak\s+confidently|"
    r"learn\s+to\s+speak\s+confidently|"
    r"speak\s+with\s+confidence)\b",
    re.IGNORECASE,
)

_VAGUE_INFO_RE = re.compile(
    r"^\s*(information|info|details)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_HOW_TO_BOOK_RE = re.compile(
    r"^\s*(how\s+(?:can|do)\s+i\s+book(?:\s+(?:a\s+)?(?:free\s+)?demo(?:\s+class)?)?\??|how\s+to\s+book(?:\s+(?:a\s+)?(?:free\s+)?demo(?:\s+class)?)?\??)\s*$",
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
    r"|\b(enquire\s+about\s+joining|interested\s+in\s+joining)\b",
    re.IGNORECASE,
)

_DEMO_TRANSACTION_RE = re.compile(
    r"\b("
    r"book\s+(?:a\s+)?(?:free\s+)?demo\s+for\s+me|"
    r"book\s+me\s+(?:a\s+)?(?:free\s+)?demo|"
    r"book\s+it\s+for\s+me|"
    r"book\s+for\s+me|"
    r"can\s+you\s+book\s+it(?:\s+for\s+me)?|"
    r"can\s+you\s+book\s+(?:a\s+)?(?:free\s+)?demo(?:\s+for\s+me)?|"
    r"can\s+you\s+book\??|"
    r"can\s+you\s+register\s+me(?:\s+for\s+(?:a\s+)?(?:free\s+)?demo)?|"
    r"register\s+me\s+for\s+(?:a\s+)?(?:free\s+)?demo|"
    r"register\s+me\s+for\s+demo|"
    r"register\s+me(?:\s+please)?|"
    r"submit\s+my\s+demo(?:\s+request)?|"
    r"submit\s+(?:a\s+)?demo(?:\s+request)?\s+for\s+me|"
    r"i\s+want\s+you\s+to\s+book\s+it|"
    r"i\s+want\s+you\s+to\s+book\s+(?:a\s+)?(?:free\s+)?demo|"
    r"can\s+you\s+sign\s+me\s+up(?:\s+for\s+(?:a\s+)?(?:free\s+)?demo)?|"
    r"sign\s+me\s+up\s+for\s+(?:a\s+)?(?:free\s+)?demo|"
    r"can\s+you\s+send\s+my\s+details\s+to\s+the\s+team|"
    r"(?:can\s+you\s+)?sign\s+(?:my\s+)?(?:child|kid)\s+up(?:\s+for\s+(?:a\s+)?(?:free\s+)?demo)?|"
    r"can\s+you\s+book\s+something\s+for\s+me"
    r")\b",
    re.IGNORECASE,
)

_DEMO_INFORMATION_RE = re.compile(
    r"\b("
    r"what\s+is\s+(?:the\s+)?(?:free\s+)?demo(?:\s+class)?|"
    r"is\s+(?:the\s+)?(?:free\s+)?demo(?:\s+class)?\s+free|"
    r"tell\s+me\s+about\s+(?:the\s+)?(?:free\s+)?demo(?:\s+class)?|"
    r"what\s+happens\s+(?:in|during)\s+(?:the\s+)?(?:free\s+)?demo(?:\s+class)?|"
    r"how\s+long\s+is\s+(?:the\s+)?(?:free\s+)?demo(?:\s+class)?|"
    r"what\s+does\s+the\s+demo\s+include|"
    r"does\s+the\s+demo\s+cost\s+anything|"
    r"is\s+there\s+any\s+cost\s+for\s+(?:the\s+)?demo"
    r")\b",
    re.IGNORECASE,
)

_DEMO_WHERE_TO_BOOK_RE = re.compile(
    r"\b(where\s+(?:do|can)\s+i\s+book(?:\s+(?:a\s+)?(?:free\s+)?demo)?|"
    r"where\s+to\s+book(?:\s+(?:a\s+)?(?:free\s+)?demo)?|"
    r"where\s+can\s+i\s+sign\s+up(?:\s+for\s+(?:a\s+)?demo)?|"
    r"where\s+is\s+the\s+book\s+free\s+demo\s+button|"
    r"how\s+can\s+i\s+sign\s+up\s+for\s+a\s+demo)\b",
    re.IGNORECASE,
)

_DEMO_FIELD_QUERY_RE = re.compile(
    r"^\s*(name|student\s+name|parent\s+name|student's\s+name|parent's\s+name|"
    r"email|phone|phone\s+number|mobile|contact|grade|subject|time|preferred\s+time)\s*[?!.]*$",
    re.IGNORECASE,
)

_FOUNDATION_GRADES_NARROW_RE = re.compile(
    r"^\s*(?:what\s+grades?\s+(?:is|are)\s+(?:it|foundation(?:\s+years?)?)\s+for\??|"
    r"what\s+grades?\s+does\s+foundation(?:\s+years?)?\s+cover\??|"
    r"what\s+grades?\s+for\s+foundation(?:\s+years?)?\??|"
    r"which\s+grades?\s+(?:is|are)\s+(?:it|foundation(?:\s+years?)?)\s+for\??)\s*$",
    re.IGNORECASE,
)

_MIDDLE_SCHOOL_GRADES_NARROW_RE = re.compile(
    r"^\s*(?:what\s+grades?\s+(?:is|are)\s+(?:it|middle(?:\s+school)?)\s+for\??|"
    r"what\s+grades?\s+does\s+middle(?:\s+school)?\s+cover\??|"
    r"which\s+grades?\s+(?:is|are)\s+(?:it|middle(?:\s+school)?)\s+for\??)\s*$",
    re.IGNORECASE,
)

_IS_ONE_TO_ONE_RE = re.compile(
    r"^\s*(?:is\s+it\s+one[- ]to[- ]one\??|is\s+it\s+1[- ]to[- ]1\??|is\s+it\s+1:1\??|is\s+it\s+one[- ]on[- ]one\??|is\s+it\s+1[- ]on[- ]1\??)\s*$",
    re.IGNORECASE,
)

_WHERE_DETAILS_RE = re.compile(
    r"\b(where\s+(?:do|can)\s+i\s+(?:enter|fill|put|submit|register|type)\s+(?:my\s+)?(?:details|info|information|name|form)|where\s+to\s+(?:enter|fill|put|submit|register)\s+(?:my\s+)?(?:details|info|information)|where\s+can\s+i\s+register(?:\s+for\s+(?:a\s+)?demo)?|where\s+do\s+i\s+register(?:\s+for\s+(?:a\s+)?demo)?|where\s+is\s+the\s+(?:book\s+free\s+demo\s+)?(?:form|button|link|option))\b",
    re.IGNORECASE,
)

# Deep semantic routing: Eligibility, Location, and Delivery intents
_ELIGIBILITY_NON_SCHOOL_RE = re.compile(
    r"\b("
    r"(?:can|could|may|is\s+it\s+possible\s+to)\s+(?:i|someone|anyone|a\s+person|we)\s+(?:still\s+)?join\b.*?\b(?:not\s+in\s+school|without\s+(?:being\s+in\s+)?school|out\s+of\s+school|not\s+a\s+student|not\s+studying)|"
    r"(?:not\s+in\s+school|without\s+(?:being\s+in\s+)?school|out\s+of\s+school|not\s+a\s+student|not\s+studying)\b.*?\b(?:can|could|may)\s+(?:i|someone|anyone|a\s+person|we)\s+(?:still\s+)?join|"
    r"can\s+i\s+join\s+as\s+a\s+person\s+(?:who['’]?s|whose|who\s+is)\s+not\s+(?:in\s+)?(?:school|a\s+student)|"
    r"(?:can|could|may|is\s+it\s+possible\s+to)\s+(?:i|someone|anyone|a\s+person|we)\s+join\s+(?:without|if\s+not|not)\s+(?:being\s+)?(?:in\s+)?(?:school|a\s+student)|"
    r"can\s+(?:someone|anyone)\s+who\s+isn['’]?t\s+(?:currently\s+)?(?:in\s+)?(?:school|a\s+student)\s+join|"
    r"can\s+(?:someone|anyone)\s+who\s+is\s+not\s+(?:currently\s+)?(?:in\s+)?(?:school|a\s+student)\s+join|"
    r"(?:i\s+want\s+to|want\s+to|wanna)\s+join\s+(?:but|though)\s+(?:i['’]?m|i\s+am)\s+not\s+(?:in\s+school|a\s+student)|"
    r"(?:i['’]?m|i\s+am)\s+not\s+(?:in\s+school|a\s+student)(?:[,.]?\s+(?:can|could)\s+i\s+(?:still\s+)?join)?|"
    r"do\s+i\s+(?:have|need)\s+to\s+be\s+in\s+school(?:\s+to\s+join)?|"
    r"do\s+i\s+need\s+to\s+be\s+a\s+student(?:\s+to\s+join)?|"
    r"must\s+i\s+be\s+in\s+school(?:\s+to\s+join)?|"
    r"is\s+(?:this|it|wementors)\s+only\s+for\s+(?:school\s+)?students?|"
    r"is\s+(?:this|it|wementors)\s+only\s+for\s+(?:school\s+)?children|"
    r"are\s+(?:your\s+)?programs\s+only\s+for\s+(?:school\s+)?(?:students?|children|kids)|"
    r"can\s+someone\s+who\s+isn['’]?t\s+a\s+student\s+use\s+this|"
    r"i['’]?m\s+not\s+currently\s+studying(?:\s*,\s*can\s+i\s+join)?|"
    r"(?:not\s+in\s+school|not\s+a\s+student)\s+though|"
    r"(?:i['’]?m|i\s+am)\s+not\s+in\s+school|"
    r"without\s+being\s+in\s+school"
    r")\b",
    re.IGNORECASE,
)

_ELIGIBILITY_ADULT_RE = re.compile(
    r"\b("
    r"(?:can|could|may|is\s+it\s+possible\s+for)\s+(?:adults?|working\s+professionals?|professionals?|homemakers?|housewives|parents?)\s+(?:to\s+)?join|"
    r"(?:can|could|may)\s+(?:i|someone|anyone)\s+(?:still\s+)?join\b.*?\b(?:as|if)\s+(?:an?\s+)?(?:adult|professional|working\s+professional|homemaker|housewife|parent)|"
    r"can\s+(?:an?\s+)?adult\s+take\s+(?:your\s+)?classes|"
    r"can\s+i\s+join\s+if\s+(?:i['’]?m|i\s+am)\s+(?:an?\s+)?(?:adult|working\s+professional|professional|homemaker|housewife|parent)|"
    r"can\s+i\s+join\s+as\s+(?:an?\s+)?(?:adult|working\s+professional|professional|homemaker|housewife|parent)|"
    r"(?:i['’]?m|i\s+am)\s+(?:an?\s+)?(?:adult|working\s+professional|professional|homemaker|housewife)(?:[,.]?\s+(?:can|could)\s+i\s+(?:still\s+)?join)?|"
    r"is\s+there\s+anything\s+for\s+(?:adults|professionals|homemakers)|"
    r"do\s+you\s+have\s+(?:anything|programs?|classes?|courses?)\s+for\s+(?:adults|professionals|homemakers)|"
    r"(?:are\s+there\s+)?classes\s+for\s+(?:adults|professionals|homemakers)|"
    r"can\s+adults\s+learn|"
    r"adult\s+learning|"
    r"is\s+this\s+(?:open\s+to|for)\s+(?:adults|professionals|homemakers)|"
    r"(?:actually\s+)?(?:i['’]?m|i\s+am)\s+an\s+adult|"
    r"(?:actually\s+)?(?:i['’]?m|i\s+am)\s+a\s+(?:working\s+)?professional|"
    r"(?:actually\s+)?(?:i['’]?m|i\s+am)\s+a\s+homemaker"
    r")\b",
    re.IGNORECASE,
)

_ELIGIBILITY_COLLEGE_RE = re.compile(
    r"\b("
    r"can\s+(?:college|university)\s+students?\s+join|"
    r"can\s+i\s+join\s+if\s+(?:i['’]?m|i\s+am)\s+in\s+(?:college|university)|"
    r"(?:i['’]?m|i\s+am)\s+(?:a\s+)?(?:college|university)\s+student(?:[,.]?\s+(?:can|could)\s+i\s+join)?|"
    r"do\s+you\s+teach\s+(?:college|university)\s+students?"
    r")\b",
    re.IGNORECASE,
)

_ELIGIBILITY_GENERAL_RE = re.compile(
    r"^\s*("
    r"who\s+can\s+(?:join|enroll|apply|participate|register)|"
    r"who\s+is\s+(?:eligible\s+to\s+join|eligible)|"
    r"who\s+are\s+(?:your\s+)?programs\s+for|"
    r"who\s+is\s+wementors\s+for|"
    r"who\s+can\s+take\s+(?:your\s+)?classes|"
    r"can\s+anyone\s+join|"
    r"can\s+i\s+join\??|"
    r"am\s+i\s+eligible(?:\s+to\s+join)?\??|"
    r"eligibility\s+criteria|"
    r"what\s+is\s+the\s+eligibility"
    r")\s*[?!.]*$",
    re.IGNORECASE,
)

_ELIGIBILITY_MIXED_ENGLISH_RE = re.compile(
    r"\b("
    r"(?:adult|adults|not\s+in\s+school|without\s+(?:being\s+in\s+)?school|not\s+a\s+student|professional|homemaker).*\b(?:english|speaking|public\s+speaking|spoken\s+english|communication)|"
    r"(?:english|speaking|public\s+speaking|spoken\s+english|communication).*\b(?:adult|adults|not\s+in\s+school|without\s+(?:being\s+in\s+)?school|not\s+a\s+student|professional|homemaker)"
    r")\b",
    re.IGNORECASE,
)

_ELIGIBILITY_MIXED_ONLINE_RE = re.compile(
    r"\b("
    r"(?:adult|adults|not\s+in\s+school|without\s+(?:being\s+in\s+)?school|not\s+a\s+student|professional|homemaker).*\b(?:online|from\s+home|virtual|remote)|"
    r"(?:online|from\s+home|virtual|remote).*\b(?:adult|adults|not\s+in\s+school|without\s+(?:being\s+in\s+)?school|not\s+a\s+student|professional|homemaker)"
    r")\b",
    re.IGNORECASE,
)

_ELIGIBILITY_MIXED_PROGRAMS_RE = re.compile(
    r"\b("
    r"(?:not\s+in\s+school|without\s+(?:being\s+in\s+)?school|not\s+a\s+student|adult|adults).*\b(?:what\s+programs|what\s+can\s+i\s+learn|which\s+programs|what\s+classes|available\s+programs)|"
    r"(?:what\s+programs|what\s+can\s+i\s+learn|which\s+programs|what\s+classes).*\b(?:not\s+in\s+school|without\s+(?:being\s+in\s+)?school|not\s+a\s+student|adult|adults)"
    r")\b",
    re.IGNORECASE,
)

_ELIGIBILITY_MIXED_FEES_RE = re.compile(
    r"\b("
    r"(?:adult|adults|not\s+in\s+school|without\s+(?:being\s+in\s+)?school|not\s+a\s+student|professional).*\b(?:cost|price|fee|fees|pricing|how\s+much)|"
    r"(?:cost|price|fee|fees|pricing|how\s+much).*\b(?:adult|adults|not\s+in\s+school|without\s+(?:being\s+in\s+)?school|not\s+a\s+student|professional)"
    r")\b",
    re.IGNORECASE,
)

_LOCATION_RE = re.compile(
    r"\b("
    r"where\s+(?:are\s+you|is\s+wementors|is\s+the\s+company|is\s+the\s+office|is\s+the\s+center|is\s+the\s+institute)\s+(?:located|based|situated)|"
    r"where\s+(?:are\s+you|are\s+your\s+classes|are\s+the\s+sessions)\s+based|"
    r"where\s+is\s+your\s+office|"
    r"where\s+are\s+you\s+located|"
    r"what\s+is\s+your\s+location|"
    r"what['’]?s\s+your\s+location|"
    r"your\s+location|"
    r"location\s+of\s+wementors|"
    r"physical\s+location|"
    r"where\s+is\s+wementors\s+headquartered|"
    r"are\s+you\s+based\s+in\s+[A-Za-z]+|"
    r"which\s+city\s+are\s+you\s+in|"
    r"what\s+city\s+are\s+you\s+located\s+in|"
    r"do\s+you\s+have\s+a\s+physical\s+(?:office|center|branch|school|location)"
    r")\b",
    re.IGNORECASE,
)

_ONLINE_CLASSES_RE = re.compile(
    r"\b("
    r"are\s+(?:the\s+)?classes\s+online|"
    r"are\s+(?:the\s+)?classes\s+(?:conducted\s+)?online\s+or\s+offline|"
    r"is\s+it\s+online\s+or\s+offline|"
    r"do\s+you\s+teach\s+online|"
    r"are\s+(?:your\s+)?sessions\s+virtual|"
    r"can\s+i\s+attend\s+from\s+home|"
    r"do\s+i\s+have\s+to\s+come\s+somewhere(?:\s+for\s+classes)?|"
    r"do\s+you\s+have\s+offline\s+(?:centers?|classes?)|"
    r"do\s+you\s+offer\s+offline\s+classes|"
    r"are\s+you\s+online\s+only|"
    r"are\s+classes\s+virtual|"
    r"online\s+or\s+offline|"
    r"(?:what\s+about\s+)?online\s+classes\??|"
    r"(?:what\s+about\s+)?virtual\s+classes\??|"
    r"how\s+are\s+classes\s+conducted"
    r")\b",
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

_CANCELLATION_RE = re.compile(
    r"^\s*(?:actually\s+)?(?:never\s*mind|nevermind|nvm|forget\s+(?:it|that)|leave\s+it|"
    r"no\s+worries|that['’]?s\s+okay|it['’]?s\s+fine|don['’]?t\s+worry(?:\s+about\s+it)?|"
    r"i\s+changed\s+my\s+mind|doesn['’]?t\s+matter|ignore\s+that|skip\s+that|let['’]?s\s+forget\s+it|"
    r"cancel|stop|drop\s+it)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_CANCELLATION_PREFIX_RE = re.compile(
    r"^\s*(?:actually\s+)?(?:never\s*mind|nevermind|nvm|forget\s+(?:it|that)|leave\s+it|"
    r"no\s+worries|that['’]?s\s+okay|it['’]?s\s+fine|don['’]?t\s+worry\s+about\s+it|don['’]?t\s+worry|"
    r"doesn['’]?t\s+matter|ignore\s+that|skip\s+that|let['’]?s\s+forget\s+it)\s*[,.:;-]?\s+",
    re.IGNORECASE,
)

_DISCOURSE_PREFIX_RE = re.compile(
    r"^\s*(?:actually|wait|okay|ok|okay\s+so|ok\s+so|so|well|look|listen|oh|ah)\s*[,.:;-]?\s+",
    re.IGNORECASE,
)

_STANDALONE_ACTUALLY_RE = re.compile(
    r"^\s*actually\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_STANDALONE_WAIT_RE = re.compile(
    r"^\s*wait\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_WHY_QUERY_RE = re.compile(
    r"^\s*why(?:\s+so|\s+though|\s+is\s+that)?\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_REALLY_QUERY_RE = re.compile(
    r"^\s*(?:really|is\s+that\s+so|are\s+you\s+sure)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_CONVERSATIONAL_REACTION_RE = re.compile(
    r"^\s*(?:interesting|hmm+|makes\s+sense|that'?s\s+helpful|helpful|i\s+see|fair\s+enough|cool|wow|awesome|nice)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_PURE_ACKNOWLEDGEMENT_RE = re.compile(
    r"^\s*(?:okay|ok|okay\s+thanks|ok\s+thanks|got\s+it|understood|alright|fine)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_STANDALONE_AFFIRMATION_RE = re.compile(
    r"^\s*(?:sure|definitely|absolutely)\s*[!.]*$",
    re.IGNORECASE,
)

_MATHS_STANDALONE_RE = re.compile(
    r"^\s*(?:maths?|mathematics)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_SCIENCE_STANDALONE_RE = re.compile(
    r"^\s*(?:science)\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_EXPLICIT_NAME_RE = re.compile(
    r"^\s*(?:my\s+name\s+is|name\s+is|i\s+am|i'm|call\s+me|you\s+can\s+call\s+me|(?:student|child|son|daughter)(?:'s)?\s+name\s+is)\s+([A-Za-z][A-Za-z\s]{0,30})\s*[.!]*\s*$",
    re.IGNORECASE,
)

_NAME_IS_MY_NAME_RE = re.compile(
    r"^\s*([A-Za-z]{2,20})\s+is\s+my\s+name\s*[.!]*\s*$",
    re.IGNORECASE,
)

_FOUNDATION_TYPO_EXACT_RE = re.compile(
    r"^\s*(foundation|found|foundating|doundation|foudation|foundaton)\s*[?!.]*$",
    re.IGNORECASE,
)
_FOUNDATION_QUERY_RE = re.compile(
    r"\b(tell\s+me\s+about\s+foundation(?:\s+years?)?|"
    r"what\s+is\s+foundation(?:\s+years?)?|"
    r"foundation\s+years?|"
    r"foundation\s+program|"
    r"grades?\s+3\s*[-–to]\s*5|"
    r"class\s+[345]|"
    r"primary\s+school\s+child|"
    r"for\s+my\s+primary\s+school)\b",
    re.IGNORECASE,
)

_MIDDLE_SCHOOL_EXACT_RE = re.compile(
    r"^\s*(middle\s+school|middle|middel|midle)\s*[?!.]*$",
    re.IGNORECASE,
)
_MIDDLE_SCHOOL_OVERVIEW_RE = re.compile(
    r"\b(tell\s+me\s+about\s+middle(?:\s+school)?|"
    r"what\s+is\s+(?:the\s+)?middle\s+school(?:\s+program)?|"
    r"what\s+is\s+middle(?:\s+school)?|"
    r"middle\s+school\s+program|"
    r"explain\s+middle\s+school)\b",
    re.IGNORECASE,
)
_MIDDLE_SCHOOL_GRADES_RE = re.compile(
    r"\b(do\s+you\s+teach\s+grades?\s+6\s*[-–to]\s*8|"
    r"what\s+do\s+you\s+offer\s+for\s+(?:grades?|class)\s+[678]|"
    r"grades?\s+6\s*[-–to]\s*8(?:\s+support|\s+program)?|"
    r"class\s+[678](?:\s+support|\s+program)?|"
    r"what\s+about\s+(?:grades?\s*6\s*[-–to]\s*8|(?:grades?|class)\s*[678])|"
    r"tell\s+me\s+about\s+(?:grades?|class)\s+[678]|"
    r"is\s+middle\s+school\s+for\s+(?:grades?|class)\s+[678]|"
    r"can\s+(?:a\s+)?(?:grade|class)\s+[678]\s+(?:student\s+)?join|"
    r"can\s+students?\s+in\s+(?:grade|class)\s+[678]\s+join|"
    r"do\s+you\s+have\s+classes\s+for\s+(?:grades?|class)\s+[678])\b",
    re.IGNORECASE,
)
_MIDDLE_SCHOOL_SUBJECTS_RE = re.compile(
    r"\b(does\s+middle\s+school\s+include\s+all\s+subjects|"
    r"does\s+middle\s+school\s+include\s+(?:science|maths?|english)|"
    r"what\s+does\s+middle\s+school\s+cover|"
    r"what\s+does\s+(?:grades?|class)\s+[678](?:\s+(?:maths?|science))?\s+cover|"
    r"what\s+subjects\s+are\s+in\s+middle\s+school|"
    r"what\s+subjects\s+are\s+taught\s+in\s+middle\s+school|"
    r"middle\s+school\s+subjects)\b",
    re.IGNORECASE,
)
_DOUBT_CLINICS_RE = re.compile(
    r"\b(what\s+are\s+doubt\s+clinics|"
    r"doubt\s+clinics?|"
    r"are\s+there\s+doubt[- ]solving\s+sessions|"
    r"doubt[- ]solving\s+sessions?)\b",
    re.IGNORECASE,
)
_PRACTICAL_LABS_RE = re.compile(
    r"\b(do\s+students\s+get\s+practical\s+labs|"
    r"practical\s+labs?|"
    r"practical\s+problem\s+sets?)\b",
    re.IGNORECASE,
)
_PROGRESS_DASHBOARD_RE = re.compile(
    r"\b(can\s+parents\s+track\s+progress|"
    r"do\s+parents\s+get\s+progress\s+access|"
    r"progress\s+dashboard(?:\s+access)?|"
    r"how\s+do\s+parents\s+track\s+progress\s+in\s+middle\s+school)\b",
    re.IGNORECASE,
)

_PERSONALIZED_MENTORING_CONCEPT_RE = re.compile(
    r"^\s*(what\s+is\s+personalized\s+mentoring|"
    r"personalized\s+mentoring|"
    r"personal\s+mentoring\s+concept|"
    r"what\s+does\s+personalized\s+mentoring\s+mean|"
    r"explain\s+personalized\s+mentoring)\s*[?!.]*$",
    re.IGNORECASE,
)

_CORRECTION_BEFORE_RE = re.compile(
    r"\b(?:no[,.]?\s*)?(?:the\s+one\s+)?before\s+([a-z0-9\s–-]+)\b",
    re.IGNORECASE,
)
_CORRECTION_AFTER_RE = re.compile(
    r"\b(?:no[,.]?\s*)?(?:the\s+one\s+)?after\s+([a-z0-9\s–-]+)\b",
    re.IGNORECASE,
)
_CORRECTION_NOT_THAT_RE = re.compile(
    r"^\s*(?:no[,.]?\s*)?(?:not\s+that(?:\s+one)?|go\s+back|the\s+previous\s+course|no,?\s*i\s+meant\b)\s*[?!.]*",
    re.IGNORECASE,
)
_WHAT_ABOUT_OTHER_ONE_RE = re.compile(
    r"\b(?:what\s+about\s+)?(?:the\s+other\s+one|what\s+about\s+the\s+other)\b",
    re.IGNORECASE,
)

_PROGRAM_NAME_HINTS = {
    "program-foundation-years": [
        "foundation", "found", "foundating", "doundation",
        "grades 3-5", "grades 3 to 5", "class 3", "class 4", "class 5", "primary school",
    ],
    "program-middle-school": [
        "middle", "middel", "midle",
        "grades 6-8", "grades 6 to 8", "class 6", "class 7", "class 8", "grade 6", "grade 7", "grade 8",
    ],
    "program-senior-school": [
        "senior", "board",
        "grades 9-10", "grades 9 to 10", "class 9", "class 10", "grade 9", "grade 10", "10th", "9th",
    ],
    "program-confident-speaker": ["confident", "speaker", "spoken", "speaking", "interview", "public speaking"],
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
    suggestions: Optional[List[str]] = None


VALID_USER_ROLES = {"user"}
VALID_USER_SOURCES = {
    "text_input",
    "explicit_suggestion_click",
    "form_submission",
}


class ConversationEngine:
    def __init__(self, entries: List[KBEntry]):
        self.entries = entries
        self.entries_by_id = {e.id: e for e in entries}
        self.retriever = Retriever(entries)

    def _get_current_subject(self, session_id: Optional[str]) -> Optional[str]:
        """Infer the subject of the ongoing conversation from database memory and recent turns."""
        if not session_id:
            return None
        try:
            mem = database.get_conversation_memory(session_id)
            if mem.active_program:
                return mem.active_program
            rows = database.get_recent_messages(session_id, limit=8)
            for row in reversed(rows):
                intent = row["intent"] or ""
                content = (row["content"] or "").lower()
                if "foundation" in intent or "foundation" in content:
                    return "foundation"
                if "middle" in intent or "middle" in content:
                    return "middle"
                if "confident_speaker" in intent or "confident speaker" in content or "speaking confidence" in content or "public speaking" in content or "spoken english" in content:
                    return "confident_speaker"
                if "board_exam" in intent or "board" in content or "class 10" in content or "class 9" in content or "grade 10" in content or "grade 9" in content or "senior school" in content:
                    return "board_exam"
                if "academic" in intent or "academic" in content:
                    return "academic"
                if intent in ("general_info", "capability"):
                    return "general"
        except Exception:
            pass
        return None

    def _program_name_to_id(self, name: str) -> Optional[str]:
        n = name.strip().lower()
        if any(w in n for w in ["foundation", "found", "3-5", "grades 3", "primary"]):
            return "program-foundation-years"
        if any(w in n for w in ["middle", "6-8", "grades 6", "class 7", "grade 7"]):
            return "program-middle-school"
        if any(w in n for w in ["senior", "board", "9-10", "grades 9", "class 10", "grade 10", "10th", "9th"]):
            return "program-senior-school"
        if any(w in n for w in ["speaker", "confident", "spoken", "speaking"]):
            return "program-confident-speaker"
        return None

    def _program_id_to_key(self, prog_id: str) -> str:
        if "foundation" in prog_id:
            return "foundation"
        if "middle" in prog_id:
            return "middle"
        if "senior" in prog_id:
            return "senior"
        if "confident" in prog_id:
            return "confident_speaker"
        if "academic" in prog_id:
            return "academic"
        return "general"

    def _extract_program_from_text(self, text: str) -> Optional[str]:
        t = text.lower()
        if re.search(r"\b(foundation(?:\s+years?)?|primary|grades?\s*3\s*[-–to]\s*5|class\s*[345])\b", t):
            return "foundation"
        if re.search(r"\b(middle(?:\s+school)?|grades?\s*6\s*[-–to]\s*8|class\s*[678])\b", t):
            return "middle"
        if re.search(r"\b(senior(?:\s+school)?|board(?:\s+exam)?s?|grades?\s*9\s*[-–to]\s*10|class\s*(?:9|10)|10th|9th)\b", t):
            return "senior"
        if re.search(r"\b(confident\s+speaker|public\s+speaking|spoken\s+english|interview\s+skills?|speaking\s+practice)\b", t):
            return "confident_speaker"
        if re.search(r"\b(academic\s+(?:courses?|classes|sessions?|programs?|subjects?|mentoring|learning)|classes\s+for\s+school\s+students|courses\s+for\s+school\s+students|school\s+students|school\s+courses|academics?)\b", t):
            return "academic"
        return None


    def _reply_for_program_id(self, prog_id: str, session_id: str) -> ReplyResult:
        if prog_id == "program-foundation-years":
            return ReplyResult(
                personality.FOUNDATION_YEARS_DIRECT_RESPONSE,
                "foundation_years_overview",
                ["program-foundation-years"],
                1.0,
            )
        if prog_id == "program-middle-school":
            return ReplyResult(
                personality.MIDDLE_SCHOOL_OVERVIEW_RESPONSE,
                "middle_school_overview",
                ["program-middle-school"],
                1.0,
            )
        if prog_id == "program-senior-school":
            entry = self.entries_by_id.get("program-senior-school")
            scored = [ScoredEntry(entry=entry, score=1.0)] if entry else []
            answer = self._generate_answer("Grades 9-10 program", scored, session_id)
            answer = self._run_response_quality_checks("Grades 9-10 program", answer, "board_exam", [entry] if entry else [])
            return ReplyResult(answer, "board_exam", ["program-senior-school"], 1.0)
        if prog_id == "program-confident-speaker":
            entry = self.entries_by_id.get("program-confident-speaker")
            scored = [ScoredEntry(entry=entry, score=1.0)] if entry else []
            answer = self._generate_answer("Confident Speaker program", scored, session_id)
            answer = self._run_response_quality_checks("Confident Speaker program", answer, "confident_speaker", [entry] if entry else [])
            return ReplyResult(answer, "confident_speaker", ["program-confident-speaker"], 1.0)
        return ReplyResult(personality.pick(personality.GREETINGS), "general", [], None)

    def _resolve_relative_or_correction(
        self, text: str, memory: database.ConversationMemory, session_id: str
    ) -> Optional[ReplyResult]:
        standard_sequence = [
            "program-foundation-years",
            "program-middle-school",
            "program-senior-school",
            "program-confident-speaker",
        ]
        last_list_ids, _ = self._get_last_turn_context(session_id)
        programs_order = memory.ordered_programs or last_list_ids

        m_before = _CORRECTION_BEFORE_RE.search(text)
        if m_before:
            target_name = m_before.group(1).strip().lower()
            target_id = self._program_name_to_id(target_name)
            order_to_use = programs_order if (programs_order and target_id in programs_order) else standard_sequence
            if target_id and target_id in order_to_use:
                idx = order_to_use.index(target_id)
                if idx > 0:
                    prev_id = order_to_use[idx - 1]
                    memory.last_expanded_index = idx - 1
                    memory.active_program = self._program_id_to_key(prev_id)
                    return self._reply_for_program_id(prev_id, session_id)
                else:
                    reply = (
                        "Foundation Years (Grades 3–5) is the earliest academic program we offer at WeMentors, "
                        "followed by Middle School (Grades 6–8) and Senior School (Grades 9–10). "
                        "Did you mean one of those, or our Confident Speaker program?"
                    )
                    return ReplyResult(
                        reply,
                        "correction_clarification",
                        ["program-foundation-years", "programs-overview"],
                        1.0,
                    )

        m_after = _CORRECTION_AFTER_RE.search(text)
        if m_after:
            target_name = m_after.group(1).strip().lower()
            target_id = self._program_name_to_id(target_name)
            order_to_use = programs_order if (programs_order and target_id in programs_order) else standard_sequence
            if target_id and target_id in order_to_use:
                idx = order_to_use.index(target_id)
                if idx < len(order_to_use) - 1:
                    next_id = order_to_use[idx + 1]
                    memory.last_expanded_index = idx + 1
                    memory.active_program = self._program_id_to_key(next_id)
                    return self._reply_for_program_id(next_id, session_id)

        # For bare ordinals and relative references, a list must have been established in this session!
        if not programs_order:
            return None

        if _WHAT_ABOUT_OTHER_ONE_RE.search(text):
            last_idx = memory.last_expanded_index if memory.last_expanded_index is not None else 0
            next_idx = (last_idx + 1) % len(programs_order)
            prog_id = programs_order[next_idx]
            memory.last_expanded_index = next_idx
            memory.active_program = self._program_id_to_key(prog_id)
            return self._reply_for_program_id(prog_id, session_id)

        ordinal_idx = self._extract_ordinal_index(text)
        if ordinal_idx is not None:
            idx = ordinal_idx if ordinal_idx >= 0 else len(programs_order) - 1
            if 0 <= idx < len(programs_order):
                prog_id = programs_order[idx]
                memory.last_expanded_index = idx
                memory.active_program = self._program_id_to_key(prog_id)
                return self._reply_for_program_id(prog_id, session_id)

        if re.search(r"\b(one\s+before\s+that|the\s+previous\s+one|the\s+previous\s+course)\b", text, re.I):
            if memory.last_expanded_index is not None and memory.last_expanded_index > 0:
                idx = memory.last_expanded_index - 1
                prog_id = programs_order[idx]
                memory.last_expanded_index = idx
                memory.active_program = self._program_id_to_key(prog_id)
                return self._reply_for_program_id(prog_id, session_id)

        return None

    def _finalize_result(
        self,
        session_id: Optional[str],
        result: ReplyResult,
        memory: database.ConversationMemory,
        user_message: str = "",
    ) -> ReplyResult:
        prefix = getattr(self, "_pending_intro_prefix", None)
        self._pending_intro_prefix = None
        if prefix and not result.reply.startswith("Nice to meet you"):
            result = ReplyResult(
                prefix + result.reply,
                result.intent,
                result.matched_entry_ids,
                result.confidence,
                result.suggestions,
            )

        prog_id = next((m for m in result.matched_entry_ids if m in _PROGRAM_NAME_HINTS), None)
        if prog_id:
            memory.active_program = self._program_id_to_key(prog_id)
        elif "middle" in result.intent:
            memory.active_program = "middle"
        elif "foundation" in result.intent:
            memory.active_program = "foundation"
        elif "board" in result.intent or "senior" in result.intent:
            memory.active_program = "senior"
        elif "confident_speaker" in result.intent:
            memory.active_program = "confident_speaker"
        elif "academic" in result.intent or result.intent == "grade7_maths_sessions":
            memory.active_program = "academic"
        elif result.intent in ("programs_overview", "comparison") or "programs-overview" in result.matched_entry_ids:
            memory.ordered_programs = [
                "program-foundation-years",
                "program-middle-school",
                "program-senior-school",
                "program-confident-speaker",
            ]
            memory.last_expanded_index = None

        if memory.active_program:
            if memory.active_program not in memory.recent_programs_in_order:
                memory.recent_programs_in_order.append(memory.active_program)
            elif memory.recent_programs_in_order[-1] != memory.active_program:
                memory.recent_programs_in_order.remove(memory.active_program)
                memory.recent_programs_in_order.append(memory.active_program)

        suggestions = personality.get_varied_follow_up_suggestions(
            memory.active_program,
            result.intent,
            memory.previously_shown_suggestions,
            count=2,
        )
        result.suggestions = suggestions
        memory.previously_shown_suggestions.extend(suggestions)
        if len(memory.previously_shown_suggestions) > 12:
            memory.previously_shown_suggestions = memory.previously_shown_suggestions[-12:]

        memory.last_intents.append(result.intent)
        if len(memory.last_intents) > 6:
            memory.last_intents = memory.last_intents[-6:]

        memory.last_user_message = user_message
        memory.last_assistant_message = result.reply
        memory.last_user_intent = result.intent
        memory.last_assistant_intent = result.intent
        memory.turn_count += 1

        if session_id:
            try:
                database.save_conversation_memory(session_id, memory)
            except Exception:
                pass

        return result

    # ---- intent detection -------------------------------------------------
    def detect_intent(self, text: str, session_id: Optional[str] = None) -> str:
        stripped = text.strip()
        normalized = normalize_query(stripped)
        word_count = len(stripped.split())
        memory = database.get_conversation_memory(session_id) if session_id else database.ConversationMemory(session_id="")

        # Pure cancellation signals (checked before prefix stripping so pure cancellation phrases are never split)
        if _CANCELLATION_RE.match(stripped):
            return "cancellation"

        # Discourse / cancellation prefix followed by substantive question
        # e.g. "nevermind, tell me about Middle School", "forget it, what subjects are offered?"
        m_cancel_pref = _CANCELLATION_PREFIX_RE.match(stripped)
        if m_cancel_pref:
            remainder = stripped[m_cancel_pref.end():].strip()
            if len(remainder) >= 3 and re.search(r"[A-Za-z]", remainder):
                stripped = remainder
                normalized = normalize_query(stripped)
                word_count = len(stripped.split())

        m_disc_pref = _DISCOURSE_PREFIX_RE.match(stripped)
        if m_disc_pref:
            remainder = stripped[m_disc_pref.end():].strip()
            if len(remainder) >= 3 and re.search(r"[A-Za-z]", remainder):
                stripped = remainder
                normalized = normalize_query(stripped)
                word_count = len(stripped.split())

        # Vocative prefix, e.g. "Raaid, what programs do you offer?"
        m_vocative = re.match(r"^\s*([A-Za-z]{2,20})\s*[,.:;-]\s*(.+)$", stripped)
        if m_vocative:
            cand_token = m_vocative.group(1).strip().lower()
            rem = m_vocative.group(2).strip()
            if (
                cand_token not in ("how", "what", "why", "when", "where", "who", "which", "can", "do", "does", "is", "are")
                and re.search(r"^(what|how|why|when|where|tell me|explain|can you|do you|which|i want|book)\b", rem, re.I)
            ):
                stripped = rem
                normalized = normalize_query(stripped)
                word_count = len(stripped.split())

        # Explicit intro + question: "My name is Raaid, tell me about Grade 7"
        m_intro_q = re.match(r"^\s*(?:my\s+name\s+is|name\s+is|i\s+am|i'm|call\s+me)\s+([A-Za-z]{2,20})\s*[,.:;-]\s*(.+)$", stripped, re.I)
        if m_intro_q:
            cand_name = m_intro_q.group(1).strip().capitalize()
            rem_q = m_intro_q.group(2).strip()
            if cand_name.lower() not in leads._NAME_BLACKLIST and len(rem_q) >= 3:
                stripped = rem_q
                normalized = normalize_query(stripped)
                word_count = len(stripped.split())

        # Deep semantic routing: Compound / Mixed intents checked FIRST
        # to ensure compound whole-utterance meaning takes priority.
        if _ELIGIBILITY_MIXED_ENGLISH_RE.search(stripped) or _ELIGIBILITY_MIXED_ENGLISH_RE.search(normalized):
            return "eligibility_mixed_english"

        if _ELIGIBILITY_MIXED_ONLINE_RE.search(stripped) or _ELIGIBILITY_MIXED_ONLINE_RE.search(normalized):
            return "eligibility_mixed_online"

        if _ELIGIBILITY_MIXED_PROGRAMS_RE.search(stripped) or _ELIGIBILITY_MIXED_PROGRAMS_RE.search(normalized):
            return "eligibility_mixed_programs"

        if _ELIGIBILITY_MIXED_FEES_RE.search(stripped) or _ELIGIBILITY_MIXED_FEES_RE.search(normalized):
            return "eligibility_mixed_fees"

        # Single-topic Location, Online delivery, and Eligibility checked next
        if _LOCATION_RE.search(stripped) or _LOCATION_RE.search(normalized):
            return "location"

        if _ONLINE_CLASSES_RE.search(stripped) or _ONLINE_CLASSES_RE.search(normalized):
            return "online_classes"

        if _ELIGIBILITY_NON_SCHOOL_RE.search(stripped) or _ELIGIBILITY_NON_SCHOOL_RE.search(normalized):
            return "eligibility_non_school"

        if _ELIGIBILITY_ADULT_RE.search(stripped) or _ELIGIBILITY_ADULT_RE.search(normalized):
            return "eligibility_adult"

        if _ELIGIBILITY_COLLEGE_RE.search(stripped) or _ELIGIBILITY_COLLEGE_RE.search(normalized):
            return "eligibility_college"

        if _ELIGIBILITY_GENERAL_RE.match(stripped) or _ELIGIBILITY_GENERAL_RE.match(normalized):
            return "eligibility_general"

        # Explicit name intro (guarded against articles and roles)
        if _EXPLICIT_NAME_RE.match(stripped) or _NAME_IS_MY_NAME_RE.match(stripped):
            m_cand = _EXPLICIT_NAME_RE.match(stripped) or _NAME_IS_MY_NAME_RE.match(stripped)
            cand_str = m_cand.group(1).strip().lower() if m_cand else ""
            cand_tokens = cand_str.split()
            if (
                cand_tokens
                and cand_tokens[0] not in ("a", "an", "the")
                and not any(t in leads._NAME_BLACKLIST for t in cand_tokens)
                and not _ELIGIBILITY_ADULT_RE.search(stripped)
                and not _ELIGIBILITY_NON_SCHOOL_RE.search(stripped)
            ):
                return "explicit_name"

        # Standalone discourse markers
        if _STANDALONE_ACTUALLY_RE.match(stripped):
            return "discourse_marker"
        if _STANDALONE_WAIT_RE.match(stripped):
            return "conversational_continuation"

        # Conversational questions
        if _WHY_QUERY_RE.match(stripped):
            return "question"
        if _REALLY_QUERY_RE.match(stripped):
            return "conversational_question"

        # Reactions & acknowledgements
        if _CONVERSATIONAL_REACTION_RE.match(stripped):
            if re.search(r"\b(makes\s+sense|that'?s\s+helpful|helpful|i\s+see|fair\s+enough)\b", stripped, re.I):
                return "acknowledgement"
            return "conversational_reaction"
        if _PURE_ACKNOWLEDGEMENT_RE.match(stripped):
            return "acknowledgement"
        if _STANDALONE_AFFIRMATION_RE.match(stripped):
            if memory.pending_clarification:
                return "clarification_affirmation"
            return "affirmation"

        # Standalone subjects / topics
        if _MATHS_STANDALONE_RE.match(stripped) or _SCIENCE_STANDALONE_RE.match(stripped):
            return "subject/topic"

        # Standalone subjects question
        if re.search(r"^\s*what\s+subjects(?:\s+are)?\s+offered\??\s*$", stripped, re.I):
            return "information_request"

        if _SHORT_CONFUSION_RE.match(stripped):
            return "confused"
        if _BOOK_ENROLL_RE.match(stripped):
            return "book_enroll"
        if _CAPABILITY_RE.match(stripped) or _CAPABILITY_RE.match(normalized):
            return "capability"
        if _GENERAL_INFO_RE.match(stripped) or _GENERAL_INFO_RE.match(normalized):
            return "general_info"

        # Multi-intent check: if user asks a compound question with multiple topics (e.g. subjects + joining)
        has_multi_intent = bool(
            re.search(r"\b(and|also)\b", stripped, re.I)
            and re.search(r"\b(subject|subjects|teach|course|courses|program|programs|curriculum)\b", stripped, re.I)
            and re.search(r"\b(join|apply|enroll|sign\s+up|register)\b", stripped, re.I)
        )

        # Standalone generic apply check (How do I apply?) when no program is active
        is_generic_apply = bool(
            re.search(r"^\s*how\s+(?:can|do)\s+i\s+apply\??\s*$", stripped, re.I)
            and not (self._extract_program_from_text(stripped) or memory.active_program)
        )

        # PRIORITY 1: Explicit Action in current user message
        if (
            not has_multi_intent
            and not is_generic_apply
            and (_ENROLLMENT_ACTION_RE.search(stripped) or _ENROLLMENT_ACTION_RE.search(normalized))
            and not re.search(r"\b(demo|trial)\b", stripped, re.I)
            and not _ELIGIBILITY_NON_SCHOOL_RE.search(stripped)
            and not _ELIGIBILITY_ADULT_RE.search(stripped)
            and not _ELIGIBILITY_COLLEGE_RE.search(stripped)
            and not _ELIGIBILITY_GENERAL_RE.match(stripped)
            and not _LOCATION_RE.search(stripped)
            and not _ONLINE_CLASSES_RE.search(stripped)
        ):
            explicit_prog = self._extract_program_from_text(stripped) or memory.active_program
            if explicit_prog == "foundation":
                return "foundation_enrollment"
            if explicit_prog == "middle":
                return "middle_enrollment"
            if explicit_prog == "senior":
                return "senior_enrollment"
            if explicit_prog == "confident_speaker":
                return "confident_speaker_enrollment"
            return "general_enrollment_clarification"

        if _DEMO_TRANSACTION_RE.search(stripped) or _DEMO_TRANSACTION_RE.search(normalized):
            return "demo_transaction_request"
        if _DEMO_INFORMATION_RE.search(stripped) or _DEMO_INFORMATION_RE.search(normalized):
            return "demo_information"
        if _DEMO_WHERE_TO_BOOK_RE.search(stripped) or _DEMO_WHERE_TO_BOOK_RE.search(normalized):
            return "demo_booking"
        if _HOW_TO_BOOK_RE.match(stripped):
            return "demo_booking"
        if _DEMO_CLASS_EXACT_RE.match(stripped):
            return "demo_inquiry"
        if _FOUNDATION_GRADES_NARROW_RE.match(stripped) or _FOUNDATION_GRADES_NARROW_RE.match(normalized):
            return "foundation_grades"
        if _MIDDLE_SCHOOL_GRADES_NARROW_RE.match(stripped) or _MIDDLE_SCHOOL_GRADES_NARROW_RE.match(normalized):
            return "middle_school_grades"
        if _IS_ONE_TO_ONE_RE.match(stripped) or _IS_ONE_TO_ONE_RE.match(normalized):
            return "one_on_one_general"

        # Mentoring questions (unless a specific program is explicitly asked, e.g. "in confident speaker")
        if not re.search(r"\b(confident|speaker|spoken)\b", stripped, re.I):
            if _MENTORING_APPROACH_RE.search(stripped) or _MENTORING_APPROACH_RE.search(normalized):
                return "mentoring_approach"
            if _MENTORING_ONE_ON_ONE_RE.search(stripped) or _MENTORING_ONE_ON_ONE_RE.search(normalized):
                return "one_on_one_general"
            if _PERSONALIZED_MENTORING_CONCEPT_RE.search(stripped) or _PERSONALIZED_MENTORING_CONCEPT_RE.search(normalized):
                return "personalized_mentoring_general"

        # PRIORITY 2: Relative references and corrections
        if (
            _CORRECTION_BEFORE_RE.search(stripped)
            or _CORRECTION_AFTER_RE.search(stripped)
            or _CORRECTION_NOT_THAT_RE.match(stripped)
            or _WHAT_ABOUT_OTHER_ONE_RE.search(stripped)
        ):
            return "reference_resolution"

        if _COMPARISON_RE.search(stripped) or _COMPARISON_RE.search(normalized):
            return "comparison"

        # PRIORITY 3: Explicit Program Mentions (Current message)
        if _GRADE7_MATHS_SESSIONS_RE.search(stripped) or _GRADE7_MATHS_SESSIONS_RE.search(normalized):
            return "grade7_maths_sessions"

        # Foundation Years explicit mentions
        if _FOUNDATION_TYPO_EXACT_RE.match(stripped):
            return "foundation_years_clarification"
        if _FOUNDATION_QUERY_RE.search(stripped) or _FOUNDATION_QUERY_RE.search(normalized):
            if _FOUNDATION_SUBJECTS_RE.search(stripped) or _FOUNDATION_SUBJECTS_RE.search(normalized):
                return "foundation_subjects"
            if _FOUNDATION_APPROACH_RE.search(stripped) or _FOUNDATION_APPROACH_RE.search(normalized):
                return "foundation_approach"
            if _FOUNDATION_PROGRESS_RE.search(stripped) or _FOUNDATION_PROGRESS_RE.search(normalized):
                return "foundation_progress"
            return "foundation_years_overview"

        # Middle School explicit mentions
        if _MIDDLE_SCHOOL_EXACT_RE.match(stripped):
            return "middle_school_overview"
        if _MIDDLE_SCHOOL_OVERVIEW_RE.search(stripped) or _MIDDLE_SCHOOL_OVERVIEW_RE.search(normalized):
            return "middle_school_overview"
        if _MIDDLE_SCHOOL_GRADES_RE.search(stripped) or _MIDDLE_SCHOOL_GRADES_RE.search(normalized):
            return "middle_school_grades"
        if _MIDDLE_SCHOOL_SUBJECTS_RE.search(stripped) or _MIDDLE_SCHOOL_SUBJECTS_RE.search(normalized):
            return "middle_school_subjects"
        if _DOUBT_CLINICS_RE.search(stripped) or _DOUBT_CLINICS_RE.search(normalized):
            return "middle_school_doubt_clinics"
        if _PRACTICAL_LABS_RE.search(stripped) or _PRACTICAL_LABS_RE.search(normalized):
            return "middle_school_practical_labs"
        if _PROGRESS_DASHBOARD_RE.search(stripped) or _PROGRESS_DASHBOARD_RE.search(normalized):
            return "middle_school_progress_dashboard"

        # PRIORITY 4: Genuine Confirmations / Clarifications
        if _AFFIRMATION_RE.match(stripped):
            if memory.pending_clarification:
                return "clarification_affirmation"
            return "confirmation_without_clarification"

        if _NEGATION_RE.match(stripped):
            if memory.pending_clarification:
                return "clarification_negation"
            return "negation_general"

        if _THE_OTHER_ONE_RE.match(stripped) and memory.pending_clarification:
            return "clarification_other_option"

        if _ACKNOWLEDGEMENT_RE.match(stripped):
            return "acknowledgement"

        # PRIORITY 5 & 6: Explicit Entity Resolution & Active Program Context
        # Rule: CURRENT EXPLICIT REFERENT > PREVIOUS CONTEXT
        explicit_prog = self._extract_program_from_text(stripped) or self._extract_program_from_text(normalized)
        has_cs_explicit = bool(
            _CONFIDENT_SPEAKER_RE.search(stripped)
            or _CONFIDENT_SPEAKER_RE.search(normalized)
            or _SPEAKING_PRACTICE_RE.search(stripped)
        )
        has_academic_explicit = bool(
            explicit_prog == "academic"
            or _ACADEMIC_COURSES_OVERVIEW_RE.search(stripped)
            or _ACADEMIC_COURSES_OVERVIEW_RE.search(normalized)
        )
        has_grade7_maths = bool(
            _GRADE7_MATHS_SESSIONS_RE.search(stripped)
            or _GRADE7_MATHS_SESSIONS_RE.search(normalized)
        )

        if has_grade7_maths:
            return "grade7_maths_sessions"

        is_session_activities = bool(
            _SESSION_ACTIVITIES_RE.search(stripped)
            or _SESSION_ACTIVITIES_RE.search(normalized)
        )

        # Resolve effective program: explicit referent in current turn overrides previous context!
        if has_cs_explicit:
            effective_prog = "confident_speaker"
        elif has_academic_explicit:
            effective_prog = "academic"
        elif explicit_prog:
            effective_prog = explicit_prog
        else:
            effective_prog = memory.active_program or self._get_current_subject(session_id)

        # Practice / Session activities routing based on effective program
        if is_session_activities:
            if effective_prog == "confident_speaker":
                return "confident_speaker_activities"
            else:
                return "academic_sessions"

        # Explicit academic courses overview / inquiries
        if has_academic_explicit:
            return "academic_courses_overview"

        # Explicit Confident Speaker inquiries
        if has_cs_explicit:
            if re.search(r"^\s*(how\s+does\s+(?:it|the\s+program)\s+work\??|how\s+does\s+it\s+work\??)\s*$", stripped, re.I):
                return "confident_speaker_format"
            if _FORMAT_RE.search(stripped) or _FORMAT_RE.search(normalized):
                return "confident_speaker_format"
            if _AUDIENCE_RE.search(stripped) or _AUDIENCE_RE.search(normalized):
                return "confident_speaker_audience"
            if _SCOPE_RE.search(stripped) or _SCOPE_RE.search(normalized):
                return "confident_speaker_scope"
            if _MENTORING_RE.search(stripped) or _MENTORING_RE.search(normalized):
                return "confident_speaker_mentoring"
            if _DEMO_BOOKING_RE.search(stripped):
                return "demo_booking"
            return "confident_speaker"

        active_prog = effective_prog

        if active_prog == "foundation":
            if _FOUNDATION_GRADES_NARROW_RE.match(stripped) or re.search(r"^\s*(what\s+grades?\??|grades?\??)\s*$", stripped, re.I):
                return "foundation_grades"
            if _FOUNDATION_SUBJECTS_RE.search(stripped) or _FOUNDATION_SUBJECTS_RE.search(normalized):
                return "foundation_subjects"
            if _FOUNDATION_APPROACH_RE.search(stripped) or _FOUNDATION_APPROACH_RE.search(normalized):
                return "foundation_approach"
            if _FOUNDATION_PROGRESS_RE.search(stripped) or _FOUNDATION_PROGRESS_RE.search(normalized):
                return "foundation_progress"

        if active_prog == "middle":
            if _MIDDLE_SCHOOL_GRADES_RE.search(stripped) or _MIDDLE_SCHOOL_GRADES_NARROW_RE.match(stripped) or re.search(r"^\s*(what\s+grades?\??|grades?\??)\s*$", stripped, re.I):
                return "middle_school_grades"
            if _MIDDLE_SCHOOL_SUBJECTS_RE.search(stripped) or _MIDDLE_SCHOOL_SUBJECTS_RE.search(normalized) or re.search(r"^\s*(what\s+subjects?\??|subjects?\??)\s*$", stripped, re.I):
                return "middle_school_subjects"
            if _DOUBT_CLINICS_RE.search(stripped) or _DOUBT_CLINICS_RE.search(normalized) or re.search(r"\b(doubt|doubts)\b", stripped, re.I):
                return "middle_school_doubt_clinics"
            if _PRACTICAL_LABS_RE.search(stripped) or _PRACTICAL_LABS_RE.search(normalized) or re.search(r"\b(labs?|practical)\b", stripped, re.I):
                return "middle_school_practical_labs"
            if _PROGRESS_DASHBOARD_RE.search(stripped) or _PROGRESS_DASHBOARD_RE.search(normalized):
                return "middle_school_progress_dashboard"
            if re.search(r"^\s*(how\s+does\s+(?:it|the\s+program)\s+work\??|how\s+does\s+it\s+work\??)\s*$", stripped, re.I):
                return "middle_school_overview"

        if active_prog == "confident_speaker":
            if re.search(r"^\s*(how\s+does\s+(?:it|the\s+program)\s+work\??|how\s+does\s+it\s+work\??)\s*$", stripped, re.I):
                return "confident_speaker_format"
            if _FORMAT_RE.search(stripped) or _FORMAT_RE.search(normalized):
                return "confident_speaker_format"
            if _AUDIENCE_RE.search(stripped) or _AUDIENCE_RE.search(normalized):
                return "confident_speaker_audience"
            if _SCOPE_RE.search(stripped) or _SCOPE_RE.search(normalized):
                return "confident_speaker_scope"
            if _MENTORING_RE.search(stripped) or _MENTORING_RE.search(normalized):
                return "confident_speaker_mentoring"
            if _ACTIVITIES_RE.search(stripped) or _ACTIVITIES_RE.search(normalized):
                return "confident_speaker_activities"
            if _DEMO_BOOKING_RE.search(stripped):
                return "demo_booking"

        has_board_mention = bool(_BOARD_EXAM_RE.search(stripped) or _BOARD_EXAM_RE.search(normalized))
        is_board_context = has_board_mention or (active_prog in ("senior", "board_exam"))
        if is_board_context:
            if _MARKS_GUARANTEE_RE.search(stripped) or _MARKS_GUARANTEE_RE.search(normalized):
                return "marks_guarantee"
            if _PARENT_UPDATES_RE.search(stripped) or _PARENT_UPDATES_RE.search(normalized):
                return "parent_updates"
            if _MENTORING_RE.search(stripped) or _MENTORING_RE.search(normalized):
                return "board_mentoring"
            if has_board_mention:
                return "board_exam"

        # Independent checks for specific fields without explicit program mention
        if _MARKS_GUARANTEE_RE.search(stripped) or _MARKS_GUARANTEE_RE.search(normalized):
            return "marks_guarantee"
        if _PARENT_UPDATES_RE.search(stripped) or _PARENT_UPDATES_RE.search(normalized):
            return "parent_updates"
        if _MENTORING_RE.search(stripped) or _MENTORING_RE.search(normalized):
            return "one_on_one_general"
        if _FORMAT_RE.search(stripped) or _FORMAT_RE.search(normalized):
            return "unclear_format"

        if _VAGUE_INFO_RE.match(stripped):
            return "vague_info"
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
        if _CONFIRMATION_RE.match(stripped):
            return "confirmation"
        if _THANKS_RE.search(stripped) and len(stripped.split()) <= 6:
            return "thanks"
        if _COMPARISON_RE.search(stripped):
            return "comparison"
        if _CONFUSED_RE.search(stripped):
            return "confused"

        # Conservative standalone name candidate check
        if (
            word_count == 1
            and re.match(r"^[A-Za-z]+$", stripped)
            and stripped.lower() not in leads._NAME_BLACKLIST
            and not any(p.search(stripped) for p, _ in leads._SUBJECT_PATTERNS)
            and not any(p.search(stripped) for p in leads._GRADE_PATTERNS)
            and stripped.lower() not in ("what", "how", "why", "where", "when", "who", "which", "yes", "no", "sure", "help", "demo", "info", "general")
        ):
            last_msg = memory.last_assistant_message or ""
            asked_name = bool(re.search(r"\b(what('s| is) your name|may i (know|have) your name|can i have your name|your name\??)\b", last_msg, re.I))
            if asked_name:
                return "explicit_name"
            return "context-dependent"

        return "faq"

    def classify_conversation_intent(self, text: str, session_id: Optional[str] = None) -> str:
        """Classify user intent into high-level conversational categories for regression and routing matrix."""
        detected = self.detect_intent(text, session_id)
        if detected in (
            "foundation_years_overview", "middle_school_overview", "senior_school",
            "confident_speaker", "academic_courses_overview", "academic_sessions",
            "grade7_maths_sessions", "confident_speaker_activities",
        ):
            return "program/topic"
        if detected.startswith("eligibility"):
            return "eligibility"
        if detected == "location":
            return "location"
        if detected == "online_classes":
            return "online_classes"
        if detected == "faq" and re.search(r"\b(subject|subjects|teach|curriculum)\b", text, re.I):
            return "information_request"
        if detected == "thanks" and re.search(r"\b(okay|ok)\b", text, re.I):
            return "acknowledgement"
        return detected


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
            is_course_query = any(item.entry.id.startswith(("program-", "curriculum-", "programs-overview", "what-is-")) for item in scored)
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

    def _get_fallback_reply(self, message: str) -> str:
        msg_norm = normalize_query(message)
        if re.search(r"\b(board|boards|exam|exams|grade\s*9|grade\s*10|class\s*9|class\s*10|10th|9th)\b", msg_norm):
            return personality.UNCLEAR_BOARD_FALLBACK
        if re.search(r"\b(duration|schedule|hours|timing|timings|frequency|material|materials|books|notes|syllabus|guarantee|rank|marks|score|teacher|teachers|faculty)\b", msg_norm):
            return personality.UNCONFIRMED_DETAILS_FALLBACK
        if re.search(r"\b(course|courses|program|programs|curriculum|class|classes)\b", msg_norm):
            return personality.VAGUE_MENU_RESPONSE
        return personality.VAGUE_MENU_RESPONSE

    def _run_response_quality_checks(
        self, query: str, reply: str, intent: str, matched_entries: List[KBEntry], last_assistant_reply: Optional[str] = None
    ) -> str:
        """Verify and sanitize response quality before returning it to the user.
        Ensures fee isolation, strips unsupported claims/hallucinations, and enforces
        clear, verified facts and next steps.
        """
        if not reply:
            return personality.AMBIGUOUS_GENERAL_FALLBACK

        query_norm = normalize_query(query)
        has_fee_intent = bool(
            set(tokenize(query_norm)) & _FEE_TRIGGER_WORDS
            or re.search(r"\b(fee|fees|cost|costs|price|prices|pricing|charge|rate|how much)\b", query_norm)
        )

        # Avoid unrelated fee information if user did not ask about fees
        if not has_fee_intent and intent not in ("fees", "unverified_course_fees"):
            if re.search(r"\b(fee details are shared individually|our fees are|fee structure|tuition fees?)\b", reply, re.IGNORECASE):
                reply = re.sub(r"\b(Fee details are shared individually[^.\n]*[.\n]?|Our fees are[^.\n]*[.\n]?)", "", reply, flags=re.IGNORECASE).strip()

        # Avoid unsupported claims or guarantees (marks, ranks, 100%, guaranteed results, fluency)
        unsupported_phrases = [
            r"\bguaranteed\s+(?:marks|results?|ranks?|success|grades?|score)\b",
            r"\b100%\s+(?:guaranteed|success|pass)\b",
            r"\bguarantee\s+(?:that|your|a|to)\b",
            r"\beliminate\s+all\s+(?:anxiety|fear)\b",
            r"\bbecome\s+(?:completely\s+)?fluent\s+overnight\b",
        ]
        for pat in unsupported_phrases:
            reply = re.sub(pat, "individual progress and guided development", reply, flags=re.IGNORECASE)

        # Action claim validation: prevent claiming persistent external actions unless actually in confirmed lead flow
        if intent not in ("demo_submitted", "demo_confirming"):
            unsupported_action_claims = [
                (r"\b(?:i've\s+noted|i\s+noted|i\s+have\s+noted)\s+(?:your\s+)?([A-Za-z0-9\s:–-]+)\b", r"You can provide your \1 in the Book Free Demo form on the website"),
                (r"\b(?:i've\s+saved|i\s+saved|i\s+have\s+saved)\s+(?:your\s+)?([A-Za-z0-9\s:–-]+)\b", r"You can enter your \1 in the Book Free Demo form on the website"),
                (r"\b(?:your\s+demo(?:\s+request)?\s+has\s+been\s+(?:submitted|booked|registered|confirmed))\b", r"To submit a demo request, use the Book Free Demo form at the top-right of the website"),
            ]
            for pat, repl in unsupported_action_claims:
                reply = re.sub(pat, repl, reply, flags=re.IGNORECASE)

        # Remove accidental leading "Yes." if it wasn't an affirmative confirmation or direct positive answer
        allowed_yes_intents = (
            "confirmation",
            "foundation_confirmation",
            "middle_school_confirmation",
            "senior_school_confirmation",
            "confident_speaker_confirmation",
            "clarification_affirmation",
            "demo_offer_affirm",
            "eligibility_non_school",
            "eligibility_adult",
            "eligibility_college",
            "eligibility_general",
            "eligibility_mixed_english",
            "eligibility_mixed_online",
            "eligibility_mixed_programs",
            "eligibility_mixed_fees",
            "online_classes",
        )
        if intent not in allowed_yes_intents:
            reply = re.sub(r"^\s*Yes\.\s*", "", reply)

        # Intent mismatch check: prevent semantic drift / wrong program dumps
        if intent.startswith("eligibility") and re.search(r"\b(Middle School covers Grades 6–8|Foundation Years covers Grades 3–5)\b", reply, re.IGNORECASE):
            reply = personality.ELIGIBILITY_NON_SCHOOL_RESPONSE if "non_school" in intent else personality.ELIGIBILITY_ADULT_RESPONSE
        elif intent == "location" and re.search(r"\b(Click Book Free Demo|Book Free Demo form at the top-right)\b", reply, re.IGNORECASE):
            reply = personality.LOCATION_RESPONSE
        elif intent == "online_classes" and re.search(r"\b(Middle School covers Grades 6–8|Which program would you like to enroll in)\b", reply, re.IGNORECASE):
            reply = personality.ONLINE_CLASSES_RESPONSE
        elif intent in ("academic_sessions", "academic_courses_overview", "grade7_maths_sessions") and re.search(r"\b(guided speaking practice|practical conversation|public speaking|confident speaker)\b", reply, re.IGNORECASE):
            if intent == "academic_courses_overview":
                reply = personality.ACADEMIC_COURSES_OVERVIEW_RESPONSE
            elif intent == "grade7_maths_sessions":
                reply = personality.GRADE7_MATHS_SESSIONS_RESPONSE
            else:
                reply = personality.ACADEMIC_SESSIONS_RESPONSE

        # Action intent check: If intent is an action intent (enrollment, booking, etc.), ensure action guidance exists
        if intent in ACTION_INTENTS:
            if not re.search(r"\b(book free demo|free demo|website|register|apply)\b", reply, re.IGNORECASE):
                reply += "\n\nYou can click **Book Free Demo** at the top-right of the website to get started."

        # For program/course/general questions, ensure a practical next step exists
        if intent in ("general_info", "board_exam", "confident_speaker") and not re.search(r"\b(book free demo|free demo)\b", reply, re.IGNORECASE):
            reply += "\n\nYou can click **Book Free Demo** at the top-right of the website to explore our mentoring."

        # Detect near-duplicate responses
        if last_assistant_reply and reply.strip().lower() == last_assistant_reply.strip().lower():
            logger.info("Near-duplicate reply detected. Validating context alignment.")

        return reply

    # ---- main entry point ---------------------------------------------------
    def handle_message(
        self,
        session_id: str,
        message: str,
        source: str = "text_input",
        role: str = "user",
    ) -> ReplyResult:
        VALID_USER_SOURCES = {
            "text_input",
            "explicit_suggestion_click",
            "form_submission",
        }
        if role != "user":
            raise ValueError(f"Invalid message role '{role}'. Only 'user' messages can be submitted.")
        if source not in VALID_USER_SOURCES:
            raise ValueError(
                f"Invalid message source '{source}'. Must be one of {sorted(list(VALID_USER_SOURCES))}."
            )

        message = message.strip()
        memory = database.get_conversation_memory(session_id) if session_id else database.ConversationMemory(session_id="")

        # Pure cancellation signals (checked before prefix stripping so pure cancellation phrases are never split)
        if _CANCELLATION_RE.match(message):
            memory.pending_clarification = None
            memory.last_requested_action = None
            database.save_conversation_memory(session_id, memory)
            database.clear_demo_lead(session_id)
            reply = personality.pick(personality.CANCELLATION_RESPONSES)
            res = ReplyResult(reply, "cancellation", ["programs-overview"], 1.0)
            return self._finalize_result(session_id, res, memory, message)

        # 1. Discourse / cancellation prefix followed by substantive question
        # e.g. "nevermind, tell me about Middle School", "forget it, what subjects are offered?"
        m_cancel_pref = _CANCELLATION_PREFIX_RE.match(message)
        if m_cancel_pref:
            remainder = message[m_cancel_pref.end():].strip()
            if len(remainder) >= 3 and re.search(r"[A-Za-z]", remainder):
                message = remainder

        m_disc_pref = _DISCOURSE_PREFIX_RE.match(message)
        if m_disc_pref:
            remainder = message[m_disc_pref.end():].strip()
            if len(remainder) >= 3 and re.search(r"[A-Za-z]", remainder):
                message = remainder

        # Vocative prefix, e.g. "Raaid, what programs do you offer?"
        m_vocative = re.match(r"^\s*([A-Za-z]{2,20})\s*[,.:;-]\s*(.+)$", message)
        if m_vocative:
            cand_token = m_vocative.group(1).strip().lower()
            rem = m_vocative.group(2).strip()
            if (
                cand_token not in ("how", "what", "why", "when", "where", "who", "which", "can", "do", "does", "is", "are")
                and re.search(r"^(what|how|why|when|where|tell me|explain|can you|do you|which|i want|book)\b", rem, re.I)
            ):
                message = rem

        # Explicit intro + question: "My name is Raaid, tell me about Grade 7"
        m_intro_q = re.match(r"^\s*(?:my\s+name\s+is|name\s+is|i\s+am|i'm|call\s+me)\s+([A-Za-z]{2,20})\s*[,.:;-]\s*(.+)$", message, re.I)
        if m_intro_q:
            cand_name = m_intro_q.group(1).strip().capitalize()
            rem_q = m_intro_q.group(2).strip()
            if cand_name.lower() not in leads._NAME_BLACKLIST and len(rem_q) >= 3:
                message = rem_q
                self._pending_intro_prefix = f"Nice to meet you, {cand_name}! "

        # Standalone discourse markers
        if _STANDALONE_ACTUALLY_RE.match(message):
            res = ReplyResult(
                "Go ahead! What would you like to know or explore about WeMentors?",
                "discourse_marker",
                ["programs-overview"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if _STANDALONE_WAIT_RE.match(message):
            res = ReplyResult(
                "Take your time! What would you like to check or ask?",
                "conversational_continuation",
                ["programs-overview"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        # Conversational questions
        if _WHY_QUERY_RE.match(message):
            res = ReplyResult(
                "Could you clarify what you'd like to know more about? I'm happy to explain our mentoring methodology, curriculum, or programs.",
                "question",
                ["programs-overview", "personalized-mentoring-concept"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if _REALLY_QUERY_RE.match(message):
            res = ReplyResult(
                "Yes, absolutely! WeMentors pairs students with dedicated 1-on-1 personal mentors who customize the learning plan to their pace and goals. Would you like to know more about how it works?",
                "conversational_question",
                ["programs-overview", "personalized-mentoring-concept"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        # Conversational reactions & pure acknowledgements
        if _CONVERSATIONAL_REACTION_RE.match(message):
            if re.search(r"\b(makes\s+sense|that'?s\s+helpful|helpful|i\s+see|fair\s+enough)\b", message, re.I):
                res = ReplyResult(
                    "Glad to help! What would you like to explore next — our academic programs (Grades 3–10) or Confident Speaker?",
                    "acknowledgement",
                    ["programs-overview"],
                    1.0,
                )
            else:
                res = ReplyResult(
                    "Glad you found that interesting! Let me know if you have any questions about our programs or mentors.",
                    "conversational_reaction",
                    ["programs-overview"],
                    1.0,
                )
            return self._finalize_result(session_id, res, memory, message)

        if _PURE_ACKNOWLEDGEMENT_RE.match(message):
            last_msg = self._last_assistant_message(session_id) or ""
            if last_msg and re.search(r"\b(what name|your name|student or parent name|name should i use)\b", last_msg, re.I):
                return self._finalize_result(
                    session_id,
                    ReplyResult(
                        "No worries. If you'd like to book a demo, you can enter your details through "
                        "the **Book Free Demo** form at the top-right of the website.",
                        "demo_booking",
                        ["how-to-book-demo"],
                        None,
                    ),
                    memory,
                    message,
                )
            res = ReplyResult(
                "Glad to help! What would you like to explore next — our academic programs (Grades 3–10) or Confident Speaker?",
                "acknowledgement",
                ["programs-overview"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        # Standalone subjects
        if _MATHS_STANDALONE_RE.match(message):
            reply = (
                "WeMentors provides personalized 1-on-1 mentoring in **Mathematics** across all school stages:\n\n"
                "- **Foundation Years (Grades 3–5)**: Builds strong number sense, mental math, and problem-solving confidence.\n"
                "- **Middle School (Grades 6–8)**: Deepens conceptual understanding with weekly doubt clinics and practical labs.\n"
                "- **Grades 9–10 Board Preparation**: Focused board exam prep, past paper practice, and score improvement.\n\n"
                "Which grade or program would you like to know more about?"
            )
            res = ReplyResult(reply, "subject_maths", ["subjects-offered", "programs-overview"], 1.0)
            return self._finalize_result(session_id, res, memory, message)

        if _SCIENCE_STANDALONE_RE.match(message):
            reply = (
                "WeMentors provides personalized 1-on-1 mentoring in **Science** across all school stages:\n\n"
                "- **Foundation Years (Grades 3–5)**: Hands-on exploration and core scientific curiosity.\n"
                "- **Middle School (Grades 6–8)**: Physics, Chemistry, and Biology concepts with weekly doubt clinics.\n"
                "- **Grades 9–10 Board Preparation**: Rigorous syllabus coverage and exam-focused mentorship.\n\n"
                "Which grade or program would you like to know more about?"
            )
            res = ReplyResult(reply, "subject_science", ["subjects-offered", "programs-overview"], 1.0)
            return self._finalize_result(session_id, res, memory, message)

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
        if _CAPABILITY_RE.match(message) or _CAPABILITY_RE.match(normalize_query(message)):
            return ReplyResult(personality.CAPABILITY_RESPONSE, "capability", ["programs-overview", "how-to-book-demo"], 1.0)

        # Single-word vague information request ("information", "info", "details")
        if _VAGUE_INFO_RE.match(message):
            return ReplyResult(personality.VAGUE_INFO_CLARIFICATION, "vague_info", ["programs-overview", "how-to-book-demo"], 1.0)

        # Transactional demo requests ("book a demo for me", "can you book it", "register me")
        # The chatbot is a booking GUIDE, not a transactional booking agent.
        if _DEMO_TRANSACTION_RE.search(message):
            res = ReplyResult(
                personality.DEMO_TRANSACTION_REQUEST_RESPONSE,
                "demo_transaction_request",
                ["how-to-book-demo", "contact-info"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        # Ambiguous form field words (e.g. "name", "email", "phone", "time")
        # Must never start a fake form or treat "name" as a person's name!
        m_field = _DEMO_FIELD_QUERY_RE.match(message)
        if m_field:
            field_word = m_field.group(1).lower()
            last_intent = self._last_assistant_intent(session_id)
            last_msg = self._last_assistant_message(session_id) or ""
            is_demo_context = (
                last_intent in ("demo_booking", "demo_inquiry", "demo_information", "demo_transaction_request")
                or bool(re.search(r"\b(demo|trial|book free demo)\b", last_msg, re.I))
            )
            if is_demo_context:
                if "name" in field_word:
                    res = ReplyResult(personality.DEMO_NAME_CLARIFICATION_RESPONSE, "name_clarification", ["how-to-book-demo"], 1.0)
                else:
                    res = ReplyResult(personality.DEMO_FIELD_CLARIFICATION_RESPONSE, "demo_field_clarification", ["how-to-book-demo"], 1.0)
                return self._finalize_result(session_id, res, memory, message)
            else:
                res = ReplyResult("Could you clarify what you'd like to know about our programs or mentoring?", "clarify", [], None)
                return self._finalize_result(session_id, res, memory, message)

        # Name detection (explicit intro e.g. "my name is raaid", "name is raaid", or standalone "Raaid" WITH context)
        # Recognition of a name is NOT permission to create a transactional state or fake lead.
        extracted_name = None
        is_explicit_intro = False
        if (
            not _FRUSTRATED_RE.search(message)
            and not _SHORT_CONFUSION_RE.match(message)
            and not _OUT_OF_SCOPE_RE.search(message)
            and not _ELIGIBILITY_ADULT_RE.search(message)
            and not _ELIGIBILITY_NON_SCHOOL_RE.search(message)
            and not _ELIGIBILITY_COLLEGE_RE.search(message)
            and not _ELIGIBILITY_GENERAL_RE.match(message)
            and not _LOCATION_RE.search(message)
            and not _ONLINE_CLASSES_RE.search(message)
        ):
            m_intro = _EXPLICIT_NAME_RE.match(message) or _NAME_IS_MY_NAME_RE.match(message)
            if m_intro:
                cand = m_intro.group(1).strip()
                cand_words = cand.split()
                if (
                    cand_words
                    and cand_words[0].lower() not in ("a", "an", "the")
                    and not any(w.lower() in leads._NAME_BLACKLIST for w in cand_words)
                    and not any(p.search(cand) for p, _ in leads._SUBJECT_PATTERNS)
                    and not any(p.search(cand) for p in leads._GRADE_PATTERNS)
                ):
                    extracted_name = cand_words[0].capitalize() if len(cand_words) == 1 else " ".join(w.capitalize() for w in cand_words)
                    is_explicit_intro = True

            if not extracted_name:
                last_intent = self._last_assistant_intent(session_id)
                last_msg = self._last_assistant_message(session_id) or ""
                if last_intent in ("cancellation", "never_mind"):
                    is_demo_context = False
                else:
                    is_demo_context = (
                        last_intent in ("demo_booking", "demo_inquiry", "demo_information", "demo_transaction_request", "demo_field_clarification", "standalone_name")
                        or bool(re.search(r"\b(book free demo|book a demo)\b", last_msg, re.I))
                    )
                asked_for_name = bool(re.search(r"\b(what('s| is) your name|may i (know|have) your name|can i have your name|your name\??)\b", last_msg, re.I))

                words = message.split()
                is_known_non_name = bool(
                    _GREETING_START_RE.match(message)
                    or _GOODBYE_RE.match(message)
                    or _THANKS_RE.match(message)
                    or _HELP_RE.match(message)
                    or _SHORT_CONFUSION_RE.match(message)
                    or _OUT_OF_SCOPE_RE.search(message)
                    or _PLAYFUL_RE.search(message)
                    or _CAPABILITY_RE.match(message)
                    or _GENERAL_INFO_RE.match(message)
                    or _BOOK_ENROLL_RE.match(message)
                    or _FRUSTRATED_RE.search(message)
                    or re.search(r"\b(foundation|found|foundating|doundation|foudation|foundaton|middle|senior|confident|speaker|program|programs|course|courses|curriculum|subject|subjects|grade|grades|class|standard|fee|fees|cost|costs|pricing|price|demo|trial|enroll|enrollment|apply|admissions?|clinic|clinics|dashboard|mentor|mentors|mentoring)\b", message, re.I)
                )
                if (
                    len(words) <= 3
                    and re.match(r"^[A-Za-z\s]+$", message)
                    and not is_known_non_name
                    and message.lower() not in leads._NAME_BLACKLIST
                    and not (set(message.lower().split()) & leads._NAME_BLACKLIST)
                    and not any(p.search(message) for p, _ in leads._SUBJECT_PATTERNS)
                    and not any(p.search(message) for p in leads._GRADE_PATTERNS)
                    and not any(p.search(message) for p in leads._TIME_PATTERNS)
                    and not re.search(r"[?]|^(what|how|why|who|where|when|tell me|explain|can you|do you|which|i want|book)\b", message, re.I)
                ):
                    # Standalone names are accepted ONLY when conversational context actively supports it
                    if is_demo_context or asked_for_name:
                        extracted_name = message.strip().title()
                    else:
                        # Low confidence / ambiguous standalone word: do NOT echo as a name!
                        # Handle conservatively with natural clarification:
                        res = ReplyResult(
                            "Could you clarify what you'd like to know about our programs or mentoring? How can I help you today?",
                            "clarify",
                            ["programs-overview"],
                            None,
                        )
                        return self._finalize_result(session_id, res, memory, message)

        if extracted_name:
            last_intent = self._last_assistant_intent(session_id)
            last_msg = self._last_assistant_message(session_id) or ""
            if last_intent in ("cancellation", "never_mind"):
                is_demo_context = False
            else:
                is_demo_context = (
                    last_intent in ("demo_booking", "demo_inquiry", "demo_information", "demo_transaction_request", "demo_field_clarification", "standalone_name")
                    or bool(re.search(r"\b(book free demo|book a demo)\b", last_msg, re.I))
                )
            if is_demo_context:
                reply = personality.STANDALONE_NAME_DEMO_RESPONSE.format(name=extracted_name)
                res = ReplyResult(reply, "standalone_name", ["how-to-book-demo"], 1.0)
            else:
                reply = f"Nice to meet you, {extracted_name}! How can I help you today with WeMentors' programs and mentoring?"
                res = ReplyResult(reply, "explicit_name" if is_explicit_intro else "greeting", ["programs-overview"], 1.0)
            return self._finalize_result(session_id, res, memory, message)

        # Contact details provided (phone number or email address)
        # The chatbot must NOT collect contact details or claim to save them.
        if (leads._PHONE_RE.search(message) or leads._EMAIL_RE.search(message)) and not re.search(r"\b(what|how|where|when|why)\b", message, re.I):
            last_intent = self._last_assistant_intent(session_id)
            last_msg = self._last_assistant_message(session_id) or ""
            is_demo_context = (
                last_intent in ("demo_booking", "demo_inquiry", "demo_information", "demo_transaction_request", "demo_field_clarification", "standalone_name")
                or bool(re.search(r"\b(demo|trial|book free demo)\b", last_msg, re.I))
            )
            if is_demo_context:
                reply = (
                    "Got it. If that's for a demo, please enter your contact details through the "
                    "**Book Free Demo** form at the top-right of the website."
                )
                res = ReplyResult(reply, "demo_booking", ["how-to-book-demo"], 1.0)
            else:
                reply = (
                    "If you'd like our team to contact you or if you're booking a demo, "
                    "please enter your details through the **Book Free Demo** form at the top-right of the website, "
                    "or contact us directly at **admin@wementors.co** or **+91 76111 92227**."
                )
                res = ReplyResult(reply, "contact_info", ["contact-info", "how-to-book-demo"], 1.0)
            return self._finalize_result(session_id, res, memory, message)

        # Narrow grade questions: direct and concise without full program dump
        if _FOUNDATION_GRADES_NARROW_RE.match(message):
            res = ReplyResult(personality.FOUNDATION_YEARS_GRADES_RESPONSE, "foundation_grades", ["program-foundation-years"], 1.0)
            return self._finalize_result(session_id, res, memory, message)
        if _MIDDLE_SCHOOL_GRADES_NARROW_RE.match(message):
            res = ReplyResult(personality.MIDDLE_SCHOOL_GRADES_NARROW_RESPONSE, "middle_school_grades", ["program-middle-school"], 1.0)
            return self._finalize_result(session_id, res, memory, message)
        if _IS_ONE_TO_ONE_RE.match(message):
            res = ReplyResult(personality.IS_ONE_TO_ONE_RESPONSE, "one_on_one_general", ["program-senior-school", "program-confident-speaker"], 1.0)
            return self._finalize_result(session_id, res, memory, message)

        # Where to book
        if _DEMO_WHERE_TO_BOOK_RE.search(message):
            res = ReplyResult(personality.DEMO_BOOKING_RESPONSE, "demo_booking", ["how-to-book-demo"], 1.0)
            return self._finalize_result(session_id, res, memory, message)

        # How to book a demo ("how do i book", "how to book a demo")
        if _HOW_TO_BOOK_RE.match(message):
            res = ReplyResult(personality.HOW_TO_BOOK_RESPONSE, "demo_booking", ["how-to-book-demo"], 1.0)
            return self._finalize_result(session_id, res, memory, message)

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
        # (No longer re-asks for name since we don't collect booking info in chat)
        if leads.is_pure_acknowledgement(message) and last_assistant_msg:
            if re.search(r"\b(what name|your name|student or parent name|name should i use)\b", last_assistant_msg, re.IGNORECASE):
                return ReplyResult(
                    "No worries. If you'd like to book a demo, you can enter your details through "
                    "the **Book Free Demo** form at the top-right of the website.",
                    "demo_booking",
                    ["how-to-book-demo"],
                    None,
                )

        # Out-of-scope questions
        if _OUT_OF_SCOPE_RE.search(message) and not any(k in message.lower() for k in ["class", "mentor", "course", "subject", "demo", "wementors"]):
            return ReplyResult(personality.OFF_TOPIC_RESPONSE, "off_topic", [], None)

        # Handle casual 'never mind' / cancellation
        if re.search(r"^\s*(never\s*mind|nevermind|no\s*worries|no\s*thanks|forget\s*it)\s*[!.]*\s*$", message, re.IGNORECASE):
            # Clear any stale demo lead state
            database.clear_demo_lead(session_id)
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

        # NOTE: The old active-lead interception block has been removed.
        # The chatbot no longer enters a multi-turn lead-collection workflow.
        # All demo-related intents are handled by the intent routing below,
        # which provides guidance to the Book Free Demo form.

        intent = self.detect_intent(message, session_id)

        # Handle relative/ordinal reference and corrections via conversation memory
        if intent == "reference_resolution" or self._extract_ordinal_index(message) is not None:
            ref_res = self._resolve_relative_or_correction(message, memory, session_id)
            if ref_res:
                return self._finalize_result(session_id, ref_res, memory, message)

        if intent == "personalized_mentoring_general":
            res = ReplyResult(
                personality.PERSONALIZED_MENTORING_GENERAL_RESPONSE,
                "personalized_mentoring_general",
                ["personalized-mentoring-concept"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "mentoring_approach":
            res = ReplyResult(
                personality.MENTORING_APPROACH_RESPONSE,
                "mentoring_approach",
                ["personalized-mentoring-concept"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "eligibility_mixed_english":
            memory.active_program = "confident_speaker"
            res = ReplyResult(
                personality.ELIGIBILITY_ENGLISH_RESPONSE,
                "eligibility_mixed_english",
                ["program-confident-speaker", "confident-speaker-audience"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "eligibility_mixed_online":
            res = ReplyResult(
                personality.ELIGIBILITY_ONLINE_RESPONSE,
                "eligibility_mixed_online",
                ["programs-overview", "confident-speaker-audience"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "eligibility_mixed_programs":
            res = ReplyResult(
                personality.ELIGIBILITY_PROGRAMS_RESPONSE,
                "eligibility_mixed_programs",
                ["programs-overview", "confident-speaker-audience"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "eligibility_mixed_fees":
            res = ReplyResult(
                personality.ELIGIBILITY_FEES_RESPONSE,
                "eligibility_mixed_fees",
                ["fee-structure", "confident-speaker-audience"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "eligibility_non_school":
            res = ReplyResult(
                personality.ELIGIBILITY_NON_SCHOOL_RESPONSE,
                "eligibility_non_school",
                ["confident-speaker-audience", "programs-overview"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "eligibility_adult":
            res = ReplyResult(
                personality.ELIGIBILITY_ADULT_RESPONSE,
                "eligibility_adult",
                ["confident-speaker-audience", "programs-overview"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "eligibility_college":
            res = ReplyResult(
                personality.ELIGIBILITY_COLLEGE_RESPONSE,
                "eligibility_college",
                ["confident-speaker-audience", "programs-overview"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "eligibility_general":
            res = ReplyResult(
                personality.ELIGIBILITY_GENERAL_RESPONSE,
                "eligibility_general",
                ["programs-overview", "confident-speaker-audience"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "location":
            res = ReplyResult(
                personality.LOCATION_RESPONSE,
                "location",
                ["contact-info"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "online_classes":
            res = ReplyResult(
                personality.ONLINE_CLASSES_RESPONSE,
                "online_classes",
                ["programs-overview"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "foundation_enrollment":
            memory.active_program = "foundation"
            res = ReplyResult(
                personality.FOUNDATION_YEARS_ENROLLMENT_RESPONSE,
                "foundation_enrollment",
                ["how-to-book-demo", "program-foundation-years"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "middle_enrollment":
            memory.active_program = "middle"
            res = ReplyResult(
                personality.MIDDLE_SCHOOL_ENROLLMENT_RESPONSE,
                "middle_enrollment",
                ["how-to-book-demo", "program-middle-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "senior_enrollment":
            memory.active_program = "senior"
            res = ReplyResult(
                personality.SENIOR_SCHOOL_ENROLLMENT_RESPONSE,
                "senior_enrollment",
                ["how-to-book-demo", "program-senior-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "confident_speaker_enrollment":
            memory.active_program = "confident_speaker"
            res = ReplyResult(
                personality.CONFIDENT_SPEAKER_ENROLLMENT_RESPONSE,
                "confident_speaker_enrollment",
                ["how-to-book-demo", "program-confident-speaker"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "general_enrollment_clarification":
            memory.pending_clarification = {
                "question": "Which program would you like to enroll in?",
                "expected_entity": None,
                "options": ["Foundation Years", "Middle School", "Grades 9–10", "Confident Speaker"],
                "created_at_turn": memory.turn_count,
            }
            res = ReplyResult(
                personality.GENERAL_ENROLLMENT_CLARIFICATION,
                "general_enrollment_clarification",
                ["programs-overview", "how-to-book-demo"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "clarification_affirmation":
            pending = memory.pending_clarification
            memory.pending_clarification = None
            expected = pending.get("expected_entity") if pending else None
            exp_str = str(expected).lower() if expected else ""
            q_str = str(pending.get("question", "")).lower() if pending else ""

            if "foundation" in exp_str or "foundation" in q_str:
                memory.active_program = "foundation"
                memory.confirmed_context = "Foundation Years"
                res = ReplyResult(
                    personality.FOUNDATION_CONFIRMATION_RESPONSE,
                    "foundation_confirmation",
                    ["program-foundation-years"],
                    1.0,
                )
                return self._finalize_result(session_id, res, memory, message)
            elif "middle" in exp_str or "middle" in q_str:
                memory.active_program = "middle"
                memory.confirmed_context = "Middle School"
                res = ReplyResult(
                    "Great — I'll use Middle School (Grades 6–8) as the program we're discussing. You can ask about its subjects, doubt clinics, practical labs, or how to enroll.",
                    "middle_school_confirmation",
                    ["program-middle-school"],
                    1.0,
                )
                return self._finalize_result(session_id, res, memory, message)
            elif "senior" in exp_str or "9" in exp_str or "10" in exp_str or "senior" in q_str:
                memory.active_program = "senior"
                memory.confirmed_context = "Grades 9–10"
                res = ReplyResult(
                    "Great — I'll use Grades 9–10 board exam preparation as the program we're discussing. You can ask about its subjects, mentoring, parent updates, or how to enroll.",
                    "senior_school_confirmation",
                    ["program-senior-school"],
                    1.0,
                )
                return self._finalize_result(session_id, res, memory, message)
            elif "confident" in exp_str or "speaker" in exp_str or "confident" in q_str:
                memory.active_program = "confident_speaker"
                memory.confirmed_context = "Confident Speaker"
                res = ReplyResult(
                    personality.CONFIDENT_SPEAKER_CONFIRMATION_RESPONSE,
                    "confident_speaker_confirmation",
                    ["program-confident-speaker"],
                    1.0,
                )
                return self._finalize_result(session_id, res, memory, message)
            res = ReplyResult(
                personality.CONFIRMATION_WITHOUT_CLARIFICATION_RESPONSE,
                "confirmation_without_clarification",
                [],
                None,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "confirmation_without_clarification":
            res = ReplyResult(
                personality.CONFIRMATION_WITHOUT_CLARIFICATION_RESPONSE,
                "confirmation_without_clarification",
                [],
                None,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent in ("clarification_negation", "negation_general"):
            if intent == "clarification_negation":
                pending = memory.pending_clarification
                memory.pending_clarification = None
                options = pending.get("options", []) if pending else []
                remaining = [opt for opt in options if opt != (pending.get("expected_entity") if pending else None)]
                if remaining:
                    reply = f"Understood. What would you like to explore instead — {', '.join(remaining)}?"
                else:
                    reply = personality.CONFIRMATION_NEGATIVE_RESPONSE
            else:
                reply = personality.CONFIRMATION_NEGATIVE_RESPONSE
            res = ReplyResult(reply, intent, [], None)
            return self._finalize_result(session_id, res, memory, message)

        if intent == "clarification_other_option":
            pending = memory.pending_clarification
            memory.pending_clarification = None
            options = pending.get("options", []) if pending else []
            remaining = [opt for opt in options if opt != (pending.get("expected_entity") if pending else None)]
            if remaining:
                next_choice = remaining[0]
                prog_id = self._program_name_to_id(next_choice)
                if prog_id:
                    memory.active_program = self._program_id_to_key(prog_id)
                    return self._finalize_result(session_id, self._reply_for_program_id(prog_id, session_id), memory, message)
            res = ReplyResult(personality.VAGUE_MENU_RESPONSE, "clarification_other_option", [], None)
            return self._finalize_result(session_id, res, memory, message)

        if intent == "acknowledgement":
            res = ReplyResult(personality.OKAY_CONFIRMATION_RESPONSE, "acknowledgement", [], None)
            return self._finalize_result(session_id, res, memory, message)

        if intent == "foundation_subjects":
            memory.active_program = "foundation"
            res = ReplyResult(
                personality.FOUNDATION_YEARS_SUBJECTS_RESPONSE,
                "foundation_subjects",
                ["program-foundation-years"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "foundation_approach":
            memory.active_program = "foundation"
            res = ReplyResult(
                personality.FOUNDATION_YEARS_TEACHING_APPROACH_RESPONSE,
                "foundation_approach",
                ["program-foundation-years"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "foundation_progress":
            memory.active_program = "foundation"
            res = ReplyResult(
                personality.FOUNDATION_YEARS_PROGRESS_RESPONSE,
                "foundation_progress",
                ["program-foundation-years"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "foundation_years_clarification":
            memory.pending_clarification = {
                "question": "foundation_years_clarification",
                "expected_entity": "Foundation Years",
                "options": ["Foundation Years", "Middle School", "Grades 9–10", "Confident Speaker"],
                "created_at_turn": memory.turn_count,
            }
            res = ReplyResult(
                personality.FOUNDATION_YEARS_CLARIFICATION,
                "foundation_years_clarification",
                ["program-foundation-years"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "foundation_years_overview":
            memory.active_program = "foundation"
            res = ReplyResult(
                personality.FOUNDATION_YEARS_DIRECT_RESPONSE,
                "foundation_years_overview",
                ["program-foundation-years"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "middle_school_overview":
            res = ReplyResult(
                personality.MIDDLE_SCHOOL_OVERVIEW_RESPONSE,
                "middle_school_overview",
                ["program-middle-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "middle_school_grades":
            res = ReplyResult(
                personality.MIDDLE_SCHOOL_GRADES_RESPONSE,
                "middle_school_grades",
                ["program-middle-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "middle_school_subjects":
            res = ReplyResult(
                personality.MIDDLE_SCHOOL_SUBJECTS_RESPONSE,
                "middle_school_subjects",
                ["middle-school-subjects", "program-middle-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "middle_school_doubt_clinics":
            res = ReplyResult(
                personality.MIDDLE_SCHOOL_DOUBT_CLINICS_RESPONSE,
                "middle_school_doubt_clinics",
                ["middle-school-doubt-clinics", "program-middle-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "middle_school_practical_labs":
            res = ReplyResult(
                personality.MIDDLE_SCHOOL_PRACTICAL_LABS_RESPONSE,
                "middle_school_practical_labs",
                ["middle-school-practical-labs", "program-middle-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "middle_school_progress_dashboard":
            res = ReplyResult(
                personality.MIDDLE_SCHOOL_PROGRESS_DASHBOARD_RESPONSE,
                "middle_school_progress_dashboard",
                ["middle-school-progress-dashboard", "program-middle-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "general_info":
            return ReplyResult(
                personality.GENERAL_INFO_DIRECT_RESPONSE,
                "general_info",
                ["what-is-wementors"],
                1.0,
            )

        if intent == "beginner_recommendation":
            return ReplyResult(
                personality.BEGINNER_RECOMMENDATION_RESPONSE,
                "beginner_recommendation",
                ["programs-overview"],
                1.0,
            )

        if intent == "board_exam":
            entry = self.entries_by_id.get("program-senior-school")
            if entry:
                scored = [ScoredEntry(entry=entry, score=1.0)]
                answer = self._generate_answer(message, scored, session_id)
                answer = self._run_response_quality_checks(message, answer, "board_exam", [entry], memory.last_assistant_message)
                res = ReplyResult(answer, "board_exam", ["program-senior-school"], 1.0)
                return self._finalize_result(session_id, res, memory, message)

        if intent == "confident_speaker":
            entry = self.entries_by_id.get("program-confident-speaker")
            if entry:
                scored = [ScoredEntry(entry=entry, score=1.0)]
                answer = self._generate_answer(message, scored, session_id)
                answer = self._run_response_quality_checks(message, answer, "confident_speaker", [entry], memory.last_assistant_message)
                res = ReplyResult(answer, "confident_speaker", ["program-confident-speaker"], 1.0)
                return self._finalize_result(session_id, res, memory, message)

        if intent == "confident_speaker_format":
            last_intent = self._last_assistant_intent(session_id)
            if last_intent == "confident_speaker_format":
                res = ReplyResult(
                    personality.CONFIDENT_SPEAKER_FORMAT_CONCISE,
                    "confident_speaker_format",
                    ["confident-speaker-format"],
                    1.0,
                )
            else:
                res = ReplyResult(
                    personality.CONFIDENT_SPEAKER_FORMAT_DIRECT,
                    "confident_speaker_format",
                    ["confident-speaker-format"],
                    1.0,
                )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "confident_speaker_audience":
            res = ReplyResult(
                personality.CONFIDENT_SPEAKER_AUDIENCE_DIRECT,
                "confident_speaker_audience",
                ["confident-speaker-audience"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "confident_speaker_scope":
            res = ReplyResult(
                personality.CONFIDENT_SPEAKER_SCOPE_DIRECT,
                "confident_speaker_scope",
                ["confident-speaker-scope"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "confident_speaker_mentoring":
            res = ReplyResult(
                personality.CONFIDENT_SPEAKER_MENTORING_DIRECT,
                "confident_speaker_mentoring",
                ["confident-speaker-mentoring"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "confident_speaker_activities":
            res = ReplyResult(
                personality.CONFIDENT_SPEAKER_ACTIVITIES_DIRECT,
                "confident_speaker_activities",
                ["confident-speaker-activities"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "academic_sessions":
            res = ReplyResult(
                personality.ACADEMIC_SESSIONS_RESPONSE,
                "academic_sessions",
                ["mentoring-process", "teaching-approach", "one-on-one-available"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "academic_courses_overview":
            res = ReplyResult(
                personality.ACADEMIC_COURSES_OVERVIEW_RESPONSE,
                "academic_courses_overview",
                ["programs-overview", "program-foundation-years", "program-middle-school", "program-senior-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "grade7_maths_sessions":
            res = ReplyResult(
                personality.GRADE7_MATHS_SESSIONS_RESPONSE,
                "grade7_maths_sessions",
                ["program-middle-school", "middle-school-subjects"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "parent_updates":
            res = ReplyResult(
                personality.PARENT_UPDATES_RESPONSE,
                "parent_updates",
                ["program-senior-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent in ("board_mentoring", "one_on_one_general"):
            res = ReplyResult(
                personality.ONE_ON_ONE_GENERAL_RESPONSE,
                intent,
                ["program-senior-school", "program-confident-speaker"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "marks_guarantee":
            res = ReplyResult(
                personality.GUARANTEE_MARKS_RESPONSE,
                "marks_guarantee",
                ["program-senior-school"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "confirmation":
            res = ReplyResult(
                personality.OKAY_CONFIRMATION_RESPONSE,
                "confirmation",
                [],
                None,
            )
            return self._finalize_result(session_id, res, memory, message)

        if intent == "unclear_format":
            res = ReplyResult(
                "Which program would you like to know the format for — our Grades 9–10 learning support or the Confident Speaker program?",
                "unclear_format",
                ["programs-overview"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)

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
        if intent == "demo_transaction_request":
            return ReplyResult(
                personality.DEMO_TRANSACTION_REQUEST_RESPONSE,
                "demo_transaction_request",
                ["how-to-book-demo", "contact-info"],
                1.0,
            )
        if intent == "demo_information":
            if re.search(r"\b(happen|happens|during|experience|include|diagnostic)\b", message, re.I):
                entry = self.entries_by_id.get("demo-class-what-happens")
            elif re.search(r"\b(long|duration|length|minutes?|30\s*mins?)\b", message, re.I):
                entry = self.entries_by_id.get("demo-class-length")
            else:
                entry = self.entries_by_id.get("demo-class-available")
            if entry:
                return ReplyResult(entry.answer, "demo_information", [entry.id, "how-to-book-demo"], 1.0)
            return ReplyResult(personality.DEMO_BOOKING_RESPONSE, "demo_information", ["how-to-book-demo"], 1.0)
        if intent in ("demo_booking", "demo_booking_instructions"):
            # Guide user to the website form — never collect booking info in chat
            res = ReplyResult(
                personality.DEMO_BOOKING_RESPONSE,
                "demo_booking",
                ["contact-info", "how-to-book-demo"],
                1.0,
            )
            return self._finalize_result(session_id, res, memory, message)
        if intent == "foundation_grades":
            return ReplyResult(
                personality.FOUNDATION_YEARS_GRADES_RESPONSE,
                "foundation_grades",
                ["program-foundation-years"],
                1.0,
            )
        if intent == "middle_school_grades":
            return ReplyResult(
                personality.MIDDLE_SCHOOL_GRADES_NARROW_RESPONSE,
                "middle_school_grades",
                ["program-middle-school"],
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
            fallback_text = self._get_fallback_reply(message)
            return ReplyResult(fallback_text, "low_confidence", [], top_score)

        # For multi-clause questions, keep the multiple matched entries; otherwise keep the best single match.
        if referenced_entry is not None:
            best = scored
        elif len(multi_scored) >= 2:
            best = scored
        else:
            best = scored[:1]

        answer = self._generate_answer(message, best, session_id)
        matched_entries = [item.entry for item in best]
        answer = self._run_response_quality_checks(message, answer, intent, matched_entries)
        matched_ids = [item.entry.id for item in best]
        resolved_intent = "multi_intent" if len(multi_scored) >= 2 else ("faq" if intent == "faq" else intent)
        return self._finalize_result(
            session_id,
            ReplyResult(answer, resolved_intent, matched_ids, top_score),
            memory,
            message,
        )
