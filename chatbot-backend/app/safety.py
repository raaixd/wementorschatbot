import re
from typing import Tuple

INJECTION_PATTERNS = [
    r"ignore (all|any|previous|prior|above) (instructions|prompts|rules)",
    r"you are now",
    r"system prompt",
    r"reveal (your )?(hidden|secret|internal) (instructions|prompt)",
    r"developer mode",
    r"jailbreak",
    r"pretend you (are|have) no restrictions",
]


def sanitize_user_text(text: str, max_length: int) -> str:
    cleaned = " ".join((text or "").replace("\x00", " ").split())
    return cleaned[:max_length]


def looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def strip_injection_noise(text: str) -> str:
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def validate_reply(reply: str, source_text: str) -> str:
    """Block obvious fabrications such as currency amounts not present in sources."""
    if re.search(r"(₹|rs\.?|inr|usd|\$)\s?\d", reply, flags=re.IGNORECASE):
        if not re.search(r"(₹|rs\.?|inr|usd|\$)\s?\d", source_text, flags=re.IGNORECASE):
            return (
                "Verified fee amounts are not in the WeMentors knowledge base. "
                "Please contact the team at +91 76111 92227 or admin@wementors.co for current fees."
            )
    return reply


def split_intent_and_data(message: str) -> Tuple[bool, str]:
    if looks_like_injection(message):
        return True, strip_injection_noise(message)
    return False, message
