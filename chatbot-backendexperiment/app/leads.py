"""
Demo enquiry lead helpers and guidance.

Capability architecture:
- DEMO_TRANSACTION_ENABLED controls whether the chatbot can collect,
  store, and submit demo booking leads through conversational chat.
- When False (current default), the chatbot understands demo-related
  user intent but NEVER collects personal information, saves leads,
  or claims a submission occurred.  It redirects users to the actual
  Book Free Demo form on the website.
- When True (future), the full multi-turn collection state machine
  in process_demo_flow() is activated.

Utility functions (is_pure_acknowledgement, is_cancellation, etc.) and
the _NAME_BLACKLIST are used by conversation.py for intent detection
and are always available regardless of the capability flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import database

# ── Capability gate ──────────────────────────────────────────────────
# Set to True ONLY when a real backend submission mechanism exists
# (e.g. Google Sheets API, CRM webhook, verified email sender).
# While False, the chatbot will guide users to the website form and
# NEVER pretend to collect, store, or submit booking information.
DEMO_TRANSACTION_ENABLED = False


_ACKNOWLEDGEMENT_RE = re.compile(
    r"^\s*(ok+|okay+|sure+|yes+|yeah+|yep+|alright+|fine+|got it|thanks|thank you|understood|i will|maybe|cool|sounds good|noted|k)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    r"\b(cancel|never\s*mind|nevermind|nvm|forget\s*it|forget\s*that|leave\s*it|not\s*interested|no\s*thanks|stop|don'?t\s*want|doesn't\s*matter|doesnt\s*matter|no\s*worries|that's\s*okay|thats\s*ok|it's\s*fine|its\s*fine|i\s*changed\s*my\s*mind|ignore\s*that|skip\s*that|let's\s*forget\s*it)\b",
    re.IGNORECASE,
)

_IS_BOOKED_QUERY_RE = re.compile(
    r"\b(is (it|my demo|the demo) (booked|scheduled|confirmed|submitted)|did (you|it) (book|schedule|submit))\b",
    re.IGNORECASE,
)

_CONFIRM_INTENT_RE = re.compile(
    r"\b(yes|confirm|submit|proceed|looks good|send it|please submit|go ahead|yes please|do it|book it|submit it|submit that)\b",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:\+?91[\s-]?)?[6-9]\d{9}\b|\+?\d{1,3}[-.\s]?(?:\d{3,5}[-.\s]?){2}\d{2,5}")

_GRADE_PATTERNS = [
    re.compile(r"\b(?:grade|class|standard)\s*([3-9]|10)\b", re.IGNORECASE),
    re.compile(r"\b([3-9]|10)(?:th|rd|nd|st)?\s*(?:grade|class|standard)\b", re.IGNORECASE),
    re.compile(r"\b([3-9]|10)th\b", re.IGNORECASE),
]

_SUBJECT_PATTERNS = [
    (re.compile(r"\b(math|maths|mathematics)\b", re.IGNORECASE), "Maths"),
    (re.compile(r"\b(science|physics|chemistry|biology)\b", re.IGNORECASE), "Science"),
    (re.compile(r"\b(confident speaker|spoken english|public speaking|business english|general communicative|communicative skills?|communication skills?|ielts|english|speaking)\b", re.IGNORECASE), "Confident Speaker"),
    (re.compile(r"\b(foundation years?)\b", re.IGNORECASE), "Foundation Years"),
    (re.compile(r"\b(middle school)\b", re.IGNORECASE), "Middle School"),
    (re.compile(r"\b(senior school)\b", re.IGNORECASE), "Senior School Focus"),
]

_TIME_PATTERNS = [
    re.compile(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2}\s*(?:to|-)\s*\d{1,2}\s*(?:am|pm))\b", re.IGNORECASE),
    re.compile(r"\b(morning|afternoon|evening|weekend|weekday|anytime|flexible)\b", re.IGNORECASE),
]

_NAME_PATTERNS = [
    re.compile(r"\b(?:my name is|i am|i'm|this is|call me|name is)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.IGNORECASE),
    re.compile(r"\b(?:student|child|son|daughter)(?:'s)? name is\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.IGNORECASE),
]


def is_pure_acknowledgement(text: str) -> bool:
    """Return True if text is a simple acknowledgement without substantive detail."""
    return bool(_ACKNOWLEDGEMENT_RE.match(text.strip()))


def is_cancellation(text: str) -> bool:
    """Return True if the user asks to cancel the demo inquiry."""
    return bool(_CANCEL_RE.search(text.strip()))


def is_confirmation(text: str) -> bool:
    """Return True if the user confirms submitting demo details."""
    return bool(_CONFIRM_INTENT_RE.match(text.strip()))


def is_asking_if_booked(text: str) -> bool:
    """Return True if the user is asking whether their demo is already booked."""
    return bool(_IS_BOOKED_QUERY_RE.search(text.strip()))


@dataclass
class DemoLead:
    name: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    contact_method: Optional[str] = None
    contact_value: Optional[str] = None
    preferred_time: Optional[str] = None
    stage: str = "idle"  # idle | collecting | confirming | submitted | cancelled
    submission_status: str = "pending"  # pending | confirmed | failed

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "name": self.name,
            "grade": self.grade,
            "subject": self.subject,
            "contact_method": self.contact_method,
            "contact_value": self.contact_value,
            "preferred_time": self.preferred_time,
            "stage": self.stage,
            "submission_status": self.submission_status,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> DemoLead:
        if not data:
            return cls()
        return cls(
            name=data.get("name"),
            grade=data.get("grade"),
            subject=data.get("subject"),
            contact_method=data.get("contact_method"),
            contact_value=data.get("contact_value"),
            preferred_time=data.get("preferred_time"),
            stage=data.get("stage") or "idle",
            submission_status=data.get("submission_status") or "pending",
        )

    def has_any_field(self) -> bool:
        return any([self.name, self.grade, self.subject, self.contact_value, self.preferred_time])

    def get_missing_field_labels(self) -> List[str]:
        missing = []
        if not self.name:
            missing.append("student or parent name")
        if not self.grade:
            missing.append("student's grade")
        if not self.subject:
            missing.append("preferred subject")
        if not self.contact_value:
            missing.append("preferred phone number or email")
        if not self.preferred_time:
            missing.append("preferred time for the demo")
        return missing

    def is_complete(self) -> bool:
        # Minimum required fields to confirm/submit
        return bool(self.name and self.grade and self.subject and self.contact_value)


_NAME_BLACKLIST = {
    # Greetings & Salutations
    "hey", "hi", "hello", "namaste", "greetings", "yo", "good morning", "good afternoon", "good evening",
    # Acknowledgements & Confirmations
    "ok", "okay", "sure", "yes", "yeah", "yep", "alright", "fine", "got it", "cool", "sounds good",
    "thanks", "thank you", "thx", "no", "nope", "nah", "never", "cancel", "stop", "what", "why", "how",
    "who", "where", "when", "none", "nothing", "maybe",
    # Conversational control, cancellation & discourse signals (MUST NEVER be person names)
    "nevermind", "never mind", "nvm", "forget", "forget it", "forget that", "leave", "leave it",
    "actually", "really", "interesting", "sense", "makes sense", "helpful", "wait", "hmm", "hmmm",
    "changed", "mind", "matter", "doesn't matter", "doesnt matter", "worries", "no worries",
    "it's fine", "its fine", "that's okay", "thats okay", "thats ok", "that's ok", "ignore", "skip",
    "confused", "confusion", "sorry", "pardon", "i see",
    # Form fields and labels (MUST NEVER be treated as person names)
    "name", "names", "student name", "parent name", "full name", "first name", "last name",
    "email", "emails", "mail", "gmail",
    "phone", "phones", "number", "numbers", "phone number", "mobile", "mobile number", "contact", "contact number",
    "grade", "grades", "class", "classes", "standard", "standards",
    "subject", "subjects",
    "time", "times", "timing", "timings", "slot", "slots", "schedule", "date", "dates", "preferred time",
    "form", "forms", "field", "fields", "button", "link", "option", "website",
    # Programs & Scope
    "foundation", "years", "middle", "school", "senior", "confident", "speaker", "academy", "platform",
    "spoken", "public", "speaking", "interview", "skills", "focus",
    "ielts", "ietls", "prep", "preparation", "coaching", "communicative", "communication", "conversations", "conversation", "communicating",
    "looking", "preparing", "interested", "inquiring", "asking",
    # Academic & Subjects
    "math", "maths", "mathematics", "science", "physics", "chemistry", "biology", "english", "evs",
    "environmental", "studies", "history", "geography", "social", "board", "cbse", "icse", "curriculum", "syllabus", "exam", "exams",
    "course", "courses", "program", "programs", "student", "students", "parent", "parents",
    "child", "children", "kid", "kids", "learner", "learners", "tutor", "tuition", "mentor", "mentors", "mentoring",
    "adult", "adults", "an adult", "grown up", "professional", "professionals", "a professional", "homemaker", "homemakers", "a homemaker", "housewife", "college", "university",
    "an", "a", "the", "person", "someone", "first", "second",
    # Countries & Global
    "saudi", "arabia", "dubai", "uae", "qatar", "oman", "kuwait", "bahrain", "usa", "uk", "america", "global", "international",
    # Actions & Inquiry types
    "demo", "trial", "session", "sessions", "book", "booking", "enroll", "enrollment", "admission", "admissions",
    "register", "registration", "fee", "fees", "cost", "price", "pricing", "details", "info", "information",
    "help", "online", "offline", "live", "free", "wementors", "wementor", "python", "java", "coding",
    # Times & Days
    "morning", "afternoon", "evening", "night", "today", "tomorrow", "am", "pm", "weekend", "weekday",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "anytime", "flexible",
}


def extract_lead_fields(text: str, current: Optional[DemoLead] = None) -> Tuple[DemoLead, List[str]]:
    """Extract structured demo fields from user text, updating the current lead."""
    lead = DemoLead.from_dict(current.to_dict()) if current else DemoLead()
    newly_found: List[str] = []

    # 1. Contact (Email / Phone)
    email_match = _EMAIL_RE.search(text)
    if email_match:
        lead.contact_method = "email"
        lead.contact_value = email_match.group(0).strip()
        newly_found.append("contact email")

    phone_match = _PHONE_RE.search(text)
    if phone_match and not lead.contact_value:
        cleaned_phone = phone_match.group(0).strip()
        lead.contact_method = "phone"
        lead.contact_value = cleaned_phone
        newly_found.append("phone number")

    # 2. Grade
    for pat in _GRADE_PATTERNS:
        m = pat.search(text)
        if m:
            grade_num = m.group(1)
            lead.grade = f"Grade {grade_num}"
            newly_found.append(f"Grade {grade_num}")
            break

    # 3. Subject (handling "X instead of Y" or "X rather than Y")
    is_question = bool(re.search(r"[?]|^(what|how|why|tell me|explain|can you|do you|which|is)\b", text.strip(), re.I))
    if not is_question:
        subject_text = re.sub(r"\b(instead of|rather than|not)\s+[A-Za-z0-9\s]+$", "", text, flags=re.IGNORECASE)
        subject_text = re.sub(r"\b(instead of|rather than|not)\s+[A-Za-z0-9]+\b", "", subject_text, flags=re.IGNORECASE)
        for pat, subj in _SUBJECT_PATTERNS:
            if pat.search(subject_text):
                lead.subject = subj
                newly_found.append(subj)
                break

    # 4. Preferred Time
    for pat in _TIME_PATTERNS:
        m = pat.search(text)
        if m:
            lead.preferred_time = m.group(0).strip().capitalize()
            newly_found.append(f"time: {lead.preferred_time}")
            break

    # 5. Name
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            name_val = m.group(1).strip().title()
            name_words = set(name_val.lower().split())
            if name_val.lower() not in _NAME_BLACKLIST and not (name_words & _NAME_BLACKLIST):
                lead.name = name_val
                newly_found.append(f"name: {name_val}")
                break

    # Heuristic for single-line name when asked for a name:
    # 1. MUST NOT run if any other field (email, phone, grade, subject, time) was already extracted from this message.
    # 2. MUST NOT match if any word in the message matches a subject pattern, grade pattern, time pattern, or blacklist.
    # 3. MUST NOT match questions, general inquiries, or conversational commands.
    if not lead.name and not newly_found and current and current.stage == "collecting" and not current.name:
        stripped = text.strip()
        words = stripped.split()
        lowered_words = set(stripped.lower().split())
        is_question_or_inquiry = bool(re.search(r"[?]|^(what|how|why|who|where|when|tell me|explain|can you|do you|which|i want|i would|book|sign|is)\b", stripped, re.I))
        if not is_question_or_inquiry and 1 <= len(words) <= 3 and re.match(r"^[A-Za-z\s]+$", stripped):
            is_blacklisted = (
                is_pure_acknowledgement(stripped)
                or stripped.lower() in _NAME_BLACKLIST
                or bool(lowered_words & _NAME_BLACKLIST)
                or any(pat.search(stripped) for pat, _ in _SUBJECT_PATTERNS)
                or any(pat.search(stripped) for pat in _GRADE_PATTERNS)
                or any(pat.search(stripped) for pat in _TIME_PATTERNS)
            )
            if not is_blacklisted:
                lead.name = stripped.title()
                newly_found.append(f"name: {lead.name}")

    return lead, newly_found


def format_lead_summary(lead: DemoLead) -> str:
    """Format structured demo details for confirmation."""
    lines = [
        "- **Student / Parent Name**: " + (lead.name or "Not provided"),
        "- **Grade**: " + (lead.grade or "Not provided"),
        "- **Subject**: " + (lead.subject or "Not provided"),
        "- **Contact**: " + (lead.contact_value or "Not provided"),
        "- **Preferred Time**: " + (lead.preferred_time or "Flexible"),
    ]
    return "\n".join(lines)


def guide_to_demo_booking(message: str, user_name: Optional[str] = None, context_program: Optional[str] = None) -> str:
    """Return a natural, context-aware redirect to the Book Free Demo form.

    This is the primary response generator when DEMO_TRANSACTION_ENABLED is
    False.  It understands what the user said and responds helpfully without
    ever claiming to collect, store, or submit information.
    """
    lowered = message.strip().lower()

    # Detect if the user is providing personal info (name, grade, phone, etc.)
    has_name_intro = bool(re.search(r"\b(?:my name is|i am|i'm|this is|call me|name is)\b", lowered))
    has_phone = bool(_PHONE_RE.search(message))
    has_email = bool(_EMAIL_RE.search(message))
    has_grade = any(p.search(message) for p in _GRADE_PATTERNS)
    has_subject = any(p.search(message) for p, _ in _SUBJECT_PATTERNS)
    is_providing_info = has_name_intro or has_phone or has_email or has_grade or has_subject

    # Build a natural response
    if user_name and is_providing_info:
        return (
            f"Thanks, {user_name}. You can enter those details through the "
            f"**Book Free Demo** form at the top-right of the website."
        )
    if is_providing_info:
        return (
            "Got it. If that's for a demo, please enter those details in the "
            "**Book Free Demo** form at the top-right of the website."
        )
    if re.search(r"\b(book|register|sign\s*up|submit|arrange|schedule)\b.*\b(for me|for us|it|demo|trial)\b", lowered):
        return (
            "I can't submit the demo booking directly from the chat right now. "
            "You can use the **Book Free Demo** button at the top-right of the website to enter your details directly, "
            "or reach out to our team at **admin@wementors.co** or **+91 76111 92227**."
        )
    # General demo booking guidance
    return (
        "You can book a free demo in either of these ways:\n\n"
        "- Use the **Book Free Demo** option at the top-right of the website and enter your details directly.\n"
        "- Contact the WeMentors team directly through phone/WhatsApp at **+91 76111 92227** "
        "(alternate: +91 90398 03526, +91 88719 34995) or email at **admin@wementors.co** "
        "(Monday to Saturday, 9:00 AM to 8:00 PM IST)."
    )


def process_demo_flow(
    session_id: str,
    message: str,
    current_lead: Optional[DemoLead] = None,
    backend_submit_override: Optional[bool] = None,
) -> Tuple[str, str, DemoLead]:
    """
    State machine transition for demo class bookings.
    Returns: (reply_text, intent, updated_lead)

    IMPORTANT: When DEMO_TRANSACTION_ENABLED is False, this function
    immediately returns guidance to the website form and never enters
    the collecting/confirming/submitted workflow.
    """
    lead = DemoLead.from_dict(current_lead.to_dict()) if current_lead else DemoLead()

    # ── Capability gate ──────────────────────────────────────────────
    if not DEMO_TRANSACTION_ENABLED:
        # Cancellation when transaction is disabled
        if is_cancellation(message):
            database.clear_demo_lead(session_id)
            return (
                "No problem at all! Feel free to reach out to the WeMentors team anytime "
                "by email at admin@wementors.co or phone/WhatsApp at +91 76111 92227. "
                "What else can I help you with?",
                "demo_cancelled",
                DemoLead(),
            )
        # User asking if demo is booked
        if is_asking_if_booked(message):
            return (
                "The chatbot doesn't handle demo bookings directly. To submit a demo request, "
                "use the **Book Free Demo** form at the top-right of the website, or contact "
                "the team at **+91 76111 92227** / **admin@wementors.co**.",
                "demo_status",
                DemoLead(),
            )
        # All other messages: redirect to website form
        reply = guide_to_demo_booking(message)
        return reply, "demo_booking", DemoLead()

    # If user asks if demo is booked before it is submitted
    if is_asking_if_booked(message):
        if lead.stage == "submitted" and lead.submission_status == "confirmed":
            return (
                f"Yes, your demo enquiry for {lead.name or 'the student'} has already been submitted! "
                f"Our team will reach out via {lead.contact_value} to confirm the session.",
                "demo_status",
                lead,
            )
        missing = lead.get_missing_field_labels()
        missing_str = ", ".join(missing) if missing else "your final confirmation"
        return (
            f"Not yet — your demo request has not been submitted. We still need {missing_str} before our team can schedule it. "
            "Would you like to provide those details now?",
            "demo_status",
            lead,
        )

    # Cancellation
    if is_cancellation(message):
        lead.stage = "cancelled"
        database.clear_demo_lead(session_id)
        return (
            "No problem at all! We've cancelled this demo request. You can reach out to the WeMentors team anytime "
            "by email at admin@wementors.co or phone/WhatsApp at +91 76111 92227. What else can I help you with?",
            "demo_cancelled",
            lead,
        )

    # Stage: Confirming -> User says yes/confirm or changes a field
    if lead.stage == "confirming":
        if is_confirmation(message) or is_pure_acknowledgement(message):
            # Step 5: Backend submits information
            success = True if backend_submit_override is None else backend_submit_override
            if success:
                lead.stage = "submitted"
                lead.submission_status = "confirmed"
                database.save_demo_lead(
                    session_id,
                    name=lead.name,
                    grade=lead.grade,
                    subject=lead.subject,
                    contact_method=lead.contact_method,
                    contact_value=lead.contact_value,
                    preferred_time=lead.preferred_time,
                    stage=lead.stage,
                    submission_status=lead.submission_status,
                )
                # Step 7: Only now may the chatbot state that the request was submitted
                reply = (
                    f"Thank you, {lead.name}! Your demo class request has been submitted successfully.\n\n"
                    f"Our team will contact you at **{lead.contact_value}** to confirm the schedule for **{lead.grade} {lead.subject}**. "
                    "If you have any questions in the meantime, feel free to ask!"
                )
                return reply, "demo_submitted", lead
            else:
                lead.submission_status = "failed"
                database.save_demo_lead(
                    session_id,
                    name=lead.name,
                    grade=lead.grade,
                    subject=lead.subject,
                    contact_method=lead.contact_method,
                    contact_value=lead.contact_value,
                    preferred_time=lead.preferred_time,
                    stage="confirming",
                    submission_status="failed",
                )
                reply = (
                    "I ran into an issue submitting your request to our backend. Please try confirming again in a moment, "
                    "or you can contact the WeMentors team directly at admin@wementors.co or +91 76111 92227."
                )
                return reply, "demo_submission_failed", lead

        # User provides an updated detail while confirming
        updated_lead, newly_found = extract_lead_fields(message, lead)
        if newly_found:
            lead = updated_lead
            database.save_demo_lead(
                session_id,
                name=lead.name,
                grade=lead.grade,
                subject=lead.subject,
                contact_method=lead.contact_method,
                contact_value=lead.contact_value,
                preferred_time=lead.preferred_time,
                stage="confirming",
                submission_status="pending",
            )
            summary = format_lead_summary(lead)
            reply = (
                f"I've updated that for you. Here are your revised demo class details:\n\n{summary}\n\n"
                "Shall I submit this demo request for you?"
            )
            return reply, "demo_confirming", lead

        return (
            "Would you like me to submit these details for your free demo class, or would you like to change anything first?",
            "demo_confirming",
            lead,
        )

    # If in collecting stage and user only replies "ok", "sure", etc.
    if lead.stage == "collecting" and is_pure_acknowledgement(message):
        return (
            "Sure. You can enter your details through the **Book Free Demo** option at the top-right of the website, "
            "or share them here whenever you're ready (name and grade to get started). "
            "You can also contact the WeMentors team directly by phone/WhatsApp at **+91 76111 92227** or email at **admin@wementors.co**.",
            "demo_acknowledgement",
            lead,
        )

    # Extract fields from the message
    updated_lead, newly_found = extract_lead_fields(message, lead)
    lead = updated_lead

    # If entering demo flow for the first time
    if lead.stage == "idle":
        lead.stage = "collecting"

    # Save to database
    database.save_demo_lead(
        session_id,
        name=lead.name,
        grade=lead.grade,
        subject=lead.subject,
        contact_method=lead.contact_method,
        contact_value=lead.contact_value,
        preferred_time=lead.preferred_time,
        stage=lead.stage,
        submission_status=lead.submission_status,
    )

    # Check if complete
    if lead.is_complete():
        lead.stage = "confirming"
        database.save_demo_lead(
            session_id,
            name=lead.name,
            grade=lead.grade,
            subject=lead.subject,
            contact_method=lead.contact_method,
            contact_value=lead.contact_value,
            preferred_time=lead.preferred_time,
            stage=lead.stage,
            submission_status="pending",
        )
        summary = format_lead_summary(lead)
        reply = (
            f"Here are the details for your free demo class request:\n\n{summary}\n\n"
            "Would you like me to submit this request?"
        )
        return reply, "demo_confirming", lead

    # Incomplete: Acknowledge what was provided and ask for missing fields
    missing_labels = lead.get_missing_field_labels()
    bullets = "\n".join(f"- {label}" for label in missing_labels)

    if newly_found:
        found_desc = ", ".join(newly_found)
        reply = (
            f"Thanks! I noted your {found_desc}. To finish setting up your free demo, could you also provide:\n\n{bullets}\n\n"
            "You can share these whenever you're ready."
        )
    else:
        reply = (
            "I'd be glad to help arrange a free demo class for you! To set this up, could you please share:\n\n"
            "- Student or parent name\n"
            "- Student's grade (Grades 3–10)\n"
            "- Preferred subject (Maths, Science, or Confident Speaker)\n"
            "- Mobile number or email address\n"
            "- Preferred day or time slot\n\n"
            "You can share these all at once or one at a time."
        )

    return reply, "demo_collecting", lead
