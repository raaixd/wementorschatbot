"""
Test suite for conversational cancellation and conservative, context-aware name detection.
Covers:
1. Section 9 Intent Classification Matrix (25 items + 6 combinations).
2. Pure cancellation phrases (nevermind, never mind, nvm, forget it, leave it, actually never mind, etc.).
3. Contextual cancellation (cancellation after Foundation Years, Middle School, Confident Speaker, demo info).
4. Explicit questions overriding cancellation.
5. Conservative name handling (standalone vs explicit disclosure vs name prompt context).
6. Conversational discourse, questions, and reactions.
7. Section 12 exact multi-turn live conversation test.
"""
import os
import sys
import tempfile
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(), "test_cancellation.db"))

from app import database, personality
from app.knowledge import load_entries
from app.conversation import ConversationEngine

database.init_db()
engine = ConversationEngine(load_entries())

failures = []

def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)

def send(session_id: str, text: str):
    database.log_message(session_id, "user", text)
    res = engine.handle_message(session_id, text)
    database.log_message(
        session_id,
        "assistant",
        res.reply,
        intent=res.intent,
        matched_entry_ids=",".join(res.matched_entry_ids) if res.matched_entry_ids else None,
        confidence=res.confidence,
    )
    return res

print("================================================================================")
print("TEST SUITE: CONVERSATIONAL CANCELLATION & CONTEXT-AWARE NAME DETECTION")
print("================================================================================")

# --- 1. Section 9 Classification Matrix ---
print("\n--- 1. Section 9 Classification Matrix ---")
matrix_cases = [
    ("My name is Raaid", "explicit_name"),
    ("I'm Raaid", "explicit_name"),
    ("I am Raaid", "explicit_name"),
    ("Call me Raaid", "explicit_name"),
    ("Raaid", "context-dependent"),
    ("nevermind", "cancellation"),
    ("never mind", "cancellation"),
    ("nvm", "cancellation"),
    ("forget it", "cancellation"),
    ("leave it", "cancellation"),
    ("okay", "acknowledgement"),
    ("okay thanks", "acknowledgement"),
    ("sure", "affirmation"),
    ("got it", "acknowledgement"),
    ("makes sense", "acknowledgement"),
    ("interesting", "conversational_reaction"),
    ("hmm", "conversational_reaction"),
    ("actually", "discourse_marker"),
    ("Maths", "subject/topic"),
    ("Science", "subject/topic"),
    ("Middle School", "program/topic"),
    ("Foundation Years", "program/topic"),
    ("what subjects are offered?", "information_request"),
    ("why?", "question"),
    ("really?", "conversational_question"),
]

for q, expected in matrix_cases:
    cat = engine.classify_conversation_intent(q)
    check(f"Matrix: '{q}' -> {expected}", cat == expected, f"got {cat}")

# --- 2. Discourse / Cancellation Prefix Overridden by Explicit Question ---
print("\n--- 2. Discourse & Cancellation Prefix Overrides ---")
comb_cases = [
    ("nevermind, tell me about Middle School", "middle_school_overview"),
    ("forget it, what subjects do you offer?", "faq"),
    ("actually never mind", "cancellation"),
    ("okay, what about Grade 7?", "middle_school_grades"),
    ("Raaid, what programs do you offer?", "faq"),
    ("My name is Raaid, tell me about Grade 7", "middle_school_grades"),
]

for q, expected in comb_cases:
    det = engine.detect_intent(q)
    check(f"Override: '{q}' -> {expected}", det == expected, f"got {det}")

# --- 3. Pure Cancellation Signals (Must Never Be Names) ---
print("\n--- 3. Pure Cancellation Signals ---")
cancel_phrases = [
    "nevermind",
    "never mind",
    "nvm",
    "forget it",
    "forget that",
    "actually never mind",
    "no worries",
    "that's okay",
    "leave it",
    "it's fine",
    "don't worry about it",
    "I changed my mind",
    "doesn't matter",
    "ignore that",
    "skip that",
    "let's forget it",
]

for phrase in cancel_phrases:
    sid = f"cancel-{uuid.uuid4()}"
    database.ensure_session(sid)
    res = send(sid, phrase)
    check(f"Phrase '{phrase}' intent is cancellation", res.intent == "cancellation", f"intent={res.intent}")
    check(f"Phrase '{phrase}' does NOT treat word as name", "nice to meet you" not in res.reply.lower(), res.reply)
    check(f"Phrase '{phrase}' has natural cancellation reply", "no problem" in res.reply.lower() or "no worries" in res.reply.lower() or "sure" in res.reply.lower(), res.reply)

# --- 4. Contextual Cancellation after Programs ---
print("\n--- 4. Contextual Cancellation after Programs ---")
# 4a. After Foundation Years
sid_fy = f"test-ctx-fy-{uuid.uuid4()}"
send(sid_fy, "Tell me about Foundation Years")
r_fy_cancel = send(sid_fy, "nevermind")
check("Cancellation after Foundation Years intent is cancellation", r_fy_cancel.intent == "cancellation", r_fy_cancel.intent)
check("Cancellation after Foundation Years has natural acknowledgement", "no problem" in r_fy_cancel.reply.lower() or "feel free" in r_fy_cancel.reply.lower(), r_fy_cancel.reply)
check("Cancellation after Foundation Years does NOT repeat FY overview", "grades 3–5" not in r_fy_cancel.reply.lower() and "grades 3-5" not in r_fy_cancel.reply.lower(), r_fy_cancel.reply)
# Subsequent query works normally
r_fy_next = send(sid_fy, "What about Middle School?")
check("Subsequent query after cancellation resolves to Middle School", "middle" in r_fy_next.reply.lower() or r_fy_next.intent == "middle_school_overview", r_fy_next.reply)

# 4b. After Middle School
sid_ms = f"test-ctx-ms-{uuid.uuid4()}"
send(sid_ms, "Tell me about Middle School")
r_ms_cancel = send(sid_ms, "never mind")
check("Cancellation after Middle School intent is cancellation", r_ms_cancel.intent == "cancellation", r_ms_cancel.intent)
check("Cancellation after Middle School does NOT name user 'Never Mind'", "never mind" not in r_ms_cancel.reply.lower(), r_ms_cancel.reply)

# 4c. After Confident Speaker
sid_cs = f"test-ctx-cs-{uuid.uuid4()}"
send(sid_cs, "Tell me about Confident Speaker")
r_cs_cancel = send(sid_cs, "forget it")
check("Cancellation after Confident Speaker intent is cancellation", r_cs_cancel.intent == "cancellation", r_cs_cancel.intent)

# 4d. After Demo Information
sid_dm = f"test-ctx-dm-{uuid.uuid4()}"
send(sid_dm, "How do I book a demo?")
r_dm_cancel = send(sid_dm, "leave it")
check("Cancellation after Demo Info intent is cancellation", r_dm_cancel.intent == "cancellation", r_dm_cancel.intent)
check("Cancellation after Demo does NOT create lead", database.get_demo_lead(sid_dm) is None, str(database.get_demo_lead(sid_dm)))

# --- 5. Questions Overriding Cancellation ---
print("\n--- 5. Questions Overriding Cancellation ---")
sid_ov1 = f"test-ov1-{uuid.uuid4()}"
r_ov1 = send(sid_ov1, "nevermind, tell me about Middle School")
check("nevermind + Middle School answers Middle School", "middle school" in r_ov1.reply.lower() and "6–8" in r_ov1.reply, r_ov1.reply)
check("nevermind + Middle School intent is middle_school_overview", r_ov1.intent == "middle_school_overview", r_ov1.intent)

sid_ov2 = f"test-ov2-{uuid.uuid4()}"
r_ov2 = send(sid_ov2, "never mind, what subjects are offered?")
check("never mind + subjects answers subjects", "mathematics" in r_ov2.reply.lower() or "subjects" in r_ov2.reply.lower(), r_ov2.reply)

sid_ov3 = f"test-ov3-{uuid.uuid4()}"
r_ov3 = send(sid_ov3, "forget it, how do I book a demo?")
check("forget it + book demo gives demo instructions", "book free demo" in r_ov3.reply.lower() or "website" in r_ov3.reply.lower(), r_ov3.reply)

# --- 6. Name Detection Precision & Conservative Standalone Handling ---
print("\n--- 6. Conservative Name Detection ---")
# 6a. Standalone 'Raaid' without conversational context
sid_name_alone = f"test-name-alone-{uuid.uuid4()}"
r_alone = send(sid_name_alone, "Raaid")
check("Standalone 'Raaid' without context does NOT assume name", "nice to meet you, raaid" not in r_alone.reply.lower(), r_alone.reply)
check("Standalone 'Raaid' without context asks clarifying question", "clarify" in r_alone.intent or "how can i help" in r_alone.reply.lower(), r_alone.reply)

# 6b. Explicit name disclosure: "My name is Raaid"
sid_explicit = f"test-name-exp-{uuid.uuid4()}"
r_exp = send(sid_explicit, "My name is Raaid")
check("Explicit 'My name is Raaid' acknowledges Raaid", "raaid" in r_exp.reply.lower() and "nice to meet you" in r_exp.reply.lower(), r_exp.reply)
check("Explicit 'My name is Raaid' intent is explicit_name", r_exp.intent == "explicit_name", r_exp.intent)

# 6c. Standalone name WHEN asked for name
sid_asked = f"test-asked-name-{uuid.uuid4()}"
database.log_message(sid_asked, "assistant", "May I know your name so I can address you properly?")
r_asked = send(sid_asked, "Raaid")
check("Standalone name when asked for name recognizes Raaid", "raaid" in r_asked.reply.lower() and "nice to meet you" in r_asked.reply.lower(), r_asked.reply)

# 6d. Explicit intro with question: "My name is Raaid, tell me about Grade 7"
sid_intro_q = f"test-intro-q-{uuid.uuid4()}"
r_intro_q = send(sid_intro_q, "My name is Raaid, tell me about Grade 7")
check("Intro + Grade 7 answers Grade 7 and addresses Raaid", "raaid" in r_intro_q.reply.lower() and "middle school" in r_intro_q.reply.lower(), r_intro_q.reply)

# --- 7. Topic / Program / Subject Preservation ---
print("\n--- 7. Topic & Program Preservation ---")
sid_maths = f"test-maths-{uuid.uuid4()}"
r_maths = send(sid_maths, "Maths")
check("Standalone 'Maths' resolves to mathematics mentoring overview", "mathematics" in r_maths.reply.lower() or "math" in r_maths.reply.lower(), r_maths.reply)
check("Standalone 'Maths' intent is subject_maths", r_maths.intent == "subject_maths", r_maths.intent)

sid_science = f"test-science-{uuid.uuid4()}"
r_science = send(sid_science, "Science")
check("Standalone 'Science' resolves to science overview", "science" in r_science.reply.lower(), r_science.reply)

sid_ms_standalone = f"test-ms-prog-{uuid.uuid4()}"
r_ms_standalone = send(sid_ms_standalone, "Middle School")
check("Standalone 'Middle School' resolves to middle school overview", r_ms_standalone.intent == "middle_school_overview", r_ms_standalone.intent)

# --- 8. Section 12 Exact Live Test Sequence ---
print("\n--- 8. Section 12 Exact Live Conversation Sequence ---")
sid_live = f"test-live-seq-{uuid.uuid4()}"

# Turn 1: actually, tell me about foundation years
t1 = send(sid_live, "actually, tell me about foundation years")
check("Turn 1: Foundation Years overview returned", "foundation years" in t1.reply.lower() and ("grades 3–5" in t1.reply.lower() or "grades 3-5" in t1.reply.lower() or "3–5" in t1.reply or "3-5" in t1.reply), t1.reply)
check("Turn 1: Not classified as name", "nice to meet you" not in t1.reply.lower(), t1.reply)

# Turn 2: nevermind
t2 = send(sid_live, "nevermind")
check("Turn 2: Natural cancellation response", "no problem" in t2.reply.lower() or "no worries" in t2.reply.lower() or "sure" in t2.reply.lower(), t2.reply)
check("Turn 2: MUST NOT say 'Nice to meet you, Nevermind!'", "nice to meet you, nevermind" not in t2.reply.lower() and "nevermind" not in t2.reply.lower(), t2.reply)
check("Turn 2: Intent is cancellation", t2.intent == "cancellation", t2.intent)

# Turn 3: Raaid (should NOT automatically assume this is a name without context)
t3 = send(sid_live, "Raaid")
check("Turn 3: Does NOT automatically assume Raaid is name", "nice to meet you, raaid" not in t3.reply.lower(), t3.reply)
check("Turn 3: Asks clarification conservatively", "clarify" in t3.intent or "how can i help" in t3.reply.lower(), t3.reply)

# Turn 4: actually, what about Middle School?
t4 = send(sid_live, "actually, what about Middle School?")
check("Turn 4: Middle School answer returned", "middle school" in t4.reply.lower() and ("grades 6–8" in t4.reply.lower() or "grades 6-8" in t4.reply.lower() or "6–8" in t4.reply or "6-8" in t4.reply), t4.reply)

# Turn 5: never mind, what subjects are offered?
t5 = send(sid_live, "never mind, what subjects are offered?")
check("Turn 5: Answers subjects question", "mathematics" in t5.reply.lower() or "subjects" in t5.reply.lower(), t5.reply)
check("Turn 5: Does NOT treat 'never mind' as name", "never mind" not in t5.reply.lower(), t5.reply)

# Turn 6: My name is Raaid
t6 = send(sid_live, "My name is Raaid")
check("Turn 6: Name can now be confidently recognized", "raaid" in t6.reply.lower() and "nice to meet you" in t6.reply.lower(), t6.reply)
check("Turn 6: Intent is explicit_name", t6.intent == "explicit_name", t6.intent)

# Turn 7: nvm
t7 = send(sid_live, "nvm")
check("Turn 7: Treats 'nvm' as cancellation, never as name", t7.intent == "cancellation", t7.intent)
check("Turn 7: MUST NOT say 'Nice to meet you, Nvm!'", "nvm" not in t7.reply.lower() and "nice to meet you" not in t7.reply.lower(), t7.reply)

print("\n" + "=" * 80)
if failures:
    print(f"FAILED {len(failures)} CHECKS:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED PERFECTLY (100% SUCCESS)!")
    print("=" * 80)
