"""
Comprehensive evaluation and regression test suite for:
- Conversational UX & personality
- Field-level intent recognition
- Confident Speaker format repetition bug fix (Tests A, B, C, D, E)
- Progressive disclosure and anaphora resolution
- Safety, unverified details, and out-of-scope handling
"""
import os
import sys
import tempfile
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(), "test_ux.db"))

from app import config, database, personality
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

print("\n=== Regression Tests: Confident Speaker Format Loop & Field Intents ===\n")

# Test A: Direct format question
sess_a = f"test-a-{uuid.uuid4().hex[:6]}"
res_a = send(sess_a, "What's the format for Confident Speaker?")
check("Test A: Intent is confident_speaker_format", res_a.intent == "confident_speaker_format", res_a.intent)
check("Test A: Answers format directly", "personalized one-to-one mentoring" in res_a.reply.lower(), res_a.reply)
check("Test A: Mentions guided speaking practice & feedback", "guided speaking practice" in res_a.reply.lower() and "feedback" in res_a.reply.lower(), res_a.reply)
check("Test A: Mentions scope (Spoken English, Public Speaking, Interview Skills)", "spoken english" in res_a.reply.lower() and "public speaking" in res_a.reply.lower() and "interview skills" in res_a.reply.lower(), res_a.reply)
check("Test A: Gives Book Free Demo next step", "book free demo" in res_a.reply.lower(), res_a.reply)
check("Test A: Does NOT tell user to ask about format", "ask about the program format" not in res_a.reply.lower(), res_a.reply)

# Test B: Contextual follow-up
sess_b = f"test-b-{uuid.uuid4().hex[:6]}"
res_b1 = send(sess_b, "Tell me about Confident Speaker.")
check("Test B1: Matches confident_speaker", "confident" in res_b1.reply.lower() or res_b1.intent == "confident_speaker", res_b1.reply)
res_b2 = send(sess_b, "What's the format?")
check("Test B2: Intent is confident_speaker_format", res_b2.intent == "confident_speaker_format", res_b2.intent)
check("Test B2: Answers Confident Speaker format directly", "personalized one-to-one mentoring" in res_b2.reply.lower(), res_b2.reply)
check("Test B2: Does NOT return generic capability menu", "i can help you explore we mentors" not in res_b2.reply.lower() and "what would you like to explore" not in res_b2.reply.lower(), res_b2.reply)

# Test C: Repeated short phrase
sess_c = f"test-c-{uuid.uuid4().hex[:6]}"
res_c1 = send(sess_c, "What's the format for Confident Speaker?")
res_c2 = send(sess_c, "Program format.")
check("Test C: Intent is confident_speaker_format", res_c2.intent == "confident_speaker_format", res_c2.intent)
check("Test C: Returns concise clarification", "in short" in res_c2.reply.lower() and len(res_c2.reply) < len(res_c1.reply), res_c2.reply)
check("Test C: Does not say 'You can ask about the program format'", "ask about the program format" not in res_c2.reply.lower(), res_c2.reply)

# Test D: Scope question
sess_d = f"test-d-{uuid.uuid4().hex[:6]}"
res_d = send(sess_d, "What does Confident Speaker cover?")
check("Test D: Intent is confident_speaker_scope", res_d.intent == "confident_speaker_scope", res_d.intent)
check("Test D: Mentions Spoken English, Public Speaking, Interview Skills", "spoken english" in res_d.reply.lower() and "public speaking" in res_d.reply.lower() and "interview skills" in res_d.reply.lower(), res_d.reply)
check("Test D: Practical development rather than rote grammar", "rote grammar" in res_d.reply.lower() or "practical development" in res_d.reply.lower(), res_d.reply)

# Test E: Session activity question
sess_e = f"test-e-{uuid.uuid4().hex[:6]}"
res_e = send(sess_e, "What happens during the sessions?")
# In context of Confident Speaker
sess_e_cs = f"test-e-cs-{uuid.uuid4().hex[:6]}"
send(sess_e_cs, "Tell me about Confident Speaker")
res_e_cs = send(sess_e_cs, "What happens during the sessions?")
check("Test E: Intent is confident_speaker_activities", res_e_cs.intent == "confident_speaker_activities", res_e_cs.intent)
check("Test E: Explains guided practice & practical conversation", "guided speaking practice" in res_e_cs.reply.lower() and "practical conversation" in res_e_cs.reply.lower(), res_e_cs.reply)
check("Test E: No invented class size or duration", "45 minutes" not in res_e_cs.reply.lower() and "batch size" not in res_e_cs.reply.lower(), res_e_cs.reply)

print("\n=== Testing General Inquiries Matrix ===\n")

for q in [
    "What is WeMentors?",
    "Tell me about your academy.",
    "What do you guys do?",
    "I want to know more about WeMentors.",
    "Can you explain your platform?",
    "What is this website about?",
]:
    sess = f"test-gen-{uuid.uuid4().hex[:6]}"
    r = send(sess, q)
    check(f"General: '{q}' matches general_info", r.intent == "general_info", f"intent={r.intent}")
    check(f"General: '{q}' direct answer without call-center pleasantries", "great question" not in r.reply.lower() and "individual mentor attention" in r.reply.lower(), r.reply)
    check(f"General: '{q}' mentions Book Free Demo", "book free demo" in r.reply.lower(), r.reply)

print("\n=== Testing Beginner and Course Matrix ===\n")

# Beginner
sess_beg = f"test-beg-{uuid.uuid4().hex[:6]}"
r_beg = send(sess_beg, "What course is good for beginners?")
check("Beginner: intent is beginner_recommendation", r_beg.intent == "beginner_recommendation", r_beg.intent)
check("Beginner: direct answer asks if for school student or self", "asking for yourself or for a school student" in r_beg.reply.lower(), r_beg.reply)
check("Beginner: no disclaimers", "teaching methodologies vary" not in r_beg.reply.lower())

# Python course
sess_py = f"test-py-{uuid.uuid4().hex[:6]}"
r_py = send(sess_py, "Tell me about your Python course.")
check("Python: intent is unverified_course", r_py.intent == "unverified_course", r_py.intent)
check("Python: does not invent curriculum or fees", "confirmed details" in r_py.reply.lower(), r_py.reply)

# Programs overview
sess_prog = f"test-prog-{uuid.uuid4().hex[:6]}"
r_prog = send(sess_prog, "What programs do you offer?")
check("Programs: lists Foundation, Middle, Senior, Confident Speaker", "foundation" in r_prog.reply.lower() and "middle" in r_prog.reply.lower() and "senior" in r_prog.reply.lower() and "confident" in r_prog.reply.lower(), r_prog.reply)

print("\n=== Testing Grades 9–10 Matrix ===\n")

sess_g = f"test-g-{uuid.uuid4().hex[:6]}"
r_g1 = send(sess_g, "Do you offer board exam support?")
check("Grades 9-10: matches board_exam", r_g1.intent == "board_exam", r_g1.intent)
check("Grades 9-10: mentions personal mentor & concept clarity", "personal mentor" in r_g1.reply.lower(), r_g1.reply)

r_g2 = send(sess_g, "Is the teaching one-on-one?")
check("Grades 9-10 follow-up: is it one-on-one", "individual attention from a personal mentor" in r_g2.reply.lower(), r_g2.reply)

r_g3 = send(sess_g, "Do parents receive progress updates?")
check("Grades 9-10 follow-up: parent updates", "tracked and shared with parents every week" in r_g3.reply.lower(), r_g3.reply)

r_g4 = send(sess_g, "Can you guarantee marks?")
check("Grades 9-10 follow-up: guarantee marks", "does not guarantee marks" in r_g4.reply.lower(), r_g4.reply)

print("\n=== Testing Demo & Enrollment Matrix ===\n")

for demo_q in ["I want a demo.", "Book a demo.", "How do I enroll?", "I want to join.", "Can you book it for me?"]:
    sess = f"test-demo-{uuid.uuid4().hex[:6]}"
    r = send(sess, demo_q)
    check(f"Demo query '{demo_q}' points to Book Free Demo", "book free demo" in r.reply.lower(), r.reply)
    check(f"Demo query '{demo_q}' does not claim false booking", "your demo has been booked" not in r.reply.lower())

# Isolated confirmation ("Okay", "Yes")
sess_ok = f"test-ok-{uuid.uuid4().hex[:6]}"
r_ok = send(sess_ok, "Okay")
check("Isolated 'Okay': does not falsely confirm booking", "your demo is booked" not in r_ok.reply.lower() and "what would you like to explore next" in r_ok.reply.lower(), r_ok.reply)

print("\n=== Testing Conversational Behavior & Vague Inputs ===\n")

for vague_q in ["What?", "Information", "Help"]:
    sess = f"test-vague-{uuid.uuid4().hex[:6]}"
    r = send(sess, vague_q)
    check(f"Vague '{vague_q}': provides concise capabilities menu", "programs" in r.reply.lower() and "confident speaker" in r.reply.lower(), r.reply)
    check(f"Vague '{vague_q}': no long disclaimers", "verified records" not in r.reply.lower() and "as an ai" not in r.reply.lower())

# Thanks
sess_thx = f"test-thx-{uuid.uuid4().hex[:6]}"
r_thx = send(sess_thx, "Thanks")
check("Thanks: intent is thanks", r_thx.intent == "thanks", r_thx.intent)
check("Thanks: closes naturally without unprompted questions", "would you like to know" not in r_thx.reply.lower(), r_thx.reply)

print("\n=== Testing Out-of-Scope Behavior ===\n")

for oos_q in [
    "Write my assignment.",
    "What is the weather?",
    "Who is the president?",
    "Tell me a joke.",
    "Give me medical advice.",
]:
    sess = f"test-oos-{uuid.uuid4().hex[:6]}"
    r = send(sess, oos_q)
    check(f"Out of scope '{oos_q}': intent is off_topic", r.intent == "off_topic", r.intent)
    check(f"Out of scope '{oos_q}': redirects politely without hallucinating", "focused on helping with wementors" in r.reply.lower(), r.reply)

print("\n==================================================")
if failures:
    print(f"FAILED {len(failures)} CHECKS:")
    for f in failures:
        print(f" - {f}")
    sys.exit(1)
else:
    print("ALL CONVERSATIONAL UX & REGRESSION CHECKS PASSED!")
    print("==================================================\n")
    sys.exit(0)
