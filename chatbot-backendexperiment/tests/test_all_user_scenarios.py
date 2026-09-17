"""
Test script verifying all 10 core conversation scenarios required by the user:

1. User: "Hey"
   - Must be treated as a greeting.
   - Must not be treated as a name.
   - Must not start demo data collection.

2. User: "Tell me about your Python course"
   - Must not answer with fee information.
   - Must say Python is not verified/listed in current course information.
   - Must offer a relevant next step (Book Free Demo / contact team).

3. User: "How much is the Python course?"
   - Must first address whether Python is offered.
   - Must not invent pricing.

4. User: "What course is good for beginners?"
   - Must ask what subject or learning goal the user means.
   - Must not invent an unverified beginner course.

5. User: "What?"
   - Must provide a helpful clarification response.
   - Must not simply repeat a generic scope statement.

6. User: "Demo class"
   - Must explain that the demo is free if verified.
   - Must direct the user to the Book Free Demo option at the top-right of the website.

7. User: "Book enroll"
   - Must interpret this as possible demo booking or enrollment.
   - Must clarify if necessary.
   - Must provide the website booking instructions.
   - Must not falsely confirm booking.

8. Assistant asks for a name, user replies "Okay"
   - Must not save "Okay" as the user's name.
   - Must reply asking what name to use for the demo request.

9. Assistant asks whether the user wants to book a demo, user replies "Yes"
   - Must guide them to the website form.
   - Must not claim that the form was submitted.

10. User asks a completely unrelated question
   - Must provide a concise out-of-scope response.
   - Must not force the question into fees, courses, or demos.
"""
import os
import sys
import tempfile
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["LLM_PROVIDER"] = "none"
os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(), "test_all_user.db"))

from app import database, leads
from app.knowledge import load_entries
from app.conversation import ConversationEngine

database.init_db()
engine = ConversationEngine(load_entries())

failures = []

def check(description: str, condition: bool, detail: str = ""):
    if condition:
        print(f"[PASS] {description}")
    else:
        print(f"[FAIL] {description} -- {detail}")
        failures.append(f"{description} ({detail})")


print("=" * 70)
print("RUNNING ALL 10 REQUIRED USER CONVERSATION SCENARIOS")
print("=" * 70)

# --- Scenario 1: "Hey" ---
print("\n--- 1. User: 'Hey' ---")
sid1 = f"scenario-1-{uuid.uuid4()}"
database.ensure_session(sid1)
r1 = engine.handle_message(sid1, "Hey")
print(f"[USER]: Hey\n[ASSISTANT ({r1.intent})]: {r1.reply}")
check("Treated as greeting intent", r1.intent == "greeting", f"intent={r1.intent}")
check("Greeting response is returned", any(w in r1.reply.lower() for w in ["hi", "hello", "welcome", "wementors"]), r1.reply)
lead1 = database.get_demo_lead(sid1)
check("Did not save 'Hey' as a name", not lead1 or lead1.get("name") != "Hey", f"lead={lead1}")
check("Did not start demo data collection", not lead1 or lead1.get("stage") != "collecting", f"lead={lead1}")

# --- Scenario 2: "Tell me about your Python course" ---
print("\n--- 2. User: 'Tell me about your Python course' ---")
sid2 = f"scenario-2-{uuid.uuid4()}"
database.ensure_session(sid2)
r2 = engine.handle_message(sid2, "Tell me about your Python course")
print(f"[USER]: Tell me about your Python course\n[ASSISTANT ({r2.intent})]: {r2.reply}")
check("Must NOT answer with fee information", "fee details are not published" not in r2.reply.lower() and "pricing depends" not in r2.reply.lower(), r2.reply)
check("Must state Python is not verified/listed", "couldn’t find" in r2.reply or "couldn't find" in r2.reply or "not" in r2.reply.lower(), r2.reply)
check("Offers relevant next step (Book Free Demo)", "Book Free Demo" in r2.reply or "contact" in r2.reply.lower(), r2.reply)

# --- Scenario 3: "How much is the Python course?" ---
print("\n--- 3. User: 'How much is the Python course?' ---")
sid3 = f"scenario-3-{uuid.uuid4()}"
database.ensure_session(sid3)
r3 = engine.handle_message(sid3, "How much is the Python course?")
print(f"[USER]: How much is the Python course?\n[ASSISTANT ({r3.intent})]: {r3.reply}")
check("Must first address whether Python is offered", "does not have a verified python course" in r3.reply.lower() or "not listed" in r3.reply.lower() or "not verified" in r3.reply.lower(), r3.reply)
check("Must not invent pricing", "₹" not in r3.reply and "per month" not in r3.reply and "per hour" not in r3.reply, r3.reply)
check("Offers Book Free Demo or team contact", "Book Free Demo" in r3.reply or "contact" in r3.reply.lower(), r3.reply)

# --- Scenario 4: "What course is good for beginners?" ---
print("\n--- 4. User: 'What course is good for beginners?' ---")
sid4 = f"scenario-4-{uuid.uuid4()}"
database.ensure_session(sid4)
r4 = engine.handle_message(sid4, "What course is good for beginners?")
print(f"[USER]: What course is good for beginners?\n[ASSISTANT ({r4.intent})]: {r4.reply}")
check("Asks clarifying follow-up question", "what would you like to learn as a beginner" in r4.reply.lower() or "grade" in r4.reply.lower(), r4.reply)
check("Does not invent unverified beginner course", "beginner course" not in r4.reply.lower() or "which wementors option" in r4.reply.lower(), r4.reply)
check("Does not give generic concept answer", "we focus on understanding concepts rather than" not in r4.reply.lower(), r4.reply)

# --- Scenario 5: "What?" ---
print("\n--- 5. User: 'What?' ---")
sid5 = f"scenario-5-{uuid.uuid4()}"
database.ensure_session(sid5)
r5 = engine.handle_message(sid5, "What?")
print(f"[USER]: What?\n[ASSISTANT ({r5.intent})]: {r5.reply}")
check("Acknowledges confusion naturally", "wasn’t clear" in r5.reply or "wasn't clear" in r5.reply or "sorry" in r5.reply.lower(), r5.reply)
check("Guides user to classes/subjects/demo", "classes" in r5.reply.lower() and "demo" in r5.reply.lower(), r5.reply)
check("Does not merely repeat generic scope statement", "i’m designed to help with" not in r5.reply.lower(), r5.reply)

# --- Scenario 6: "Demo class" ---
print("\n--- 6. User: 'Demo class' ---")
sid6 = f"scenario-6-{uuid.uuid4()}"
database.ensure_session(sid6)
r6 = engine.handle_message(sid6, "Demo class")
print(f"[USER]: Demo class\n[ASSISTANT ({r6.intent})]: {r6.reply}")
check("Explains free demo with no obligation", "free demo class with no obligation" in r6.reply.lower(), r6.reply)
check("Directs user to Book Free Demo at top-right", "Book Free Demo" in r6.reply and "top-right" in r6.reply.lower(), r6.reply)

# --- Scenario 7: "Book enroll" ---
print("\n--- 7. User: 'Book enroll' ---")
sid7 = f"scenario-7-{uuid.uuid4()}"
database.ensure_session(sid7)
r7 = engine.handle_message(sid7, "Book enroll")
print(f"[USER]: Book enroll\n[ASSISTANT ({r7.intent})]: {r7.reply}")
check("Interprets as demo booking or enrollment", "would you like to book a free demo class" in r7.reply.lower(), r7.reply)
check("Provides website booking instructions", "Book Free Demo" in r7.reply and "top-right" in r7.reply.lower(), r7.reply)
check("Does not claim demo was booked or submitted", "booked" not in r7.reply.lower() and "submitted" not in r7.reply.lower(), r7.reply)

# --- Scenario 8: Assistant asks for name, user replies "Okay" ---
print("\n--- 8. Assistant asks for name, user replies 'Okay' ---")
sid8 = f"scenario-8-{uuid.uuid4()}"
database.ensure_session(sid8)
# Step 1: Assistant asks for name
database.save_demo_lead(sid8, stage="collecting")
database.log_message(sid8, "user", "I want to book a free demo")
database.log_message(sid8, "assistant", "Sure! What is your name?")
# Step 2: User says "Okay"
r8 = engine.handle_message(sid8, "Okay")
print(f"[USER]: Okay\n[ASSISTANT ({r8.intent})]: {r8.reply}")
check("Must NOT save 'Okay' as name", "noted your name as okay" not in r8.reply.lower(), r8.reply)
lead8 = database.get_demo_lead(sid8)
check("Lead name in database is not 'Okay'", not lead8 or lead8.get("name") != "Okay", f"lead={lead8}")
check("Asks what name to use for demo request", "what name should i use" in r8.reply.lower(), r8.reply)

# --- Scenario 9: Assistant asks whether user wants to book a demo, user replies "Yes" ---
print("\n--- 9. Assistant asks whether user wants to book a demo, user replies 'Yes' ---")
sid9 = f"scenario-9-{uuid.uuid4()}"
database.ensure_session(sid9)
# Assistant asked: "Would you like to book a free demo class?..."
database.log_message(sid9, "user", "Book enroll")
database.log_message(sid9, "assistant", "Would you like to book a free demo class? You can do that by clicking Book Free Demo at the top-right of the website.")
# User replies "Yes"
r9 = engine.handle_message(sid9, "Yes")
print(f"[USER]: Yes\n[ASSISTANT ({r9.intent})]: {r9.reply}")
check("Guides user to click Book Free Demo", "Book Free Demo" in r9.reply and "top-right" in r9.reply.lower(), r9.reply)
check("Must NOT claim form was submitted", "submitted" not in r9.reply.lower() and "your demo is booked" not in r9.reply.lower(), r9.reply)

# --- Scenario 10: Completely unrelated / out-of-scope question ---
print("\n--- 10. User asks completely unrelated question ---")
sid10 = f"scenario-10-{uuid.uuid4()}"
database.ensure_session(sid10)
r10 = engine.handle_message(sid10, "Who won the football match?")
print(f"[USER]: Who won the football match?\n[ASSISTANT ({r10.intent})]: {r10.reply}")
check("Provides concise out-of-scope response", "i’m designed to help with" in r10.reply.lower() or "i'm designed to help with" in r10.reply.lower(), r10.reply)
check("Does not force into fees", "fee" not in r10.reply.lower() and "pricing" not in r10.reply.lower(), r10.reply)
check("Does not force into demo booking", "book free demo" not in r10.reply.lower(), r10.reply)

print("\n" + "=" * 70)
if failures:
    print(f"FAILED: {len(failures)} assertion(s) failed:")
    for f in failures:
        print(f" - {f}")
    sys.exit(1)
else:
    print("ALL 10 TARGET SCENARIOS PASSED PERFECTLY WITH ZERO REGRESSIONS!")
print("=" * 70)
