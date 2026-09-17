"""
Comprehensive test suite verifying natural conversational behavior,
ensuring the assistant answers directly and stops without automatically
appending unwanted follow-up suggestions or questions.
"""
import os
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SKIP_DOTENV"] = "1"
os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(), "test.db"))

from app import database
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

print("\n--- Testing 12 Required Conversational Behavior Scenarios ---\n")

# 1. "How are classes conducted?"
res1 = engine.handle_message("test-session-1", "How are classes conducted?")
check("1. Classes conducted: No 'Would you like to know'", "Would you like to know" not in res1.reply, res1.reply)
check("1. Classes conducted: Answers video calls & digital whiteboard", "video call" in res1.reply.lower() or "whiteboard" in res1.reply.lower(), res1.reply)
check("1. Classes conducted: Stops directly without asking another question", not res1.reply.strip().endswith("?"), res1.reply)

# 2. "Do you offer online classes?"
res2 = engine.handle_message("test-session-2", "Do you offer online classes?")
check("2. Online classes: No 'Would you like to know'", "Would you like to know" not in res2.reply, res2.reply)
check("2. Online classes: Answers online classes clearly", "online" in res2.reply.lower(), res2.reply)
check("2. Online classes: Stops directly without asking another question", not res2.reply.strip().endswith("?"), res2.reply)

# 3. "What subjects do you teach?"
res3 = engine.handle_message("test-session-3", "What subjects do you teach?")
check("3. Subjects: No 'Would you like to know'", "Would you like to know" not in res3.reply, res3.reply)
check("3. Subjects: Accurately mentions subjects", "math" in res3.reply.lower() or "science" in res3.reply.lower(), res3.reply)
check("3. Subjects: Stops directly without asking another question", not res3.reply.strip().endswith("?"), res3.reply)

# 4. "What is the curriculum?"
res4 = engine.handle_message("test-session-4", "What is the curriculum?")
check("4. Curriculum: No 'Would you like to know'", "Would you like to know" not in res4.reply, res4.reply)
check("4. Curriculum: Mentions curriculum details", "cbse" in res4.reply.lower() or "icse" in res4.reply.lower() or "curriculum" in res4.reply.lower(), res4.reply)
check("4. Curriculum: Stops directly without asking another question", not res4.reply.strip().endswith("?"), res4.reply)

# 5. "Tell me about the classes."
res5 = engine.handle_message("test-session-5", "Tell me about the classes.")
check("5. Tell me about classes: No 'Would you like to know'", "Would you like to know" not in res5.reply, res5.reply)
check("5. Tell me about classes: Relevant information provided", len(res5.reply) > 20, res5.reply)
check("5. Tell me about classes: No pushy sales funnel", "book a demo" not in res5.reply.lower() or "admin@wementors.co" in res5.reply, res5.reply)

# 6. "Help."
res6 = engine.handle_message("test-session-6", "Help.")
check("6. Help: Intent is help", res6.intent == "help", f"Intent: {res6.intent}")
check("6. Help: Clarifies what user needs help with", "what would you like help with" in res6.reply.lower(), res6.reply)
check("6. Help: Never mentions fees or sales pitches", "fee" not in res6.reply.lower() and "price" not in res6.reply.lower(), res6.reply)

# 7. "I want to know more."
res7 = engine.handle_message("test-session-7", "I want to know more.")
check("7. I want to know more: Intent is clarify", res7.intent == "clarify", f"Intent: {res7.intent}")
check("7. I want to know more: Asks which area user is interested in", "classes" in res7.reply.lower() and "subjects" in res7.reply.lower(), res7.reply)

# 8. "How much?"
res8 = engine.handle_message("test-session-8", "How much?")
check("8. How much (no context): Intent is clarify", res8.intent == "clarify", f"Intent: {res8.intent}")
check("8. How much: Asks for clarification on fees vs duration vs program", "clarify" in res8.reply.lower() or "fees" in res8.reply.lower(), res8.reply)

# 9. "Thanks."
res9 = engine.handle_message("test-session-9", "Thanks.")
check("9. Thanks: Intent is thanks", res9.intent == "thanks", f"Intent: {res9.intent}")
check("9. Thanks: Simple polite response", any(w in res9.reply.lower() for w in ["welcome", "happy to help", "anytime"]), res9.reply)
check("9. Thanks: Does NOT append a suggested question", not res9.reply.strip().endswith("?"), res9.reply)
check("9. Thanks: No 'Would you like to know'", "Would you like to know" not in res9.reply, res9.reply)

# 10. A clear question followed by a relevant follow-up question
sid10 = "test-session-10"
res10_a = engine.handle_message(sid10, "How are classes conducted?")
database.log_message(sid10, "user", "How are classes conducted?")
database.log_message(sid10, "assistant", res10_a.reply, res10_a.intent, ",".join(res10_a.matched_entry_ids))

res10_b = engine.handle_message(sid10, "What is the batch size?")
check("10. Follow-up: Answers batch size accurately", "batch" in res10_b.reply.lower() or "students" in res10_b.reply.lower() or "one-on-one" in res10_b.reply.lower(), res10_b.reply)
check("10. Follow-up: No 'Would you like to know'", "Would you like to know" not in res10_b.reply, res10_b.reply)
check("10. Follow-up: Stops cleanly without asking another question", not res10_b.reply.strip().endswith("?"), res10_b.reply)

# 11. A question requiring clarification
res11 = engine.handle_message("test-session-11", "Compare programs")
check("11. Clarification needed: Asks which two programs", "which two" in res11.reply.lower() or "clarify" in res11.intent, res11.reply)

# 12. An out-of-scope question
res12 = engine.handle_message("test-session-12", "Do you teach culinary arts and cooking?")
check("12. Out-of-scope: Off-topic or low-confidence redirect", res12.intent in ("off_topic", "low_confidence", "general"), f"Intent: {res12.intent}")
check("12. Out-of-scope: Does not fabricate cooking courses", "cooking" not in res12.reply.lower() or "not in my verified records" in res12.reply or "i'm here to help with wementors" in res12.reply.lower(), res12.reply)
check("12. Out-of-scope: No 'Would you like to know'", "Would you like to know" not in res12.reply, res12.reply)

print("\n--- Regression Check: No 'Would you like to know' in any KB entry reply ---")
for entry in engine.entries:
    res = engine.handle_message(f"test-kb-{entry.id}", entry.question)
    if "Would you like to know" in res.reply:
        check(f"KB Entry '{entry.id}' clean of suggestions", False, res.reply)
        break
else:
    check("All KB entries answer cleanly without appending 'Would you like to know'", True)

print("\n" + "=" * 50)
if failures:
    print(f"FAILED checks ({len(failures)}):")
    for f in failures:
        print(f" - {f}")
    sys.exit(1)
else:
    print("ALL 12 SCENARIOS AND REGRESSION CHECKS PASSED!")
    print("=" * 50 + "\n")
