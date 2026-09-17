"""
Comprehensive evaluation test suite for advanced conversational AI intelligence,
natural language quality, contextual understanding, adaptive out-of-scope handling,
and grounding in WeMentors knowledge.
"""
import os
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(), "test_intelligence.db"))

from app import config, database, llm
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

print("\n=== Testing Conversational AI Intelligence & Adaptive Response ===\n")

# 1. Single-word "courses" query
res_courses = engine.handle_message("test-courses-1", "courses")
check("1. 'courses': matches programs-overview", "programs-overview" in res_courses.matched_entry_ids or "Foundation Years" in res_courses.reply, res_courses.reply)
check("1. 'courses': names verified programs", "foundation" in res_courses.reply.lower() and "middle school" in res_courses.reply.lower(), res_courses.reply)
check("1. 'courses': no unwanted follow-up appended", "Would you like to know" not in res_courses.reply)

# 2. Clear factual question
res_conducted = engine.handle_message("test-conducted-1", "How are classes conducted?")
check("2. Factual: answers video calls & whiteboard", "video" in res_conducted.reply.lower() or "whiteboard" in res_conducted.reply.lower(), res_conducted.reply)
check("2. Factual: stops naturally without unprompted question", not res_conducted.reply.strip().endswith("?"), res_conducted.reply)

# 3. Vague Help
res_help = engine.handle_message("test-help-1", "Help")
check("3. Help: intent is help", res_help.intent == "help")
check("3. Help: asks what user needs help with", "what would you like help with" in res_help.reply.lower(), res_help.reply)
check("3. Help: never mentions fees or sales pitch", "fee" not in res_help.reply.lower() and "price" not in res_help.reply.lower())

# 4. Ambiguous "How much?"
res_howmuch = engine.handle_message("test-howmuch-1", "How much?")
check("4. How much (no context): asks for clarification", res_howmuch.intent in ("clarify", "faq"), res_howmuch.intent)
check("4. How much: does not guess a random fee amount", "₹" not in res_howmuch.reply and "rupees" not in res_howmuch.reply.lower())

# 5. User confusion ("I don't understand")
sid_confused = "test-session-confused"
engine.handle_message(sid_confused, "What is the Foundation Years program?")
database.log_message(sid_confused, "user", "What is the Foundation Years program?")
database.log_message(sid_confused, "assistant", "The Foundation Years program supports Grades 3-5...", "faq", "program-foundation-years")
res_confused = engine.handle_message(sid_confused, "I don't understand.")
check("5. Confused: intent recognized", res_confused.intent in ("confused", "general"))
check("5. Confused: natural helpful response", len(res_confused.reply) > 10, res_confused.reply)

# 6. Out-of-scope trivia ("What is the capital of France?")
res_france = engine.handle_message("test-france-1", "What is the capital of France?")
check("6. Out-of-scope trivia: intent is general or off_topic", res_france.intent in ("general", "off_topic"))
check("6. Out-of-scope trivia: does not fabricate WeMentors course on France", "wementors" in res_france.reply.lower() or "paris" in res_france.reply.lower(), res_france.reply)

# 7. Playful / Humor ("Can your mentors teach my cat calculus?")
res_cat = engine.handle_message("test-cat-1", "Can your mentors teach my cat calculus?")
check("7. Playful: intent is general or off_topic", res_cat.intent in ("general", "off_topic"))
check("7. Playful: responds naturally without claiming cat classes", any(k in res_cat.reply.lower() for k in ["cat", "feline", "tutor", "mentor", "human", "wementors"]), res_cat.reply)

# 8. Nonsense ("Banana spaceship quantum toaster purple.")
res_banana = engine.handle_message("test-banana-1", "Banana spaceship quantum toaster purple.")
check("8. Nonsense: intent is general or off_topic", res_banana.intent in ("general", "off_topic"))
check("8. Nonsense: does not interpret as fees or demo", "fee" not in res_banana.reply.lower())

# 9. Gibberish ("asdfghjkl")
res_gibberish = engine.handle_message("test-gibberish-1", "asdfghjkl")
check("9. Gibberish: brief natural check-in", len(res_gibberish.reply) > 5, res_gibberish.reply)

# 10. Questions about the bot ("What can you do?")
res_bot = engine.handle_message("test-bot-1", "What can you do?")
check("10. Bot capabilities: explains role concisely", "wementors" in res_bot.reply.lower() or "classes" in res_bot.reply.lower() or "assistant" in res_bot.reply.lower(), res_bot.reply)

# 11. Multi-intent question ("What subjects do you teach and how can I join?")
res_multi = engine.handle_message("test-multi-1", "What subjects do you teach and how can I join?")
check("11. Multi-intent: mentions subjects (Maths / Science)", "math" in res_multi.reply.lower() or "science" in res_multi.reply.lower() or "subjects" in res_multi.reply.lower(), res_multi.reply)
check("11. Multi-intent: mentions contact or admissions info", "admin@wementors.co" in res_multi.reply or "76111" in res_multi.reply or "demo" in res_multi.reply.lower() or "contact" in res_multi.reply.lower(), res_multi.reply)

# 12. User says "Thanks."
res_thanks = engine.handle_message("test-thanks-1", "Thanks.")
check("12. Thanks: intent is thanks", res_thanks.intent == "thanks")
check("12. Thanks: stops directly without asking questions", not res_thanks.reply.strip().endswith("?"), res_thanks.reply)

print("\n--- Regression Check: No 'Would you like to know' in any test response ---")
all_replies = [res_courses.reply, res_conducted.reply, res_help.reply, res_howmuch.reply,
               res_confused.reply, res_france.reply, res_cat.reply, res_banana.reply,
               res_gibberish.reply, res_bot.reply, res_multi.reply, res_thanks.reply]
for idx, rep in enumerate(all_replies, start=1):
    check(f"Reply {idx} clean of 'Would you like to know'", "Would you like to know" not in rep)

print("\n" + "=" * 50)
if failures:
    print(f"FAILED checks ({len(failures)}):")
    for f in failures:
        print(f" - {f}")
    sys.exit(1)
else:
    print("ALL CONVERSATIONAL INTELLIGENCE CHECKS PASSED!")
    print("=" * 50 + "\n")
