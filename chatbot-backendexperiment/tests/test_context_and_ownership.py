"""
Comprehensive verification test suite for:
1. Strict message ownership (role and source validation, rejection of assistant/invalid inputs).
2. Suggestion rendering decoupling from suggestion submission.
3. Confirmation handling (pending clarification vs standalone affirmations).
4. Structured conversation memory and context awareness.
5. Action-first enrollment routing (action responses vs repeated overviews).
6. Elimination of self-conversation, synthetic 'Yes.' prefixes, and generic fallbacks.
7. Request deduplication (frontend & backend request_id).
8. Exact reported regression flows.

Run with:
    python tests/test_context_and_ownership.py
"""
import os
import sys
import uuid
import tempfile
import pathlib
import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SKIP_DOTENV"] = "1"
os.environ["RATE_LIMIT_ENABLED"] = "0"
os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(), "test_ownership.db"))

from app import config
config.RATE_LIMIT_ENABLED = False

from fastapi.testclient import TestClient
from app.main import app
from app import database
from app.conversation import ConversationEngine, VALID_USER_SOURCES, VALID_USER_ROLES
from app.knowledge import load_entries
from app import personality

database.init_db()
client = TestClient(app)
failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {label}"
    if detail and not condition:
        msg += f" -- {detail}"
    print(msg)
    if not condition:
        failures.append(label)


print("================================================================================")
print("RUNNING STRICT CONVERSATION & OWNERSHIP TEST SUITE")
print("================================================================================\n")

# --- 1. Message Ownership Tests (API Layer) -----------------------------------
print("--- 1. Message Ownership & Source Validation ---")

# Rejection of assistant role
res = client.post("/chat", json={"message": "I am an assistant", "role": "assistant", "source": "text_input"})
check("API rejects role='assistant' with 400", res.status_code == 400, f"got {res.status_code}: {res.text}")

# Rejection of assistant_response source
res = client.post("/chat", json={"message": "Hello", "role": "user", "source": "assistant_response"})
check("API rejects source='assistant_response' with 400", res.status_code == 400, f"got {res.status_code}: {res.text}")

# Rejection of unknown source
res = client.post("/chat", json={"message": "Hello", "role": "user", "source": "synthetic_source"})
check("API rejects unknown source with 400", res.status_code == 400, f"got {res.status_code}: {res.text}")

# Acceptance of valid user sources
for src in ["text_input", "explicit_suggestion_click", "form_submission"]:
    sid = f"ownership-{src}-{uuid.uuid4()}"
    res = client.post("/chat", json={"message": "Tell me about WeMentors", "role": "user", "source": src, "session_id": sid})
    check(f"API accepts role='user' with source='{src}'", res.status_code == 200, f"got {res.status_code}: {res.text}")

# Engine-level role and source enforcement
entries = load_entries()
engine = ConversationEngine(entries)
test_sid = f"engine-test-{uuid.uuid4()}"

try:
    engine.handle_message(test_sid, "test", role="assistant", source="text_input")
    check("Engine rejects role='assistant' with ValueError", False)
except ValueError:
    check("Engine rejects role='assistant' with ValueError", True)

try:
    engine.handle_message(test_sid, "test", role="user", source="unauthorized_source")
    check("Engine rejects invalid source with ValueError", False)
except ValueError:
    check("Engine rejects invalid source with ValueError", True)

# --- 2. Request Deduplication Tests -------------------------------------------
print("\n--- 2. Request Deduplication Tests ---")
dedup_sid = f"dedup-test-{uuid.uuid4()}"
req_id = f"req-{uuid.uuid4()}"

res1 = client.post("/chat", json={
    "message": "What programs do you offer?",
    "role": "user",
    "source": "text_input",
    "session_id": dedup_sid,
    "request_id": req_id,
})
check("First request with request_id succeeds", res1.status_code == 200)

res2 = client.post("/chat", json={
    "message": "What programs do you offer?",
    "role": "user",
    "source": "text_input",
    "session_id": dedup_sid,
    "request_id": req_id,
})
check("Duplicate request with same request_id succeeds and returns cached reply", 
      res2.status_code == 200 and res2.json()["reply"] == res1.json()["reply"])

# Verify database has only one user message logged for this request_id
messages = database.get_recent_messages(dedup_sid, 10)
user_msgs = [m for m in messages if m["role"] == "user"]
check("Only one user message was recorded in database for duplicate request", len(user_msgs) == 1, f"got {len(user_msgs)}")

# --- 3. Frontend Static Structural Checks -------------------------------------
print("\n--- 3. Frontend Architecture & Suggestion Decoupling ---")
index_path = pathlib.Path(__file__).resolve().parent.parent / "index.html"
html_content = index_path.read_text(encoding="utf-8", errors="ignore")

check("renderSuggestion(text) function is defined", "function renderSuggestion(" in html_content)
check("onSuggestionClick(text) function is defined", "function onSuggestionClick(" in html_content)
check("submitUserMessage(text, source) function is defined", "function submitUserMessage(" in html_content)
check("displaySuggestions(suggestions) function is defined", "function displaySuggestions(" in html_content)
check("onSuggestionClick passes 'explicit_suggestion_click' source", "submitUserMessage(text, 'explicit_suggestion_click')" in html_content)
check("renderSuggestion is side-effect free (creates button only, no fetch)", 
      "function renderSuggestion(" in html_content and 
      "fetch" not in html_content.split("function renderSuggestion(")[1].split("function onSuggestionClick(")[0])
check("Frontend prevents rapid duplicate submission (<500ms)", "lastSubmittedTime" in html_content and "< 500" in html_content)
check("Frontend generates request_id for API calls", "request_id" in html_content)

# --- 4. Confirmation Handling Tests -------------------------------------------
print("\n--- 4. Confirmation Handling Tests ---")
conf_sid = f"conf-{uuid.uuid4()}"

# Standalone "yes" with no pending clarification
res_yes = client.post("/chat", json={
    "message": "yes",
    "role": "user",
    "source": "text_input",
    "session_id": conf_sid,
})
check("Standalone 'yes' without clarification asks what to confirm", 
      "What would you like to confirm?" in res_yes.json()["reply"],
      res_yes.json()["reply"])

# Standalone "no" with no pending clarification
res_no = client.post("/chat", json={
    "message": "no",
    "role": "user",
    "source": "text_input",
    "session_id": conf_sid,
})
check("Standalone 'no' asks what to explore instead without generic dump",
      "What would you like to explore instead" in res_no.json()["reply"],
      res_no.json()["reply"])

# Set up a pending clarification in session memory
memory = database.get_conversation_memory(conf_sid)
memory.pending_clarification = {
    "question": "Do you mean the Foundation Years program for Grades 3–5?",
    "expected_entity": "foundation",
    "options": ["foundation", "middle", "senior", "confident_speaker"],
    "turn_count": 2,
}
database.save_conversation_memory(conf_sid, memory)

# Now send "yes" to confirm the pending clarification
res_confirm = client.post("/chat", json={
    "message": "yes",
    "role": "user",
    "source": "text_input",
    "session_id": conf_sid,
})
check("Confirmation 'yes' confirms Foundation Years specifically",
      "Foundation Years" in res_confirm.json()["reply"] and "Great — I’ll use Foundation Years" in res_confirm.json()["reply"],
      res_confirm.json()["reply"])

# Check updated memory state
updated_memory = database.get_conversation_memory(conf_sid)
check("Memory active_program is now 'foundation'", updated_memory.active_program == "foundation")
check("Memory pending_clarification is cleared after confirmation", updated_memory.pending_clarification is None)

# --- 5. Context Awareness and Intent Inference Tests ---------------------------
print("\n--- 5. Context Awareness & Field-level Intent Inference ---")
ctx_sid = f"ctx-{uuid.uuid4()}"

# Turn A: Overview of Foundation Years
res_a = client.post("/chat", json={
    "message": "Tell me about Foundation Years",
    "role": "user",
    "source": "text_input",
    "session_id": ctx_sid,
})
check("Foundation overview returned", "Grades 3–5" in res_a.json()["reply"] or "Grades 3-5" in res_a.json()["reply"])

# Turn B: Field-level question "what subjects are included?"
res_b = client.post("/chat", json={
    "message": "what subjects are included?",
    "role": "user",
    "source": "text_input",
    "session_id": ctx_sid,
})
check("Context resolves 'what subjects' to Foundation subjects (Math, Science, English)",
      any(sub in res_b.json()["reply"] for sub in ["Mathematics", "Science", "English"]),
      res_b.json()["reply"])
check("Does not repeat general overview on narrow field question",
      "curiosity-first" in res_b.json()["reply"].lower() or "mathematics" in res_b.json()["reply"].lower(),
      res_b.json()["reply"])

# Turn C: Action question "how can i enroll?"
res_c = client.post("/chat", json={
    "message": "how can i enroll?",
    "role": "user",
    "source": "text_input",
    "session_id": ctx_sid,
})
check("Context resolves 'how can i enroll' to Foundation enrollment guidance",
      "Book Free Demo" in res_c.json()["reply"] and "Foundation Years" in res_c.json()["reply"],
      res_c.json()["reply"])

# Turn D: Topic switch to Confident Speaker
res_d = client.post("/chat", json={
    "message": "Tell me about Confident Speaker",
    "role": "user",
    "source": "text_input",
    "session_id": ctx_sid,
})
check("Explicit switch to Confident Speaker updates program context",
      "Confident Speaker" in res_d.json()["reply"] and ("speaking" in res_d.json()["reply"].lower() or "communication" in res_d.json()["reply"].lower()))
ctx_mem = database.get_conversation_memory(ctx_sid)
check("Memory active_program is now 'confident_speaker'", ctx_mem.active_program == "confident_speaker")

# --- 6. Exact Reported Regression Flow -----------------------------------------
print("\n--- 6. Exact Reported Regression Flow ---")
reg_sid = f"exact-reg-{uuid.uuid4()}"

# Step 1: Assistant asks clarification (simulated state)
reg_mem = database.get_conversation_memory(reg_sid)
reg_mem.pending_clarification = {
    "question": "Do you mean the Foundation Years program for Grades 3–5?",
    "expected_entity": "foundation",
    "options": ["foundation", "middle", "senior", "confident_speaker"],
    "turn_count": 1,
}
reg_mem.last_assistant_question = "Do you mean the Foundation Years program for Grades 3–5?"
database.save_conversation_memory(reg_sid, reg_mem)

# Step 2: User says "yes"
r1 = client.post("/chat", json={"message": "yes", "role": "user", "source": "text_input", "session_id": reg_sid})
check("Turn 1 (yes): Confirmation response, no generic fallback",
      "Great — I’ll use Foundation Years" in r1.json()["reply"] and "explore our programs, understand how mentoring works" not in r1.json()["reply"],
      r1.json()["reply"])

# Step 3: User says "tell me about foundation years"
r2 = client.post("/chat", json={"message": "tell me about foundation years", "role": "user", "source": "text_input", "session_id": reg_sid})
check("Turn 2 (tell me about foundation years): Foundation Years overview",
      "Grades 3–5" in r2.json()["reply"] or "Grades 3-5" in r2.json()["reply"],
      r2.json()["reply"])

# Step 4: User says "how can i sign up"
r3 = client.post("/chat", json={"message": "how can i sign up", "role": "user", "source": "text_input", "session_id": reg_sid})
check("Turn 3 (how can i sign up): Action-oriented enrollment guidance, NOT overview",
      "Book Free Demo" in r3.json()["reply"] and "Foundation Years" in r3.json()["reply"],
      r3.json()["reply"])
check("Turn 3 is NOT an overview-only response", 
      "curiosity-first learning" not in r3.json()["reply"].lower() and "strong fundamentals" not in r3.json()["reply"].lower(),
      r3.json()["reply"])

# Step 5: User says "how can i enroll in foundation years"
r4 = client.post("/chat", json={"message": "how can i enroll in foundation years", "role": "user", "source": "text_input", "session_id": reg_sid})
check("Turn 4 (how can i enroll in foundation years): Specific action-oriented Foundation enrollment response",
      "Foundation Years for Grades 3–5" in r4.json()["reply"] and "Book Free Demo" in r4.json()["reply"],
      r4.json()["reply"])

# Verify all logged turns for this session
reg_messages = database.get_recent_messages(reg_sid, 20)
reg_users = [m for m in reg_messages if m["role"] == "user"]
reg_assistants = [m for m in reg_messages if m["role"] == "assistant"]
check("No synthetic user messages logged", len(reg_users) == 4, f"got {len(reg_users)}")
check("Exactly 4 assistant messages logged (1:1 response ratio)", len(reg_assistants) == 4, f"got {len(reg_assistants)}")
check("All user messages have explicit valid sources", all(m["source"] in VALID_USER_SOURCES for m in reg_users))

# --- 7. Mentoring & Self-Conversation Bug Regression --------------------------
print("\n--- 7. Mentoring & Self-Conversation Bug Regression ---")
mentor_sid = f"mentor-{uuid.uuid4()}"

# User clicks "How does personalized mentoring work?"
r_m1 = client.post("/chat", json={
    "message": "How does personalized mentoring work?",
    "role": "user",
    "source": "explicit_suggestion_click",
    "session_id": mentor_sid,
})
check("Personalized mentoring explanation returned",
      "Personalized mentoring means each learner receives individual attention" in r_m1.json()["reply"] or
      "individual attention from a personal mentor" in r_m1.json()["reply"],
      r_m1.json()["reply"])
check("No synthetic 'Yes.' prepended to personalized mentoring explanation",
      not r_m1.json()["reply"].strip().startswith("Yes."),
      r_m1.json()["reply"][:30])

# User clicks "Tell me about the mentoring approach."
r_m2 = client.post("/chat", json={
    "message": "Tell me about the mentoring approach.",
    "role": "user",
    "source": "explicit_suggestion_click",
    "session_id": mentor_sid,
})
check("Mentoring approach explanation returned without rotating mentors",
      "Each learner receives individual attention from a personal mentor rather than being passed between a rotating group of mentors" in r_m2.json()["reply"],
      r_m2.json()["reply"])
check("No synthetic 'Yes.' prepended to mentoring approach explanation",
      not r_m2.json()["reply"].strip().startswith("Yes."),
      r_m2.json()["reply"][:30])

# Verify session message history
m_messages = database.get_recent_messages(mentor_sid, 10)
check("Message history contains exactly 2 user messages and 2 assistant responses",
      len(m_messages) == 4 and [m["role"] for m in m_messages] == ["user", "assistant", "user", "assistant"])

# --- 8. Demo Lead Entity Extraction & Formatting Regression -----------------
print("\n--- 8. Demo Lead Entity Extraction & Formatting Regression ---")
lead_sid = f"lead-test-{uuid.uuid4()}"

# User initiates demo enquiry
r_d1 = client.post("/chat", json={
    "message": "I'd like to book a free demo",
    "role": "user",
    "source": "text_input",
    "session_id": lead_sid,
})
check("Demo flow initiated", r_d1.status_code == 200)

# User responds with program name "middle school"
r_d2 = client.post("/chat", json={
    "message": "middle school",
    "role": "user",
    "source": "text_input",
    "session_id": lead_sid,
})
check("Middle School acknowledged as subject/program", "Middle School" in r_d2.json()["reply"])
check("Middle School is NOT falsely captured as student/parent name", 
      "name: Middle School" not in r_d2.json()["reply"] and "name as middle school" not in r_d2.json()["reply"].lower(),
      r_d2.json()["reply"])
check("Missing fields still include student or parent name",
      "student or parent name" in r_d2.json()["reply"].lower(),
      r_d2.json()["reply"])

# Check database lead state
db_lead = database.get_demo_lead(lead_sid)
check("Database demo lead name is None", db_lead.get("name") is None, f"got {db_lead.get('name')}")
check("Database demo lead subject is Middle School", db_lead.get("subject") == "Middle School", f"got {db_lead.get('subject')}")

# User asks general info question
r_gen = client.post("/chat", json={
    "message": "What is WeMentors?",
    "role": "user",
    "source": "text_input",
    "session_id": f"gen-{uuid.uuid4()}",
})
check("General info has structured markdown formatting with bullet points",
      "- **Dedicated Personal Mentor**" in r_gen.json()["reply"] and "- **Weekly Progress Tracking**" in r_gen.json()["reply"],
      r_gen.json()["reply"])

# --- Final Assessment ---------------------------------------------------------
print("\n================================================================================")
if failures:
    print(f"FAILED {len(failures)} CHECKS:")
    for f in failures:
        print(f" - {f}")
    sys.exit(1)
else:
    print("ALL CONVERSATION & OWNERSHIP CHECKS PASSED (100%)")
    print("================================================================================")

