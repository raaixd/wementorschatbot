import urllib.request
import json
import time

def test_chat(msg, session_id="test-live-sess"):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/chat",
        data=json.dumps({"message": msg, "session_id": session_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"User: {msg}")
    print(f"Bot ({res.get('intent')}, {elapsed_ms:.1f}ms):\n{res.get('reply')}\n")

test_chat("what can u help me with")
test_chat("information")
test_chat("Tell me about your Python course")
test_chat("How much is the Python course?")
test_chat("Demo class")
test_chat("How do I book?")
test_chat("What?")
test_chat("Which subjects do you teach in middle school, and how do mentors assess my child's progress?")
