import sys
import urllib.request
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

queries = [
    "Hi",
    "Help",
    "I want to know more",
    "How much?",
    "I want a demo class",
    "I want to enquire about joining",
    "Can you help me choose a class?",
    "What are the course fees?",
    "What subjects do you teach?",
    "Can you make me a pizza?",
    "Thanks a lot!",
]

session_id = "00000000-0000-4000-8000-000000000099"

for q in queries:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/chat",
        data=json.dumps({"message": q, "session_id": session_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode("utf-8"))
    print(f"--- Q: {q} ---")
    print(f"Intent: {data.get('intent')}")
    print(f"Reply: {data.get('reply')}\n")
