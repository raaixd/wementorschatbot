"""The request cache and rate limiter are shared across sessions and threads."""
import sys
import threading
import time
import uuid
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi.testclient import TestClient

from app import main
from app.ratelimit import RateLimiter


def test_reused_request_id_never_crosses_sessions():
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    shared_id = str(uuid.uuid4())
    with TestClient(main.app) as client:
        a = client.post("/chat", json={"message": "My name is Priya, tell me about grade 7",
                                       "session_id": first, "request_id": shared_id}).json()
        b = client.post("/chat", json={"message": "do you offer online classes?",
                                       "session_id": second, "request_id": shared_id}).json()
        repeat = client.post("/chat", json={"message": "tell me about grade 7",
                                            "session_id": first, "request_id": shared_id}).json()
    assert b["session_id"] == second
    assert "Priya" not in b["reply"]
    assert repeat == a  # same session and id is still deduplicated


def _hammer(fn, threads=4, seconds=0.5):
    errors = []
    deadline = time.time() + seconds

    def work(t):
        i = 0
        try:
            while time.time() < deadline:
                i += 1
                fn(t, i)
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(repr(exc))

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force frequent thread switches
    try:
        workers = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    finally:
        sys.setswitchinterval(interval)
    return errors


def test_request_cache_is_thread_safe():
    response = main.ChatResponse(reply="x", session_id="s", intent="i")
    main._recent_request_cache.update({("seed", str(i)): (time.time(), response) for i in range(2000)})
    try:
        assert _hammer(lambda t, i: main._check_and_cache_request("s", f"{t}-{i}", response)) == []
    finally:
        main._recent_request_cache.clear()


def test_rate_limiter_is_thread_safe():
    limiter = RateLimiter(5, 60, prune_interval_seconds=0)
    for i in range(2000):
        limiter.allow(f"seed{i}")
    assert _hammer(lambda t, i: limiter.allow(f"{t}-{i}")) == []
