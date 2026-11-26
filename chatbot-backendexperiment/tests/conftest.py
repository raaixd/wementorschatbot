"""Most tests exercise the routed engine, which is the backup when the model
fails. test_full_kb_mode.py covers the path used while the model works."""
import os

os.environ.setdefault("ANSWER_MODE", "routed")
