"""export_feedback_cases turns thumbs-down ratings into unlabeled candidates."""
import importlib.util
import sqlite3
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_feedback_cases.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_feedback_cases", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            source TEXT, intent TEXT, matched_entry_ids TEXT, confidence REAL, created_at TEXT);
        CREATE TABLE feedback (id INTEGER PRIMARY KEY, session_id TEXT, message_id INTEGER,
            rating TEXT, comment TEXT, created_at TEXT);
        """
    )
    rows = [
        (1, "s1", "user", "Do you teach Grade 12?"),
        (2, "s1", "assistant", "Not listed."),
        (3, "s1", "user", "What about JEE?"),
        (4, "s1", "assistant", "Not listed either."),
        (5, "s2", "user", "hello"),
        (6, "s2", "assistant", "Hi!"),
    ]
    for mid, sid, role, content in rows:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, intent, confidence) VALUES (?,?,?,?,?,?)",
            (mid, sid, role, content, "x" if role == "assistant" else None, 0.1 if role == "assistant" else None),
        )
    return conn


class FeedbackExportTests(unittest.TestCase):
    def test_only_thumbs_down_become_candidates(self):
        conn = _db()
        conn.execute("INSERT INTO feedback VALUES (1,'s1',4,'down','wrong','t')")
        conn.execute("INSERT INTO feedback VALUES (2,'s2',6,'up',NULL,'t')")
        cases = _load().build_candidates(conn)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["turns"], ["Do you teach Grade 12?", "What about JEE?"])
        self.assertTrue(cases[0]["needs_review"])
        self.assertIsNone(cases[0]["expected_intent"])
        self.assertEqual(cases[0]["observed"]["bot_reply"], "Not listed either.")
        self.assertEqual(cases[0]["observed"]["visitor_comment"], "wrong")

    def test_turns_stop_at_the_rated_reply(self):
        conn = _db()
        conn.execute("INSERT INTO feedback VALUES (1,'s1',2,'down',NULL,'t')")
        self.assertEqual(_load().build_candidates(conn)[0]["turns"], ["Do you teach Grade 12?"])

    def test_missing_message_id_uses_latest_reply(self):
        conn = _db()
        conn.execute("INSERT INTO feedback VALUES (1,'s1',NULL,'down',NULL,'t')")
        self.assertEqual(_load().build_candidates(conn)[0]["observed"]["bot_reply"], "Not listed either.")

    def test_duplicate_ratings_of_same_reply_collapse(self):
        conn = _db()
        conn.execute("INSERT INTO feedback VALUES (1,'s1',4,'down',NULL,'t')")
        conn.execute("INSERT INTO feedback VALUES (2,'s1',4,'down','again','t')")
        self.assertEqual(len(_load().build_candidates(conn)), 1)

    def test_only_with_comment_filter(self):
        conn = _db()
        conn.execute("INSERT INTO feedback VALUES (1,'s1',4,'down',NULL,'t')")
        self.assertEqual(_load().build_candidates(conn, include_empty_comments=False), [])


if __name__ == "__main__":
    unittest.main()
