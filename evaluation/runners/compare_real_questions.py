"""Compare the current bot with a simpler design on real visitor questions.

  current: the production ConversationEngine (routing + retrieval + LLM)
  simple : two hard rules (injection, booking requests), then the LLM with
           the whole knowledge base as context

A person grades the answers without being told which system wrote which.

    python evaluation/runners/compare_real_questions.py run --questions q.txt
    # fill grade_A and grade_B in sheet.csv with "pass" or "fail"
    python evaluation/runners/compare_real_questions.py score

Question file: one visitor message per line. A blank line starts a new
conversation. Lines starting with # are ignored.

Questions are sent to the configured LLM provider. Emails and long numbers
are redacted first, but names are NOT: remove those yourself. Output is
written under the gitignored evaluation/cache/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "evaluation" / "cache" / "compare"
SYSTEMS = ("current", "simple")
LLM_FAILED = "[LLM FAILED - do not grade]"
FIELDS = ["id", "conversation", "turn", "question", "answer_A", "answer_B", "grade_A", "grade_B", "notes"]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUMBER_RE = re.compile(r"\+?\d(?:[ ().-]{0,2}\d){7,}")


def redact(text: str) -> str:
    return _NUMBER_RE.sub("[number]", _EMAIL_RE.sub("[email]", text))


def read_conversations(path: Path) -> list[list[str]]:
    conversations: list[list[str]] = [[]]
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line:
            conversations[-1].append(redact(line))
        elif conversations[-1]:
            conversations.append([])
    return [c for c in conversations if c]


def _load_app():
    # A throwaway database keeps test traffic out of the real one.
    os.environ["DATABASE_PATH"] = str(Path(tempfile.mkdtemp()) / "compare.db")
    sys.path.insert(0, str(ROOT / "chatbot-backendexperiment"))
    from app import config, database, llm, personality
    from app.conversation import ConversationEngine
    from app.knowledge import load_entries
    from app.retrieval import ScoredEntry

    return config, database, llm, personality, ConversationEngine, load_entries, ScoredEntry


def run(questions: Path, out: Path, delay: float, seed: int | None) -> int:
    conversations = read_conversations(questions)
    if not conversations:
        print("No questions found.", file=sys.stderr)
        return 1
    config, database, llm, personality, Engine, load_entries, ScoredEntry = _load_app()
    if isinstance(llm.get_active_provider(), llm.NullProvider):
        print("No LLM provider is configured. Set GEMINI_API_KEY (or GROQ_API_KEY) in .env.", file=sys.stderr)
        return 1
    sheet = out / "sheet.csv"
    if sheet.exists():
        print(f"{sheet} already exists and may hold grades. Move it or pick another --out.", file=sys.stderr)
        return 1

    database.init_db()
    entries = load_entries()
    engine = Engine(entries)
    everything = [ScoredEntry(entry, 1.0) for entry in entries]
    hard_rules = {
        "injection_attempt": personality.INJECTION_DEFLECTION,
        "demo_transaction_request": personality.DEMO_TRANSACTION_REQUEST_RESPONSE,
    }
    rng = random.Random(seed)
    rows, key, failures = [], {}, 0

    for c_index, turns in enumerate(conversations, 1):
        session_id = str(uuid.uuid4())
        database.ensure_session(session_id)
        history: list[dict] = []
        # One order per conversation, so each column stays one system across follow-ups.
        order = rng.sample(SYSTEMS, 2)
        for t_index, question in enumerate(turns, 1):
            rule = hard_rules.get(engine.detect_intent(question))
            simple = rule or llm.generate_answer(question, everything, history[-config.LLM_HISTORY_TURNS:])
            if simple is None:
                simple, failures = LLM_FAILED, failures + 1
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": simple}]

            database.log_message(session_id, "user", question, source="text_input")
            result = engine.handle_message(session_id, question)
            database.log_message(session_id, "assistant", result.reply, intent=result.intent)

            answers = {"current": result.reply, "simple": simple}
            row_id = f"c{c_index:03d}t{t_index:02d}"
            key[row_id] = {"A": order[0], "B": order[1]}
            rows.append({
                "id": row_id, "conversation": c_index, "turn": t_index, "question": question,
                "answer_A": answers[order[0]], "answer_B": answers[order[1]],
                "grade_A": "", "grade_B": "", "notes": "",
            })
            time.sleep(delay)

    out.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {sheet}. Do not open key.json until grading is done.")
    if failures:
        print(f"WARNING: {failures} simple-design call(s) failed or were rejected; those rows are not scored.")
    return 0


def score_rows(rows: list[dict], key: dict) -> dict:
    totals = {"graded": 0, "skipped": 0, "current": 0, "simple": 0, "only_current": 0, "only_simple": 0}
    for row in rows:
        grades = {key[row["id"]][side]: row[f"grade_{side}"].strip().lower() for side in "AB"}
        failed = LLM_FAILED in (row["answer_A"], row["answer_B"])
        if failed or any(g not in ("pass", "fail") for g in grades.values()):
            totals["skipped"] += 1
            continue
        totals["graded"] += 1
        passed = {name: grade == "pass" for name, grade in grades.items()}
        for name in SYSTEMS:
            totals[name] += passed[name]
        if passed["current"] != passed["simple"]:
            totals["only_current" if passed["current"] else "only_simple"] += 1
    return totals


def score(out: Path) -> int:
    with (out / "sheet.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    totals = score_rows(rows, json.loads((out / "key.json").read_text(encoding="utf-8")))
    if not totals["graded"]:
        print("No fully graded rows yet.", file=sys.stderr)
        return 1
    print(f"Graded rows: {totals['graded']} (skipped: {totals['skipped']})")
    for name in SYSTEMS:
        print(f"  {name:8} passed {totals[name]}/{totals['graded']} ({100 * totals[name] / totals['graded']:.1f}%)")
    print(f"Only current passed: {totals['only_current']} | only simple passed: {totals['only_simple']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--questions", required=True, type=Path)
    r.add_argument("--delay", type=float, default=4.0, help="seconds between questions, for rate limits")
    r.add_argument("--seed", type=int)
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "score":
        return score(args.out)
    return run(args.questions, args.out, args.delay, args.seed)


if __name__ == "__main__":
    sys.exit(main())
