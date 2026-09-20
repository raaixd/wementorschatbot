"""
Human Preference Evaluation Tool for comparing chatbot responses side-by-side.
Allows developers/evaluators to score two models or prompt variants against
a 7-criteria rubric without complex external tooling or RLHF.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RUBRIC_CRITERIA = [
    "correctness",       # Factual accuracy and alignment with WeMentors KB
    "relevance",         # Directly addresses the user's specific query
    "naturalness",       # Warm, conversational, human-grade tone
    "conciseness",       # Direct, avoiding boilerplate or repetitive disclaimers
    "grounding",         # Absence of unverified claims or invented facts
    "context_awareness", # Appropriate resolution of pronouns, history, and active program
    "helpfulness",       # Provides clear, actionable next steps (e.g. Book Free Demo)
]

EVAL_RECORDS_FILE = Path(__file__).resolve().parent.parent / "reports" / "human_eval_records.json"


def load_records() -> List[Dict[str, Any]]:
    if EVAL_RECORDS_FILE.exists():
        try:
            with open(EVAL_RECORDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_record(record: Dict[str, Any]) -> None:
    records = load_records()
    records.append(record)
    EVAL_RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Record saved. Total human evaluations: {len(records)}")


def record_evaluation(
    question: str,
    response_a: str,
    response_b: str,
    criterion: str,
    human_preference: str,  # 'A', 'B', or 'Tie'
    notes: str = "",
    model_a: str = "variant_a",
    model_b: str = "variant_b",
) -> Dict[str, Any]:
    """Programmatically record a human preference comparison entry."""
    if criterion not in RUBRIC_CRITERIA:
        raise ValueError(f"Invalid criterion '{criterion}'. Must be one of: {RUBRIC_CRITERIA}")
    if human_preference not in ("A", "B", "Tie"):
        raise ValueError("human_preference must be 'A', 'B', or 'Tie'")

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "response_a": response_a,
        "response_b": response_b,
        "model_a": model_a,
        "model_b": model_b,
        "criterion": criterion,
        "human_preference": human_preference,
        "notes": notes,
    }
    save_record(record)
    return record


def summarize_human_eval() -> Dict[str, Any]:
    records = load_records()
    if not records:
        return {"total_evaluations": 0}

    prefs = {"A": 0, "B": 0, "Tie": 0}
    by_criterion: Dict[str, Dict[str, int]] = {}
    for r in records:
        p = r.get("human_preference", "Tie")
        prefs[p] = prefs.get(p, 0) + 1
        c = r.get("criterion", "overall")
        if c not in by_criterion:
            by_criterion[c] = {"A": 0, "B": 0, "Tie": 0}
        by_criterion[c][p] = by_criterion[c].get(p, 0) + 1

    return {
        "total_evaluations": len(records),
        "overall_preferences": prefs,
        "breakdown_by_criterion": by_criterion,
    }


if __name__ == "__main__":
    # Self-test / seeding sample records for demo
    print("WeMentors Human Preference Evaluation Framework initialized.")
    print("Rubric criteria available:", RUBRIC_CRITERIA)
    summary = summarize_human_eval()
    print("Current records summary:", summary)
