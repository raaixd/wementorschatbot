"""
Evaluation metrics calculator for the WeMentors Chatbot RAG Benchmark.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_FALSE_COMPLETION_PATTERNS = [
    re.compile(r"\b(i'?ve|i have|we have|we'?ve)\s+(booked|scheduled|registered|enrolled|confirmed|submitted)\b", re.I),
    re.compile(r"\b(your demo|your class|the demo)\s+(?:class\s+)?(is|has been)\s+(booked|scheduled|confirmed)\b", re.I),
    re.compile(r"\beverything is confirmed\b", re.I),
    re.compile(r"\bsuccessfully (booked|registered|enrolled|submitted)\b", re.I),
]

_DISCLAIMER_PATTERNS = [
    re.compile(r"\b(team|contact|wementors\.co|book free demo|unverified|confirm)\b", re.I),
]


def evaluate_single_turn(
    reply: str,
    intent: str,
    matched_entry_ids: List[str],
    case: Dict[str, Any],
    latency_ms: float = 0.0,
    provider: str = "none",
    fallback_used: bool = False,
) -> Dict[str, Any]:
    """Evaluate an individual turn against the case specification."""
    reply_clean = reply.strip()
    reply_lower = reply_clean.lower()

    # 1. Demo Safety check (must NEVER claim booking was made)
    is_demo_safe = True
    for pat in _FALSE_COMPLETION_PATTERNS:
        if pat.search(reply_clean):
            is_demo_safe = False
            break

    # 2. Forbidden Keywords / Hallucination check
    hallucination_detected = False
    hallucinated_terms = []
    for forbidden in case.get("forbidden_keywords", []):
        if forbidden.lower() in reply_lower:
            hallucination_detected = True
            hallucinated_terms.append(forbidden)

    # 3. Groundedness / Unknown-question Handling
    requires_fallback = case.get("requires_honest_fallback", False)
    fallback_handled_safely = True
    if requires_fallback:
        has_safe_disclaimer = any(p.search(reply_clean) for p in _DISCLAIMER_PATTERNS)
        if not has_safe_disclaimer:
            fallback_handled_safely = False

    # 4. Expected Keywords check (Answer Correctness)
    expected_keywords = case.get("expected_keywords", [])
    keywords_found = []
    for kw in expected_keywords:
        if kw.lower() in reply_lower:
            keywords_found.append(kw)

    keyword_coverage = (len(keywords_found) / len(expected_keywords)) if expected_keywords else 1.0
    answer_correct = (keyword_coverage >= 0.5) and not hallucination_detected

    # 5. Intent Accuracy
    expected_intent = case.get("expected_intent")
    intent_correct = True
    if expected_intent:
        if isinstance(expected_intent, list):
            intent_correct = intent in expected_intent
        else:
            # Allow compatible/derived intents
            intent_correct = (intent == expected_intent) or (expected_intent in intent) or (intent in expected_intent)

    # 6. Context Resolution
    context_resolved = True
    if case.get("category") in ("category_b_follow_up", "category_c_context_switching", "category_i_program_semantics"):
        context_resolved = answer_correct and not hallucination_detected

    return {
        "case_id": case["id"],
        "category": case["category"],
        "intent": intent,
        "matched_entry_ids": matched_entry_ids,
        "answer_correct": answer_correct,
        "keyword_coverage": keyword_coverage,
        "keywords_found": keywords_found,
        "demo_safe": is_demo_safe,
        "hallucination_detected": hallucination_detected,
        "hallucinated_terms": hallucinated_terms,
        "fallback_handled_safely": fallback_handled_safely,
        "intent_correct": intent_correct,
        "context_resolved": context_resolved,
        "latency_ms": latency_ms,
        "provider": provider,
        "fallback_used": fallback_used,
    }


def aggregate_benchmark_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute benchmark summary statistics across all evaluated test cases."""
    total = len(results)
    if total == 0:
        return {}

    correct_count = sum(1 for r in results if r["answer_correct"])
    demo_safe_count = sum(1 for r in results if r["demo_safe"])
    hallucination_count = sum(1 for r in results if r["hallucination_detected"])
    fallback_safe_count = sum(1 for r in results if r["fallback_handled_safely"])
    intent_correct_count = sum(1 for r in results if r["intent_correct"])
    context_resolved_count = sum(1 for r in results if r["context_resolved"])
    fallback_used_count = sum(1 for r in results if r["fallback_used"])

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Per category aggregation
    categories: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {
                "total": 0,
                "correct": 0,
                "hallucinations": 0,
                "demo_safe": 0,
            }
        categories[cat]["total"] += 1
        if r["answer_correct"]:
            categories[cat]["correct"] += 1
        if r["hallucination_detected"]:
            categories[cat]["hallucinations"] += 1
        if r["demo_safe"]:
            categories[cat]["demo_safe"] += 1

    return {
        "total_cases": total,
        "answer_correctness_pct": round((correct_count / total) * 100, 2),
        "demo_safety_pct": round((demo_safe_count / total) * 100, 2),
        "hallucination_rate_pct": round((hallucination_count / total) * 100, 2),
        "groundedness_pct": round(((total - hallucination_count) / total) * 100, 2),
        "intent_accuracy_pct": round((intent_correct_count / total) * 100, 2),
        "context_resolution_pct": round((context_resolved_count / total) * 100, 2),
        "unknown_handling_pct": round((fallback_safe_count / total) * 100, 2),
        "provider_fallback_rate_pct": round((fallback_used_count / total) * 100, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "error_rate_pct": 0.0,
        "categories": categories,
    }
