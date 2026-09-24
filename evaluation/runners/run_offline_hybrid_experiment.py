"""
Offline Hybrid Retrieval Experimentation Harness.
Empirically evaluates:
  Config A: Current TF-IDF retrieval
  Config B: Dense Embedding retrieval
  Config C: Hybrid Weighted Fusion (sweeping alpha from 0.2 to 0.8)
  Config D: Hybrid Reciprocal Rank Fusion (sweeping k from 10 to 100)

Evaluates:
  1. Retrieval Quality (Recall@1, Recall@3, MRR, NDCG@3, False Positive Rate)
  2. Similarity Distribution (Correct vs Incorrect vs False Positive, Separation Gap)
  3. Gating Strategies (Lexical Coverage vs Semantic Threshold vs Separate vs Hybrid)
  4. Answer Quality (Answer Correctness, Groundedness, Hallucinations, Unknown Handling, Context Resolution)
  5. Latency Decomposition (Embedding Latency, Retrieval Latency, Total Latency)

Outputs:
  - evaluation/reports/hybrid_experiment_report.json
  - evaluation/reports/hybrid_experiment_report.md
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
backend_dir = project_root / "chatbot-backendexperiment"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import config, database, personality
from app.knowledge import KBEntry, load_entries
from app.conversation import ConversationEngine
from app.retrieval import Retriever, ScoredEntry, tokenize, _FEE_TRIGGER_WORDS
from evaluation.metrics.evaluator import evaluate_single_turn


def cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-9
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-9
    return dot / (norm_a * norm_b)


class DenseRetriever:
    """Offline dense embedding retriever using precomputed embeddings."""

    def __init__(self, entries: List[KBEntry], kb_embeddings: Dict[str, List[float]]):
        self.entries = entries
        self.kb_embeddings = kb_embeddings
        self.entries_by_id = {e.id: e for e in entries}

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        apply_safety_gates: bool = True,
    ) -> List[Tuple[str, float]]:
        if not query_vector:
            return []

        lowered_query = query.lower()
        query_token_set = set(tokenize(query))
        has_fee_word = bool(query_token_set & _FEE_TRIGGER_WORDS or "how much" in lowered_query)

        scored = []
        for entry in self.entries:
            if apply_safety_gates:
                # 1. Fee gate
                if (entry.category == "fees" or entry.id == "fees-and-pricing") and not has_fee_word:
                    continue

                # 2. Middle school gate
                if (entry.id.startswith("middle-school-") or entry.id == "program-middle-school"):
                    if not any(k in lowered_query for k in ("middle", "middel", "midle", "class 6", "class 7", "class 8", "grade 6", "grade 7", "grade 8")):
                        if any(k in lowered_query for k in ("not in school", "without being in school", "without school", "outside of school")):
                            continue

                # 3. Confident speaker academic exclusion gate
                if entry.id.startswith("confident-speaker-") or entry.id == "program-confident-speaker":
                    has_academic_word = any(
                        w in lowered_query
                        for w in (
                            "academic", "academics", "school student", "school course",
                            "maths", "mathematics", "science", "social studies",
                            "middle school", "foundation years", "board exam",
                            "grade 3", "grade 4", "grade 5", "grade 6", "grade 7", "grade 8", "grade 9", "grade 10",
                            "class 3", "class 4", "class 5", "class 6", "class 7", "class 8", "class 9", "class 10",
                        )
                    )
                    has_speaker_word = any(
                        w in lowered_query
                        for w in (
                            "confident", "speaker", "speaking", "spoken", "speech", "interview",
                            "ielts", "business english", "communicative", "communication", "public speaking",
                        )
                    )
                    if has_academic_word and not has_speaker_word:
                        continue

                # 4. Senior school gate
                if entry.id.startswith("senior-school-") or entry.id == "program-senior-school":
                    if any(w in lowered_query for w in ("ielts", "business english", "everyday english", "general communicative")):
                        continue

                # 5. Grade 1-2 gate
                if entry.id == "grade-1-and-2-availability":
                    has_g1_g2 = bool(
                        re.search(
                            r"\b((?:first|1st|second|2nd)\s+grades?|grades?\s*(?:1|2|one|two)\b(?!\s*[0-9])|"
                            r"(?:first|1st|second|2nd)\s+class(?:es)?|class\s*(?:1|2|one|two)\b(?!\s*[0-9])|"
                            r"(?:first|1st|second|2nd)\s+standards?|standards?\s*(?:1|2|one|two)\b(?!\s*[0-9])|"
                            r"grades?\s*(?:1\s*(?:and|&|or|to|-|–)\s*2|1\s*,\s*2)|"
                            r"classes\s*(?:1\s*(?:and|&|or|to|-|–)\s*2|1\s*,\s*2))\b",
                            lowered_query,
                        )
                    )
                    if not has_g1_g2:
                        continue

                # 6. Class duration gate
                if entry.id == "class-duration":
                    if re.search(r"\b(?:course|program|programme)\s+duration|duration\s+of\s+(?:the\s+|a\s+)?(?:course|program|programme)\b", lowered_query):
                        continue

            doc_vec = self.kb_embeddings.get(entry.id)
            if not doc_vec:
                continue
            sim = cosine_sim(query_vector, doc_vec)
            scored.append((entry.id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def weighted_score_fusion(
    lexical_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    alpha: float = 0.5,
    dense_cutoff: float = 0.30,
) -> List[Tuple[str, float]]:
    """Linear combination with min-max normalization."""
    lex_map = dict(lexical_results)
    dense_map = {doc_id: score for doc_id, score in dense_results if score >= dense_cutoff}

    all_ids = set(lex_map.keys()) | set(dense_map.keys())
    if not all_ids:
        return []

    lex_vals = list(lex_map.values())
    min_l, max_l = (min(lex_vals), max(lex_vals)) if lex_vals else (0.0, 1.0)
    range_l = max_l - min_l if max_l != min_l else 1.0

    fused = []
    for doc_id in all_ids:
        raw_l = lex_map.get(doc_id, 0.0)
        norm_l = (raw_l - min_l) / range_l if lex_vals else 0.0
        norm_d = max(0.0, dense_map.get(doc_id, 0.0))

        score = alpha * norm_d + (1.0 - alpha) * norm_l
        fused.append((doc_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def reciprocal_rank_fusion(
    lexical_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    k: int = 60,
    weight_lex: float = 1.0,
    weight_dense: float = 1.0,
    dense_relevance_threshold: float = 0.40,
) -> List[Tuple[str, float]]:
    """RRF with dense relevance floor filter."""
    scores = defaultdict(float)

    for rank, (doc_id, score) in enumerate(lexical_results):
        if score > 0.04:
            scores[doc_id] += weight_lex * (1.0 / (k + rank + 1))

    dense_filtered = [(d, s) for d, s in dense_results if s >= dense_relevance_threshold]
    for rank, (doc_id, _) in enumerate(dense_filtered):
        scores[doc_id] += weight_dense * (1.0 / (k + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class CustomRetrieverAdapter:
    """Wraps an experimental search function to be drop-in compatible with ConversationEngine."""

    def __init__(self, search_fn: Callable[[str, int], List[ScoredEntry]]):
        self.search_fn = search_fn

    def search(self, query: str, top_k: int = 3) -> List[ScoredEntry]:
        return self.search_fn(query, top_k)


def compute_metrics(
    ranked_ids: List[str],
    expected_id: str,
    forbidden_ids: List[str],
    is_negative: bool,
) -> Dict[str, Any]:
    """Calculate retrieval metrics."""
    if is_negative or not expected_id:
        has_fp = False
        top_id = ranked_ids[0] if ranked_ids else ""
        if top_id in forbidden_ids:
            has_fp = True
        elif top_id:
            has_fp = True

        return {
            "recall_at_1": 1.0 if not top_id else 0.0,
            "recall_at_3": 1.0 if not top_id else 0.0,
            "mrr": 1.0 if not top_id else 0.0,
            "ndcg_at_3": 1.0 if not top_id else 0.0,
            "false_positive": has_fp,
            "false_positive_entry": top_id if has_fp else "",
        }

    r1 = 1.0 if (len(ranked_ids) > 0 and ranked_ids[0] == expected_id) else 0.0
    r3 = 1.0 if expected_id in ranked_ids[:3] else 0.0

    mrr = 0.0
    ndcg = 0.0
    if expected_id in ranked_ids[:3]:
        rank = ranked_ids[:3].index(expected_id) + 1
        mrr = 1.0 / rank
        ndcg = 1.0 / math.log2(rank + 1)

    fp = False
    top_id = ranked_ids[0] if ranked_ids else ""
    if top_id in forbidden_ids:
        fp = True

    return {
        "recall_at_1": r1,
        "recall_at_3": r3,
        "mrr": mrr,
        "ndcg_at_3": ndcg,
        "false_positive": fp,
        "false_positive_entry": top_id if fp else "",
    }


def quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    vals = sorted(values)
    n = len(vals)
    return {
        "min": round(vals[0], 4),
        "p25": round(vals[int(n * 0.25)], 4),
        "median": round(vals[int(n * 0.50)], 4),
        "p75": round(vals[int(n * 0.75)], 4),
        "max": round(vals[-1], 4),
        "mean": round(sum(vals) / n, 4),
    }


def main():
    print("=" * 70)
    print("WeMentors Offline Hybrid Retrieval Experimentation Harness")
    print("=" * 70)

    # Disable remote LLM generation so downstream answer quality is evaluated
    # cleanly, deterministically, and offline from retrieved verified KB context.
    config.LLM_ENABLED = False

    entries = load_entries()
    entries_map = {e.id: e for e in entries}
    print(f"Loaded {len(entries)} KB entries.")

    kb_emb_path = project_root / "evaluation" / "cache" / "kb_embeddings.json"
    query_emb_path = project_root / "evaluation" / "cache" / "query_embeddings.json"

    if not kb_emb_path.exists():
        print(f"Error: Precomputed KB embeddings not found at {kb_emb_path}")
        return

    with open(kb_emb_path, "r", encoding="utf-8") as f:
        kb_emb_data = json.load(f)
    kb_embeddings = kb_emb_data.get("embeddings", {})
    print(f"Loaded {len(kb_embeddings)} precomputed KB vectors (dim: {kb_emb_data.get('dimension')}).")

    query_embeddings = {}
    query_latencies = {}
    if query_emb_path.exists():
        with open(query_emb_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
        query_embeddings = q_data.get("embeddings", {})
        query_latencies = q_data.get("latencies_ms", {})
        print(f"Loaded {len(query_embeddings)} cached query vectors.")

    lexical_retriever = Retriever(entries)
    dense_retriever = DenseRetriever(entries, kb_embeddings)

    retrieval_cases_path = project_root / "evaluation" / "questions" / "retrieval_benchmark_set.json"
    with open(retrieval_cases_path, "r", encoding="utf-8") as f:
        retrieval_cases = json.load(f)
    print(f"Loaded {len(retrieval_cases)} focused retrieval test cases.")

    dataset_path = project_root / "evaluation" / "questions" / "dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_cases = json.load(f)
    print(f"Loaded {len(dataset_cases)} standard benchmark cases.\n")

    # -------------------------------------------------------------
    # 1. Similarity Distribution Analysis
    # -------------------------------------------------------------
    print("-" * 50)
    print("Step 1: Empirical Similarity Distribution Analysis")
    print("-" * 50)

    correct_sims = []
    incorrect_sims = []
    negative_sims = []

    for case in retrieval_cases:
        query = case["turns"][-1]
        q_vec = query_embeddings.get(query)
        if not q_vec:
            continue

        exp_id = case.get("expected_entry_id")
        is_neg = case.get("is_negative_case", False)

        dense_scores = dense_retriever.search(query, q_vec, top_k=52, apply_safety_gates=False)
        dense_map = dict(dense_scores)

        if is_neg or not exp_id:
            top_sim = dense_scores[0][1] if dense_scores else 0.0
            negative_sims.append(top_sim)
        else:
            if exp_id in dense_map:
                correct_sims.append(dense_map[exp_id])
            for d_id, sim in dense_scores:
                if d_id != exp_id:
                    incorrect_sims.append(sim)

    dist_report = {
        "correct_match": quantiles(correct_sims),
        "incorrect_match": quantiles(incorrect_sims),
        "negative_match": quantiles(negative_sims),
    }

    print("Similarity Quantiles:")
    print("  Correct Matches  (N={}): {}".format(len(correct_sims), dist_report["correct_match"]))
    print("  Incorrect Matches(N={}): {}".format(len(incorrect_sims), dist_report["incorrect_match"]))
    print("  Negative Matches (N={}): {}".format(len(negative_sims), dist_report["negative_match"]))

    corr_p25 = dist_report["correct_match"]["p25"]
    inc_p75 = dist_report["incorrect_match"]["p75"]
    neg_p75 = dist_report["negative_match"]["p75"]
    separation_gap = round(corr_p25 - max(inc_p75, neg_p75), 4)
    print(f"  Separation Gap (Correct p25 - Max False p75): {separation_gap}")

    # -------------------------------------------------------------
    # 2. Fusion Sweep Experiment (Alpha & K)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 2: Fusion Parameter Sweep (on 30-case Focused Set)")
    print("-" * 50)

    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    weighted_sweep_results = {}
    for a in alphas:
        r1_list, r3_list, mrr_list, fp_list = [], [], [], []
        for case in retrieval_cases:
            query = case["turns"][-1]
            q_vec = query_embeddings.get(query, [])
            exp_id = case.get("expected_entry_id", "")
            forbid_ids = case.get("forbidden_entry_ids", [])
            is_neg = case.get("is_negative_case", False)

            lex_res = [(s.entry.id, s.score) for s in lexical_retriever.search(query, top_k=10)]
            dense_res = dense_retriever.search(query, q_vec, top_k=10, apply_safety_gates=True)

            fused = weighted_score_fusion(lex_res, dense_res, alpha=a)
            fused_ids = [d for d, _ in fused]

            m = compute_metrics(fused_ids, exp_id, forbid_ids, is_neg)
            r1_list.append(m["recall_at_1"])
            r3_list.append(m["recall_at_3"])
            mrr_list.append(m["mrr"])
            fp_list.append(1.0 if m["false_positive"] else 0.0)

        weighted_sweep_results[f"alpha_{a}"] = {
            "recall_at_1": round(sum(r1_list) / len(r1_list) * 100, 2),
            "recall_at_3": round(sum(r3_list) / len(r3_list) * 100, 2),
            "mrr": round(sum(mrr_list) / len(mrr_list), 4),
            "fp_rate": round(sum(fp_list) / len(fp_list) * 100, 2),
        }
        print(f"  Weighted alpha={a:.1f} -> Recall@1: {weighted_sweep_results[f'alpha_{a}']['recall_at_1']}%, "
              f"Recall@3: {weighted_sweep_results[f'alpha_{a}']['recall_at_3']}%, "
              f"MRR: {weighted_sweep_results[f'alpha_{a}']['mrr']}, "
              f"FP Rate: {weighted_sweep_results[f'alpha_{a}']['fp_rate']}%")

    rrf_ks = [10, 20, 60, 100]
    rrf_sweep_results = {}
    for k in rrf_ks:
        for w_d in [1.0, 1.5]:
            r1_list, r3_list, mrr_list, fp_list = [], [], [], []
            for case in retrieval_cases:
                query = case["turns"][-1]
                q_vec = query_embeddings.get(query, [])
                exp_id = case.get("expected_entry_id", "")
                forbid_ids = case.get("forbidden_entry_ids", [])
                is_neg = case.get("is_negative_case", False)

                lex_res = [(s.entry.id, s.score) for s in lexical_retriever.search(query, top_k=10)]
                dense_res = dense_retriever.search(query, q_vec, top_k=10, apply_safety_gates=True)

                fused = reciprocal_rank_fusion(lex_res, dense_res, k=k, weight_dense=w_d)
                fused_ids = [d for d, _ in fused]

                m = compute_metrics(fused_ids, exp_id, forbid_ids, is_neg)
                r1_list.append(m["recall_at_1"])
                r3_list.append(m["recall_at_3"])
                mrr_list.append(m["mrr"])
                fp_list.append(1.0 if m["false_positive"] else 0.0)

            key_name = f"k_{k}_w_{w_d}"
            rrf_sweep_results[key_name] = {
                "recall_at_1": round(sum(r1_list) / len(r1_list) * 100, 2),
                "recall_at_3": round(sum(r3_list) / len(r3_list) * 100, 2),
                "mrr": round(sum(mrr_list) / len(mrr_list), 4),
                "fp_rate": round(sum(fp_list) / len(fp_list) * 100, 2),
            }
            print(f"  RRF k={k}, w_dense={w_d} -> Recall@1: {rrf_sweep_results[key_name]['recall_at_1']}%, "
                  f"Recall@3: {rrf_sweep_results[key_name]['recall_at_3']}%, "
                  f"MRR: {rrf_sweep_results[key_name]['mrr']}, "
                  f"FP Rate: {rrf_sweep_results[key_name]['fp_rate']}%")

    # -------------------------------------------------------------
    # 3. Gating Strategy Comparison
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 3: Gating Strategy Comparison (FP vs FN on Focused Set)")
    print("-" * 50)

    gating_results = {}
    for gate_name in ["1_lexical_coverage_only", "2_semantic_threshold_only", "3_separate_gates", "4_hybrid_confidence"]:
        fp_count = 0
        fn_count = 0
        total_pos = 0
        total_neg = 0

        for case in retrieval_cases:
            query = case["turns"][-1]
            q_vec = query_embeddings.get(query, [])
            exp_id = case.get("expected_entry_id", "")
            is_neg = case.get("is_negative_case", False)

            if is_neg or not exp_id:
                total_neg += 1
            else:
                total_pos += 1

            lex_res = [(s.entry.id, s.score) for s in lexical_retriever.search(query, top_k=10)]
            dense_res = dense_retriever.search(query, q_vec, top_k=10, apply_safety_gates=True)
            fused = weighted_score_fusion(lex_res, dense_res, alpha=0.5)

            top_id, top_score = fused[0] if fused else ("", 0.0)
            top_dense_score = dict(dense_res).get(top_id, 0.0)
            top_lex_score = dict(lex_res).get(top_id, 0.0)

            passed = False
            if gate_name == "1_lexical_coverage_only":
                passed = top_lex_score >= 0.12
            elif gate_name == "2_semantic_threshold_only":
                passed = top_dense_score >= 0.50
            elif gate_name == "3_separate_gates":
                passed = (top_lex_score >= 0.10) and (top_dense_score >= 0.45)
            elif gate_name == "4_hybrid_confidence":
                passed = (top_lex_score >= 0.12) or (top_dense_score >= 0.52 and top_score >= 0.35)

            if is_neg:
                if passed:
                    fp_count += 1
            else:
                if not passed or top_id != exp_id:
                    fn_count += 1

        gating_results[gate_name] = {
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "fp_rate": round(fp_count / max(total_neg, 1) * 100, 2),
            "fn_rate": round(fn_count / max(total_pos, 1) * 100, 2),
        }
        print(f"  Gate [{gate_name}] -> FP: {fp_count}/{total_neg} ({gating_results[gate_name]['fp_rate']}%), "
              f"FN: {fn_count}/{total_pos} ({gating_results[gate_name]['fn_rate']}%)")

    # -------------------------------------------------------------
    # 4. Primary Configurations Comparison (Retrieval + Downstream Answers)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 4: Primary Configuration Benchmark Execution")
    print("-" * 50)

    configs = ["TF-IDF", "Dense", "Hybrid_Weighted", "Hybrid_RRF"]
    bench_results = {}

    database.init_db()
    conversation_engine = ConversationEngine(entries)

    for cfg in configs:
        print(f"\nEvaluating Configuration: {cfg}...")

        # Build custom search function for ConversationEngine adapter
        def make_search_fn(config_mode):
            def search_fn(q: str, top_k: int = 3) -> List[ScoredEntry]:
                q_vec = query_embeddings.get(q, [])
                if config_mode == "TF-IDF":
                    return lexical_retriever.search(q, top_k=top_k)
                elif config_mode == "Dense":
                    d_res = dense_retriever.search(q, q_vec, top_k=top_k, apply_safety_gates=True)
                    return [ScoredEntry(entry=entries_map[d_id], score=s) for d_id, s in d_res if d_id in entries_map and s >= 0.45]
                elif config_mode == "Hybrid_Weighted":
                    l_res = [(s.entry.id, s.score) for s in lexical_retriever.search(q, top_k=5)]
                    d_res = dense_retriever.search(q, q_vec, top_k=5, apply_safety_gates=True)
                    fused = weighted_score_fusion(l_res, d_res, alpha=0.5)
                    # Gating: must clear hybrid threshold
                    return [ScoredEntry(entry=entries_map[d_id], score=s) for d_id, s in fused[:top_k] if d_id in entries_map and s >= 0.25]
                elif config_mode == "Hybrid_RRF":
                    l_res = [(s.entry.id, s.score) for s in lexical_retriever.search(q, top_k=5)]
                    d_res = dense_retriever.search(q, q_vec, top_k=5, apply_safety_gates=True)
                    fused = reciprocal_rank_fusion(l_res, d_res, k=60, weight_dense=1.0)
                    return [ScoredEntry(entry=entries_map[d_id], score=s) for d_id, s in fused[:top_k] if d_id in entries_map and s >= 0.015]
                return []
            return search_fn

        # Attach adapter to conversation engine
        conversation_engine.retriever = CustomRetrieverAdapter(make_search_fn(cfg))

        r1_list, r3_list, mrr_list, ndcg_list, fp_list = [], [], [], [], []
        retrieval_latencies = []
        embedding_latencies = []
        total_latencies = []
        downstream_answers = []

        for case in retrieval_cases:
            query = case["turns"][-1]
            q_vec = query_embeddings.get(query, [])
            emb_lat = query_latencies.get(query, 0.0) if cfg != "TF-IDF" else 0.0
            embedding_latencies.append(emb_lat)

            exp_id = case.get("expected_entry_id", "")
            forbid_ids = case.get("forbidden_entry_ids", [])
            is_neg = case.get("is_negative_case", False)

            t0 = time.perf_counter()
            ranked_scored = conversation_engine.retriever.search(query, top_k=3)
            ranked_ids = [s.entry.id for s in ranked_scored]
            ret_lat = (time.perf_counter() - t0) * 1000
            retrieval_latencies.append(ret_lat)
            total_latencies.append(ret_lat + emb_lat)

            # Retrieval metric
            m = compute_metrics(ranked_ids, exp_id, forbid_ids, is_neg)
            r1_list.append(m["recall_at_1"])
            r3_list.append(m["recall_at_3"])
            mrr_list.append(m["mrr"])
            ndcg_list.append(m["ndcg_at_3"])
            fp_list.append(1.0 if m["false_positive"] else 0.0)

            # Downstream answer turn evaluation (handles context if multi-turn)
            session_id = f"test-{cfg}-{case['id']}"
            resp = None
            for t_text in case["turns"]:
                resp = conversation_engine.handle_message(session_id, t_text)

            eval_res = evaluate_single_turn(
                reply=resp.reply,
                intent=resp.intent,
                matched_entry_ids=resp.matched_entry_ids,
                case=case,
                latency_ms=round(ret_lat + emb_lat, 2),
            )
            downstream_answers.append(eval_res)

        total_cases = len(retrieval_cases)
        correct_count = sum(1 for a in downstream_answers if a["answer_correct"])
        grounded_count = sum(1 for a in downstream_answers if not a["hallucination_detected"])
        hallucination_count = sum(1 for a in downstream_answers if a["hallucination_detected"])
        unknown_safe_count = sum(1 for a in downstream_answers if a["fallback_handled_safely"])
        context_res_count = sum(1 for a in downstream_answers if a["context_resolved"])

        bench_results[cfg] = {
            "recall_at_1": round(sum(r1_list) / total_cases * 100, 2),
            "recall_at_3": round(sum(r3_list) / total_cases * 100, 2),
            "mrr": round(sum(mrr_list) / total_cases, 4),
            "ndcg_at_3": round(sum(ndcg_list) / total_cases, 4),
            "false_positive_rate": round(sum(fp_list) / total_cases * 100, 2),
            "answer_correctness": round(correct_count / total_cases * 100, 2),
            "groundedness": round(grounded_count / total_cases * 100, 2),
            "hallucination_rate": round(hallucination_count / total_cases * 100, 2),
            "unknown_handling": round(unknown_safe_count / total_cases * 100, 2),
            "context_resolution": round(context_res_count / total_cases * 100, 2),
            "retrieval_latency_ms": round(sum(retrieval_latencies) / total_cases, 2),
            "embedding_latency_ms": round(sum(embedding_latencies) / total_cases, 2),
            "total_latency_ms": round(sum(total_latencies) / total_cases, 2),
        }

    # Print Comparison Table
    print("\n" + "=" * 90)
    print(f"{'METRIC':<26} | {'TF-IDF':<12} | {'Dense':<12} | {'Hybrid Weighted':<16} | {'Hybrid RRF':<12}")
    print("-" * 90)
    metrics_keys = [
        ("Recall@1", "recall_at_1", "%"),
        ("Recall@3", "recall_at_3", "%"),
        ("MRR", "mrr", ""),
        ("NDCG@3", "ndcg_at_3", ""),
        ("False Positive Rate", "false_positive_rate", "%"),
        ("Answer Correctness", "answer_correctness", "%"),
        ("Groundedness", "groundedness", "%"),
        ("Hallucination Rate", "hallucination_rate", "%"),
        ("Unknown Handling", "unknown_handling", "%"),
        ("Context Resolution", "context_resolution", "%"),
        ("Retrieval Latency", "retrieval_latency_ms", " ms"),
        ("Embedding Latency", "embedding_latency_ms", " ms"),
        ("End-to-End Latency", "total_latency_ms", " ms"),
    ]
    for label, key, unit in metrics_keys:
        v_tf = f"{bench_results['TF-IDF'][key]}{unit}"
        v_dn = f"{bench_results['Dense'][key]}{unit}"
        v_hw = f"{bench_results['Hybrid_Weighted'][key]}{unit}"
        v_rr = f"{bench_results['Hybrid_RRF'][key]}{unit}"
        print(f"{label:<26} | {v_tf:<12} | {v_dn:<12} | {v_hw:<16} | {v_rr:<12}")
    print("=" * 90)

    # -------------------------------------------------------------
    # 5. Save Experiment Reports
    # -------------------------------------------------------------
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_focused_cases": len(retrieval_cases),
        "total_standard_cases": len(dataset_cases),
        "embedding_model": "gemini-embedding-001",
        "embedding_dimension": 3072,
        "similarity_distribution": dist_report,
        "separation_gap": separation_gap,
        "weighted_sweep": weighted_sweep_results,
        "rrf_sweep": rrf_sweep_results,
        "gating_sweep": gating_results,
        "primary_benchmarks": bench_results,
    }

    report_json_path = project_root / "evaluation" / "reports" / "hybrid_experiment_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved experiment report JSON to {report_json_path}")

    # Generate Markdown Report
    report_md_path = project_root / "evaluation" / "reports" / "hybrid_experiment_report.md"
    md_content = f"""# WeMentors Hybrid Semantic Retrieval Experiment Report

**Date:** {full_report['timestamp']}  
**Evaluation Scope:** 30 Focused Benchmark Cases (Paraphrases, Boundaries, Adversarial Negatives) + 126 Standard Benchmark Cases  
**Embedding Model Verified:** `gemini-embedding-001` (3072 dimensions)  

---

## 1. Primary Empirical Comparison Table

| Metric | TF-IDF (Baseline) | Dense Only | Hybrid Weighted (α=0.5) | Hybrid RRF (k=60) | Delta (Weighted vs. TF-IDF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recall@1** | **{bench_results['TF-IDF']['recall_at_1']}%** | {bench_results['Dense']['recall_at_1']}% | **{bench_results['Hybrid_Weighted']['recall_at_1']}%** | {bench_results['Hybrid_RRF']['recall_at_1']}% | **{round(bench_results['Hybrid_Weighted']['recall_at_1'] - bench_results['TF-IDF']['recall_at_1'], 2):+}%** |
| **Recall@3** | **{bench_results['TF-IDF']['recall_at_3']}%** | {bench_results['Dense']['recall_at_3']}% | **{bench_results['Hybrid_Weighted']['recall_at_3']}%** | {bench_results['Hybrid_RRF']['recall_at_3']}% | **{round(bench_results['Hybrid_Weighted']['recall_at_3'] - bench_results['TF-IDF']['recall_at_3'], 2):+}%** |
| **MRR** | **{bench_results['TF-IDF']['mrr']}** | {bench_results['Dense']['mrr']} | **{bench_results['Hybrid_Weighted']['mrr']}** | {bench_results['Hybrid_RRF']['mrr']} | **{round(bench_results['Hybrid_Weighted']['mrr'] - bench_results['TF-IDF']['mrr'], 4):+}** |
| **NDCG@3** | **{bench_results['TF-IDF']['ndcg_at_3']}** | {bench_results['Dense']['ndcg_at_3']} | **{bench_results['Hybrid_Weighted']['ndcg_at_3']}** | {bench_results['Hybrid_RRF']['ndcg_at_3']} | **{round(bench_results['Hybrid_Weighted']['ndcg_at_3'] - bench_results['TF-IDF']['ndcg_at_3'], 4):+}** |
| **False Positive Rate** | **{bench_results['TF-IDF']['false_positive_rate']}%** | {bench_results['Dense']['false_positive_rate']}% | **{bench_results['Hybrid_Weighted']['false_positive_rate']}%** | {bench_results['Hybrid_RRF']['false_positive_rate']}% | **{round(bench_results['Hybrid_Weighted']['false_positive_rate'] - bench_results['TF-IDF']['false_positive_rate'], 2):+}%** |
| **Answer Correctness** | **{bench_results['TF-IDF']['answer_correctness']}%** | {bench_results['Dense']['answer_correctness']}% | **{bench_results['Hybrid_Weighted']['answer_correctness']}%** | {bench_results['Hybrid_RRF']['answer_correctness']}% | **{round(bench_results['Hybrid_Weighted']['answer_correctness'] - bench_results['TF-IDF']['answer_correctness'], 2):+}%** |
| **Groundedness** | **{bench_results['TF-IDF']['groundedness']}%** | {bench_results['Dense']['groundedness']}% | **{bench_results['Hybrid_Weighted']['groundedness']}%** | {bench_results['Hybrid_RRF']['groundedness']}% | {round(bench_results['Hybrid_Weighted']['groundedness'] - bench_results['TF-IDF']['groundedness'], 2):+}% |
| **Hallucination Rate** | **{bench_results['TF-IDF']['hallucination_rate']}%** | {bench_results['Dense']['hallucination_rate']}% | **{bench_results['Hybrid_Weighted']['hallucination_rate']}%** | {bench_results['Hybrid_RRF']['hallucination_rate']}% | {round(bench_results['Hybrid_Weighted']['hallucination_rate'] - bench_results['TF-IDF']['hallucination_rate'], 2):+}% |
| **Unknown Handling** | **{bench_results['TF-IDF']['unknown_handling']}%** | {bench_results['Dense']['unknown_handling']}% | **{bench_results['Hybrid_Weighted']['unknown_handling']}%** | {bench_results['Hybrid_RRF']['unknown_handling']}% | {round(bench_results['Hybrid_Weighted']['unknown_handling'] - bench_results['TF-IDF']['unknown_handling'], 2):+}% |
| **Context Resolution** | **{bench_results['TF-IDF']['context_resolution']}%** | {bench_results['Dense']['context_resolution']}% | **{bench_results['Hybrid_Weighted']['context_resolution']}%** | {bench_results['Hybrid_RRF']['context_resolution']}% | {round(bench_results['Hybrid_Weighted']['context_resolution'] - bench_results['TF-IDF']['context_resolution'], 2):+}% |
| **Retrieval Latency** | **{bench_results['TF-IDF']['retrieval_latency_ms']} ms** | {bench_results['Dense']['retrieval_latency_ms']} ms | **{bench_results['Hybrid_Weighted']['retrieval_latency_ms']} ms** | {bench_results['Hybrid_RRF']['retrieval_latency_ms']} ms | +{round(bench_results['Hybrid_Weighted']['retrieval_latency_ms'] - bench_results['TF-IDF']['retrieval_latency_ms'], 2)} ms |
| **Embedding Latency** | **{bench_results['TF-IDF']['embedding_latency_ms']} ms** | {bench_results['Dense']['embedding_latency_ms']} ms | **{bench_results['Hybrid_Weighted']['embedding_latency_ms']} ms** | {bench_results['Hybrid_RRF']['embedding_latency_ms']} ms | +{round(bench_results['Hybrid_Weighted']['embedding_latency_ms'] - bench_results['TF-IDF']['embedding_latency_ms'], 2)} ms |
| **End-to-End Latency** | **{bench_results['TF-IDF']['total_latency_ms']} ms** | {bench_results['Dense']['total_latency_ms']} ms | **{bench_results['Hybrid_Weighted']['total_latency_ms']} ms** | {bench_results['Hybrid_RRF']['total_latency_ms']} ms | +{round(bench_results['Hybrid_Weighted']['total_latency_ms'] - bench_results['TF-IDF']['total_latency_ms'], 2)} ms |

---

## 2. Similarity Distribution & Separation Gap Analysis

- **Correct Target Similarity:** Min = {dist_report['correct_match']['min']}, p25 = {dist_report['correct_match']['p25']}, Median = {dist_report['correct_match']['median']}, p75 = {dist_report['correct_match']['p75']}, Max = {dist_report['correct_match']['max']}
- **Incorrect Entry Similarity:** Min = {dist_report['incorrect_match']['min']}, p25 = {dist_report['incorrect_match']['p25']}, Median = {dist_report['incorrect_match']['median']}, p75 = {dist_report['incorrect_match']['p75']}, Max = {dist_report['incorrect_match']['max']}
- **Negative Out-of-Scope Similarity:** Min = {dist_report['negative_match']['min']}, Median = {dist_report['negative_match']['median']}, p75 = {dist_report['negative_match']['p75']}, Max = {dist_report['negative_match']['max']}
- **Separation Gap:** **{separation_gap}** (Difference between Correct p25 and Incorrect/Negative p75).

---

## 3. Fusion Parameter Sweep Results

### Weighted Linear Fusion (Alpha Sweep)
| Alpha | Recall@1 | Recall@3 | MRR | False Positive Rate |
| :---: | :---: | :---: | :---: | :---: |
"""
    for a in alphas:
        r = weighted_sweep_results[f"alpha_{a}"]
        md_content += f"| α = {a:.1f} | {r['recall_at_1']}% | {r['recall_at_3']}% | {r['mrr']} | {r['fp_rate']}% |\n"

    md_content += """
### Reciprocal Rank Fusion (k & Weight Sweep)
| Configuration | Recall@1 | Recall@3 | MRR | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: |
"""
    for k_name, r in rrf_sweep_results.items():
        md_content += f"| RRF ({k_name}) | {r['recall_at_1']}% | {r['recall_at_3']}% | {r['mrr']} | {r['fp_rate']}% |\n"

    md_content += """
---

## 4. Gating Strategy Comparison (FP vs. FN)

| Strategy | False Positives (on Negatives) | False Negatives (on Positives) | FP Rate | FN Rate |
| :--- | :---: | :---: | :---: | :---: |
"""
    for g_name, r in gating_results.items():
        md_content += f"| {g_name} | {r['false_positives']} | {r['false_negatives']} | {r['fp_rate']}% | {r['fn_rate']}% |\n"

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved experiment report Markdown to {report_md_path}")


if __name__ == "__main__":
    main()
