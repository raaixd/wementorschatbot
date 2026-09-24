# WeMentors Hybrid Semantic Retrieval Experiment Report

**Date:** 2026-09-25 02:17:43  
**Evaluation Scope:** 30 Focused Benchmark Cases (Paraphrases, Boundaries, Adversarial Negatives) + 126 Standard Benchmark Cases  
**Embedding Model Verified:** `gemini-embedding-001` (3072 dimensions)  

---

## 1. Primary Empirical Comparison Table

| Metric | TF-IDF (Baseline) | Dense Only | Hybrid Weighted (α=0.5) | Hybrid RRF (k=60) | Delta (Weighted vs. TF-IDF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recall@1** | **63.33%** | 60.0% | **70.0%** | 66.67% | **+6.67%** |
| **Recall@3** | **73.33%** | 76.67% | **76.67%** | 76.67% | **+3.34%** |
| **MRR** | **0.6778** | 0.6667 | **0.7333** | 0.7111 | **+0.0555** |
| **NDCG@3** | **0.6921** | 0.6921 | **0.7421** | 0.7254 | **+0.05** |
| **False Positive Rate** | **16.67%** | 16.67% | **10.0%** | 16.67% | **-6.67%** |
| **Answer Correctness** | **76.67%** | 86.67% | **80.0%** | 53.33% | **+3.33%** |
| **Groundedness** | **93.33%** | 93.33% | **93.33%** | 100.0% | +0.0% |
| **Hallucination Rate** | **6.67%** | 6.67% | **6.67%** | 0.0% | +0.0% |
| **Unknown Handling** | **93.33%** | 93.33% | **93.33%** | 93.33% | +0.0% |
| **Context Resolution** | **100.0%** | 100.0% | **100.0%** | 100.0% | +0.0% |
| **Retrieval Latency** | **0.48 ms** | 16.09 ms | **16.34 ms** | 16.76 ms | +15.86 ms |
| **Embedding Latency** | **0.0 ms** | 104.22 ms | **104.22 ms** | 104.22 ms | +104.22 ms |
| **End-to-End Latency** | **0.48 ms** | 120.31 ms | **120.56 ms** | 120.98 ms | +120.08 ms |

---

## 2. Similarity Distribution & Separation Gap Analysis

- **Correct Target Similarity:** Min = 0.4828, p25 = 0.6224, Median = 0.6674, p75 = 0.7203, Max = 0.8075
- **Incorrect Entry Similarity:** Min = 0.4262, p25 = 0.508, Median = 0.5391, p75 = 0.5723, Max = 0.7547
- **Negative Out-of-Scope Similarity:** Min = 0.4634, Median = 0.5225, p75 = 0.6161, Max = 0.6404
- **Separation Gap:** **0.0063** (Difference between Correct p25 and Incorrect/Negative p75).

---

## 3. Fusion Parameter Sweep Results

### Weighted Linear Fusion (Alpha Sweep)
| Alpha | Recall@1 | Recall@3 | MRR | False Positive Rate |
| :---: | :---: | :---: | :---: | :---: |
| α = 0.2 | 60.0% | 70.0% | 0.65 | 20.0% |
| α = 0.3 | 60.0% | 70.0% | 0.65 | 20.0% |
| α = 0.4 | 63.33% | 70.0% | 0.6667 | 16.67% |
| α = 0.5 | 63.33% | 70.0% | 0.6667 | 16.67% |
| α = 0.6 | 63.33% | 70.0% | 0.6667 | 16.67% |
| α = 0.7 | 63.33% | 70.0% | 0.6667 | 16.67% |
| α = 0.8 | 63.33% | 70.0% | 0.6667 | 16.67% |

### Reciprocal Rank Fusion (k & Weight Sweep)
| Configuration | Recall@1 | Recall@3 | MRR | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: |
| RRF (k_10_w_1.0) | 66.67% | 76.67% | 0.7111 | 16.67% |
| RRF (k_10_w_1.5) | 60.0% | 76.67% | 0.6778 | 16.67% |
| RRF (k_20_w_1.0) | 66.67% | 76.67% | 0.7111 | 16.67% |
| RRF (k_20_w_1.5) | 60.0% | 76.67% | 0.6778 | 16.67% |
| RRF (k_60_w_1.0) | 66.67% | 76.67% | 0.7111 | 16.67% |
| RRF (k_60_w_1.5) | 60.0% | 76.67% | 0.6778 | 16.67% |
| RRF (k_100_w_1.0) | 66.67% | 76.67% | 0.7111 | 16.67% |
| RRF (k_100_w_1.5) | 60.0% | 76.67% | 0.6778 | 16.67% |

---

## 4. Gating Strategy Comparison (FP vs. FN)

| Strategy | False Positives (on Negatives) | False Negatives (on Positives) | FP Rate | FN Rate |
| :--- | :---: | :---: | :---: | :---: |
| 1_lexical_coverage_only | 1 | 8 | 20.0% | 32.0% |
| 2_semantic_threshold_only | 2 | 6 | 40.0% | 24.0% |
| 3_separate_gates | 1 | 8 | 20.0% | 32.0% |
| 4_hybrid_confidence | 2 | 6 | 40.0% | 24.0% |
