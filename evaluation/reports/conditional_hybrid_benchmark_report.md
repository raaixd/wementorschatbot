# WeMentors Conditional Hybrid Retrieval Benchmark Report

**Scope:** 156 Total Evaluation Queries (30 Focused Retrieval Cases + 126 Standard Benchmark Cases)  
**Embedding Model:** `gemini-embedding-001` (3072 dimensions)  
**Configuration:** 50/50 Weighted Fusion ($lpha=0.5$), Fast-path Lexical $\ge 0.35$ & Coverage $\ge 0.75$, Hybrid Acceptance $\ge 0.28$  

---

## 1. Full Benchmark Comparison Table

| Metric | Current TF-IDF | New Conditional Hybrid | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1** | 72.0% | **76.0%** | +4.00% | Better |
| **Recall@3** | 84.0% | **84.0%** | +0.00% | Same |
| **MRR** | 0.7733 | **0.8** | +0.0267 | Better |
| **NDCG@3** | 0.7905 | **0.8105** | +0.0200 | Better |
| **False Positive Rate** | 0.0% | **60.0%** | +60.00% | **Worse** |
| **False Negative Rate** | 16.0% | **16.0%** | +0.00% | Same |
| **Answer Correctness** | 78.85% | **82.69%** | +3.84% | Better |
| **Groundedness** | 96.79% | **96.79%** | +0.00% | Same |
| **Hallucination Rate** | 3.21% | **3.21%** | +0.00% | Same |
| **Unknown Handling** | 92.31% | **92.95%** | +0.64% | Better |
| **Context Resolution** | 90.38% | **92.95%** | +2.57% | Better |
| **In-Memory Retrieval Latency** | 8.81 ms | **10.22 ms** | +1.41 ms | **Worse** |
| **Embedding API Calls** | 0 / 156 | **67 / 156** | +67 | Cost |
| **Embedding API Call Rate** | 0.00% | **42.95%** | +42.95% | Cost |
| **Embedding Latency P50** | 0.0 ms | **104.42 ms** | +104.42 ms | Cost |
| **Embedding Latency P95** | 0.0 ms | **119.65 ms** | +119.65 ms | Cost |

**Regressions vs TF-IDF:** **False Positive Rate** (0.0% -> 60.0%); **In-Memory Retrieval Latency** (8.81 ms -> 10.22 ms).

---

## 2. API Call Rate & Efficiency Analysis

- **Total Queries Evaluated:** 156
- **Fast Path Lexical Passes (Zero Embedding API Calls):** 89 (57.05%)
- **Semantic Path Invocations:** 67 (42.95%)
- **Efficiency Finding:** The conditional fast path prevents calling the Gemini Embedding API on **over 57% of user turns**, eliminating network latency for standard queries while seamlessly activating semantic vector fusion for natural-language paraphrases.
