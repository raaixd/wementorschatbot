# Changes report: `improve/eval-honesty-and-client-ip`

Branch: `improve/eval-honesty-and-client-ip` (cut from `main` at `f65dec9`). Nothing here is merged. Review and merge deliberately.

This report is written so another agent can pick the work up without the original conversation.

## Why this branch exists

A review of the repo produced five improvement tips. This branch implements the first slice of two of them:

| Tip | Topic | Status on this branch |
|---|---|---|
| 1 | Serverless state and rate limiting on Vercel | **Partly done** (client-IP fix only) |
| 2 | Make eval reports tell the truth | **Done** (generators and committed reports) |
| 3 | Deduplicate KB / `index.html` / logo copies, slim the repo | Not started |
| 4 | Split `conversation.py` (5,059 lines) | Not started |
| 5 | Close the loop with real feedback and knowledge gaps | Not started |

## What changed

### 1. Eval reports now compute their verdicts (tip 2)

**Problem.** The report templates hard-coded their verdicts:
- `evaluation/runners/run_benchmark.py` wrote `**PASS**` for every metric, whatever the value. The committed baseline report showed Intent Classification Accuracy 35.71% (target >= 90%), Answer Correctness 89.68% (target >= 90%) and Unknown Question Handling 92.06% (target 100%) all as PASS.
- `evaluation/runners/run_conditional_hybrid_benchmark.py` hard-coded `0.00%` as the delta for several rows and `Measured` as the status. It reported False Positive Rate going 0.0% -> 60.0% as `0.00%`. The `+` sign was also hard-coded, so a negative delta would have printed as `+-`.

**Fix.**
- `run_benchmark.py`: new `_status(value, op, target)` helper. Every Status cell is now computed from `summary[...]` against the same thresholds that were already printed in the table. Failures render as `**FAIL**`. "MEETS TARGET" and "ZERO CRASHES" became computed PASS/FAIL too.
- `run_conditional_hybrid_benchmark.py`: new `_COMPARISON_METRICS`, `_comparison_rows()` and `_regression_note()`. Deltas are signed and computed. The Status column is `Better` / `Same` / `**Worse**`, direction-aware (for example lower is better for False Positive Rate and latency). A "Regressions vs TF-IDF" line is emitted under the table. Embedding-cost rows are labelled `Cost`.
- Committed reports `evaluation/reports/baseline_report.md` and `conditional_hybrid_benchmark_report.md` were updated by applying the same logic to the existing JSON in `evaluation/reports/`. **The benchmarks were not re-run** (they need API keys and about 80 s of live calls). Numbers are unchanged. Only the verdict and delta columns differ.

**What the honest reports now show (not fixed here):**
- Baseline: Answer Correctness, Intent Classification Accuracy and Unknown Question Handling are FAIL.
- Hybrid: False Positive Rate regressed 0.0% -> 60.0% and retrieval latency regressed 8.81 -> 10.22 ms. Recall@1, MRR, NDCG@3, Answer Correctness, Unknown Handling and Context Resolution improved.

### 2. Rate limiter keys on the real visitor behind a proxy (part of tip 1)

**Problem.** `_client_key()` in `chatbot-backendexperiment/app/main.py` used `request.client.host`. Behind Vercel's proxy that is the proxy address, so visitors share rate-limit buckets.

**Fix.**
- `app/config.py`: new `TRUST_PROXY_HEADERS` (env, default `False`).
- `app/main.py`: when enabled, `_client_key()` prefers `X-Real-IP`, then the first hop of `X-Forwarded-For`, then falls back to the socket peer. Default behaviour is unchanged.
- `api/index.py`: `os.environ.setdefault("TRUST_PROXY_HEADERS", "true")`, since Vercel's edge sets those headers. Anywhere else the flag stays off, because turning it on without a proxy that overwrites the headers would let clients spoof their IP and dodge the limiter.
- New tests: `chatbot-backendexperiment/tests/test_client_key.py` (5 tests).

## Verification

- Full suite: `pytest` from the repo root, ignoring `test_live_api.py` as `pytest.ini` does: **228 passed, 185 subtests passed**.
- The new report helpers were exercised against the existing `baseline_report.json` and `conditional_hybrid_benchmark_report.json`. They produce the verdicts above.
- **Not verified:** a live benchmark re-run, and the real Vercel deployment. Check on a preview deploy that `X-Real-IP` is present and that two different visitors get separate buckets.

## Known issues found but not changed

1. **Intent accuracy of 35.71% may be an evaluator bug, not a product bug.** Look at `evaluation/metrics/evaluator.py` and the expected-intent labels in `evaluation/questions/dataset.json` before trusting or "fixing" it.
2. **Hybrid False Positive Rate 60%** is computed only over the focused negative cases (`total_focused_neg` in the hybrid runner), so the sample is probably small. Check its size and investigate before claiming hybrid retrieval is a win.
3. The two reports use different query sets, so baseline correctness is not comparable (78.85% vs 89.68%). Docs also disagree on KB size: 52 (README), 40 (audit), 35 (performance doc).
4. **Rest of tip 1 is still open.** On Vercel, sessions live in SQLite at `/tmp/chatbot.db`, and the rate limiter and dedupe cache are in memory. Each serverless instance has its own copy, so multi-turn memory and rate limits are not shared. A hosted store (Redis or a hosted SQLite/Postgres) is needed. `ratelimit.py` is designed so only that file changes.
5. CORS allows the `"null"` origin, and `requirements.txt` is unpinned.

## Merging notes

- Touches `evaluation/`, `chatbot-backendexperiment/app/{config,main}.py`, `api/index.py`, one new test file and this report. No data files or knowledge-base changes.
- Regenerating the reports with a real benchmark run will overwrite the two `.md` files with the new format automatically.
