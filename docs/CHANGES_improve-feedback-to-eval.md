# Changes report: `improve/feedback-to-eval`

Branch: `improve/feedback-to-eval` (cut from `main` at `f65dec9`, independent of the other two improve branches). Not merged.

Covers tip 5 of the repo review ("close the loop with real users"). Only the tooling and the question sheet are done. The knowledge-base content is **not** changed, because the facts must come from the WeMentors team.

## What changed
- `scripts/export_feedback_cases.py` (new): reads the chatbot SQLite DB read-only, finds replies rated thumbs-down, rebuilds the user turns that led to each one, and writes candidate cases in the `dataset.json` shape (default output `evaluation/cache/feedback_candidates.json`, which is already gitignored because it holds real visitor text).
  - Candidates are **not runnable as-is**: `expected_intent` is null, `expected_keywords` is empty and `needs_review` is true. It records the bot's actual reply, intent, matched entries, confidence and the visitor's comment under `observed`, so a person can label the right answer.
  - Falls back to the session's latest assistant reply when a rating has no `message_id`. Repeat ratings of the same reply collapse into one case. `--only-with-comment` skips silent ratings.
- `chatbot-backendexperiment/tests/test_feedback_export.py` (new): 5 tests on an in-memory database.
- `docs/OWNER_QUESTIONS.md` (new): 10 fill-in questions covering all open gaps in `KNOWLEDGE_GAPS.md` (demo timing, Grades 11-12, fees, state boards, refunds and rescheduling, mentors, time zones, discounts), plus the steps for turning answers into KB entries and eval cases.

## Verification
- Full pytest: 228 passed, 185 subtests passed.
- The exporter was tested only against a synthetic in-memory database. It has **not** been run on a real production database. Vercel keeps the DB at `/tmp`, so feedback there is not persisted (see tip 1 in `CHANGES_improve-eval-honesty-and-client-ip.md`). Real feedback is only collected where the DB survives, for example a local or long-running deployment.

## Not done
- No KB entries were added and no eval cases were labeled. Both depend on the answers in `OWNER_QUESTIONS.md`.
- Low-confidence assistant replies (`confidence` below the retrieval threshold) are another good source of candidates. The exporter does not read them yet.
