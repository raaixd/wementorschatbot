# Changes report: `improve/kb-owner-answers`

Branch: `improve/kb-owner-answers` (cut from `main` at `f65dec9`). Not merged. **CI will be red until `knowledge/kb_embeddings.json` is regenerated** (see "Required before merge").

## Source of the facts
Answers from the project owner on 2026-10-16, to the questions in `docs/OWNER_QUESTIONS.md` (that sheet lives on `improve/feedback-to-eval`):

1. Demo scheduling: once you sign up, someone will reach out and connect you accordingly.
2. Grades 11-12 / JEE / NEET: not now, and no plans as of now.
3. Fees: stay as is (unpublished, contact the team).
4. Refunds and rescheduling: dependent (case by case) and on genuine reasons, up to the mentors. Rescheduling will happen so the child does not suffer.

Nothing was invented beyond these. In particular **no notice period, weekend or evening availability, or refund conditions were added**, because the owner did not give any.

## What changed
- `knowledge/wementors_kb.json` (53 entries, was 52). Edited as exact text replacements to keep the file's formatting.
  - `demo-class-available`: now says that after sign-up someone from the team will reach out to connect the visitor.
  - `grades-11-12-jee-neet-unsupported`: adds "there are no plans to offer them at the moment".
  - `missed-classes-catch-up`: adds that rescheduling depends on a genuine reason, is up to the mentors, and is done so the child does not fall behind or suffer. The phrases the existing test requires ("genuine reason", "specific topic", "rescheduled") are kept.
  - **New** `refund-policy` entry: refunds are case by case, contact the team (same contact details as `fees-and-pricing`). It does not promise or deny a refund.
  - `policy-restrictions`: removed "refunds" from the list of things the bot cannot discuss, and the `refund` keyword, so refund questions route to the new entry.
  - `fees-and-pricing`: unchanged (owner said keep).
- `api/wementors_kb.json` and `chatbot-backendexperiment/app/wementors_kb.json`: copied from the canonical `knowledge/` file so all three stay identical.
- `chatbot-backendexperiment/app/personality.py`: the hard-coded `GRADE_11_12_UNSUPPORTED_RESPONSE` also gets the "no plans" sentence, because that intent answers from this string, not from the KB.
- `chatbot-backendexperiment/tests/test_core_pipeline.py`: the "Do you offer a refund policy?" negative case was removed, since refunds are now a known topic. Replaced with an unrelated unsupported query.
- `chatbot-backendexperiment/tests/test_owner_answers.py` (new): checks refund answers are case by case with no invented promises, rescheduling wording, demo answer says the team reaches out without inventing timings, Grades 11-12 says no plans, fees show no amounts, and **embeddings match the KB hash and entry ids**.
- `docs/KNOWLEDGE_GAPS.md`: status of GAP-01, 02, 03, 06 updated. GAP-02 is only partial.

## Required before merge
`knowledge/kb_embeddings.json` stores a hash of the KB, and the app disables semantic retrieval when it no longer matches. Regenerate with a Gemini key:

    GEMINI_API_KEY=... python scripts/generate_kb_embeddings.py

Currently failing for that reason only: `test_owner_answers.py::test_kb_embeddings_match_knowledge_base`, and two semantic tests in `test_hybrid_retrieval.py` (`test_sick_and_skip_recovers_missed_classes`, `test_adult_business_meeting_speaking`). All 229 other tests pass. Then re-run the benchmark (`evaluation/runners/run_benchmark.py`), since a new entry can shift retrieval.

## Things to check
- **Refund wording.** "Dependent ... on genuine reasons ... up to the mentors" was read as applying to rescheduling; refunds were written as "case by case" with no stated conditions. Confirm that is what the owner meant.
- **Code vs KB drift.** The hard-coded Grades 11-12 response mentions "Communicative English" and "IELTS Preparation", while the KB entry says "Confident Speaker". One of them is out of date.
- **Still unanswered** from the question sheet: state boards (GAP-04), mentor qualifications (GAP-08), time zones (GAP-09), discounts (GAP-07), and demo notice periods (GAP-02).
