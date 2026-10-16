# Changes report: `improve/dedupe-guard`

Branch: `improve/dedupe-guard` (cut from `main` at `f65dec9`, independent of `improve/eval-honesty-and-client-ip`). Not merged.

Covers tip 3 of the repo review ("keep one copy of each file"), in its safe form: the copies stay, but drift between them is now detected in CI and fixable with one command.

## Why copies were not deleted
The same files are needed in several places: `api/` is the Vercel function bundle, `chatbot-backendexperiment/` is the backend package, and the repo root serves the site. Removing copies could break a deploy, and a Vercel deploy could not be tested from this environment. The docs already recorded a past incident where slightly different KB copies made tests pass or fail depending on which file loaded.

## What changed
- `scripts/sync_duplicates.py` (new): one canonical file per group, the rest are copies.
  - `knowledge/wementors_kb.json` -> `app/wementors_kb.json`, `api/wementors_kb.json`
  - `index.html` -> `chatbot-backendexperiment/index.html`, `wementors.html`
  - `wm_brand_logo.png` -> `chatbot-backendexperiment/wm_brand_logo.png`, `api/wm_brand_logo.png`
  - `--check` prints drift and exits 1. Without it, copies are overwritten from the canonical file.
- `chatbot-backendexperiment/tests/test_no_duplicate_drift.py` (new): fails if any copy differs, plus a test that drift is detected.
- `.github/workflows/ci.yml`: runs `sync_duplicates.py --check` before pytest.
- `docs/KNOWLEDGE_GAPS.md`: the "synchronize all 3 copies" step now says to edit the canonical KB and run the script.

## Verification
- All copies were byte-identical at the start (checked by SHA-256).
- `--check` passes on a clean tree. After a deliberate edit to `api/wementors_kb.json` it reported the drift and exited 1, and the sync mode repaired it.
- Full pytest: 225 passed, 185 subtests passed.

## Not done (still open from tip 3)
- `brag-output/` (about 7 MB of mp4/mp3), the duplicate PDFs in `docs/` (`PROJECT_REPORT.pdf` and `WeMentors_Project_Report.pdf` are the same size) and the `.agent*` folders are still tracked. Moving them out of git is a judgement call for the owner. History would still contain them unless rewritten.
- The `kb_embeddings.json` hash check only covers `knowledge/wementors_kb.json`. If the canonical KB changes, regenerate embeddings with `scripts/generate_kb_embeddings.py`.
