"""Keep the intentionally duplicated deployment files identical.

The same files must exist in several places (the Vercel function bundle in
api/, the backend package, and the repo root), so deleting copies would break
a deploy. Instead one copy of each is canonical and the rest are verified, or
synced, against it.

    python scripts/sync_duplicates.py --check   # exit 1 if any copy differs (CI)
    python scripts/sync_duplicates.py           # overwrite copies from canonical

Edit the canonical file, then run this script without --check.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# canonical file -> copies that must match it
DUPLICATES = {
    "knowledge/wementors_kb.json": [
        "chatbot-backendexperiment/app/wementors_kb.json",
        "api/wementors_kb.json",
    ],
    "index.html": [
        "chatbot-backendexperiment/index.html",
        "wementors.html",
    ],
    "wm_brand_logo.png": [
        "chatbot-backendexperiment/wm_brand_logo.png",
        "api/wm_brand_logo.png",
    ],
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_drift(root: Path = ROOT) -> list[tuple[str, str]]:
    """Return (canonical, copy) pairs whose contents differ or are missing."""
    drift = []
    for canonical, copies in DUPLICATES.items():
        source = root / canonical
        if not source.exists():
            drift.append((canonical, "<canonical file missing>"))
            continue
        want = _digest(source)
        for copy in copies:
            target = root / copy
            if not target.exists() or _digest(target) != want:
                drift.append((canonical, copy))
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="report drift and exit 1 instead of fixing it")
    args = parser.parse_args()

    drift = find_drift()
    if not drift:
        print("OK: all duplicated files match their canonical copy.")
        return 0

    for canonical, copy in drift:
        print(f"{'DRIFT' if args.check else 'SYNC '}: {copy}  <-  {canonical}")
    if args.check:
        print("\nRun `python scripts/sync_duplicates.py` to fix.", file=sys.stderr)
        return 1

    for canonical, copy in drift:
        if copy.startswith("<"):
            print(f"Cannot sync: {canonical} is missing.", file=sys.stderr)
            return 1
        (ROOT / copy).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / canonical, ROOT / copy)
    print(f"Synced {len(drift)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
