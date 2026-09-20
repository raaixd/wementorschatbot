"""
Master Release Test Suite Runner for WeMentors AI Chatbot.

Executes all 21 regression and verification suites, including core pipeline,
natural language, false-booking prevention, failure-path reliability, and production smoke tests.
Generates an aggregated release scorecard.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = TESTS_DIR.parent
PROJECT_ROOT = BACKEND_ROOT.parent

# Discover all test_*.py files in tests directory
TEST_FILES = sorted([
    f for f in TESTS_DIR.glob("test_*.py")
    if f.name != "test_live_api.py"  # Live API requires a running external server port
])


def run_all():
    print("=" * 80)
    print("WEMENTORS v1.0 PRODUCTION RELEASE TEST SUITE")
    print(f"Total Test Suites to Execute: {len(TEST_FILES)}")
    print(f"Python Executable: {sys.executable}")
    print("=" * 80 + "\n")

    t_suite_start = time.perf_counter()
    suite_results = []
    total_passed = 0
    total_failed = 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_ROOT)

    for idx, test_file in enumerate(TEST_FILES, start=1):
        rel_path = test_file.relative_to(PROJECT_ROOT)
        print(f"[{idx:02d}/{len(TEST_FILES):02d}] Running {test_file.name}...", end=" ", flush=True)

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        duration = round(time.perf_counter() - t0, 2)

        if proc.returncode == 0:
            print(f"PASS ({duration}s)")
            total_passed += 1
            suite_results.append((test_file.name, "PASS", duration, ""))
        else:
            print(f"FAIL ({duration}s)")
            total_failed += 1
            err_snippet = (proc.stderr or proc.stdout)[-300:].strip()
            suite_results.append((test_file.name, "FAIL", duration, err_snippet))

    total_duration = round(time.perf_counter() - t_suite_start, 2)

    print("\n" + "=" * 80)
    print("RELEASE SUITE EXECUTION SUMMARY")
    print("=" * 80)
    for name, status, dur, err in suite_results:
        status_str = "PASS" if status == "PASS" else "FAIL"
        print(f"  {status_str:4s} | {name:<45s} | {dur:>6.2f}s")
        if err:
            print(f"       Error: {err.replace(chr(10), ' ')}")

    print("-" * 80)
    print(f"Total Suites Executed : {len(TEST_FILES)}")
    print(f"Total Passed          : {total_passed}")
    print(f"Total Failed          : {total_failed}")
    print(f"Total Execution Time  : {total_duration}s")
    print(f"Pass Rate             : {(total_passed / len(TEST_FILES) * 100):.1f}%")
    print("=" * 80)

    if total_failed == 0:
        print("\nALL RELEASE SUITES PASSED — PRODUCTION READINESS VERIFIED!\n")
        sys.exit(0)
    else:
        print(f"\n{total_failed} SUITE(S) FAILED — RELEASE BLOCKED.\n")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
