"""Run the repository-wide CPU/static integration test suite.

The runner intentionally does not install or execute heavyweight GPU model stacks.
It verifies FastSD's unit tests, the pinned SpecEdge source/config integration, and
Python syntax for the official SpecEdge source before GPU deployment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> bool:
    print(f"\n== {label} ==", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        print(f"[FAIL] {label} exited with {completed.returncode}", flush=True)
        return False
    print(f"[PASS] {label}", flush=True)
    return True


def main() -> int:
    checks = (
        (
            "FastSD and baseline unit tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        ),
        (
            "SpecEdge pinned-source and paper-config doctor",
            [sys.executable, "baselines/specedge/repro.py", "doctor"],
        ),
        (
            "SpecEdge Python source compilation",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "baselines/specedge/repro.py",
                "baselines/specedge/official/src",
            ],
        ),
    )

    passed = True
    for label, command in checks:
        passed = run(label, command) and passed

    print("\nRESULT: " + ("all integration checks passed" if passed else "checks failed"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
