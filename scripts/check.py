"""Run the project's local quality checks with the active interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ["ruff", "check", "."],
    ["ruff", "format", "--check", "."],
    ["mypy", "src/retrieval_regression"],
    ["coverage", "erase"],
    ["coverage", "run", "-m", "pytest", "-q", "--basetemp=work/pytest-tests"],
    ["coverage", "combine"],
    ["coverage", "report"],
    ["pytest", "examples/test_retrieval_quality.py", "-q"],
]


def main() -> int:
    (ROOT / "work").mkdir(exist_ok=True)
    for command in COMMANDS:
        print("Running: python -m " + " ".join(command), flush=True)
        result = subprocess.run([sys.executable, "-m", *command], cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    print("All local checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
