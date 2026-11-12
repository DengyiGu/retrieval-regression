"""Verify an installed wheel in an isolated environment, outside the source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path, label: str) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")
    print(f"PASS: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--with-pytest", action="store_true")
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wheel-smoke-", dir=work) as temporary:
        directory = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(directory / "env")
        python = (
            directory / "env" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        )
        run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
            directory,
            "install wheel without dependencies",
        )
        code = (
            "import importlib.util; import retrieval_regression as rr; "
            "assert importlib.util.find_spec('pytest') is None; "
            "assert 'site-packages' in rr.__file__; "
            "assert rr.evaluate({'q':{'d':1}}, {'q':{'d':1}}, k=[1]).summary['recall@1']==1"
        )
        run([str(python), "-I", "-c", code], directory, "core API without pytest")
        run(
            [str(python), "-I", "-m", "retrieval_regression", "--version"],
            directory,
            "module entry point",
        )
        cli = directory / "env" / ("Scripts/rr.exe" if sys.platform == "win32" else "bin/rr")
        run(
            [
                str(cli),
                "evaluate",
                "--qrels",
                str(ROOT / "examples/toy/qrels.json"),
                "--run",
                str(ROOT / "examples/toy/baseline.json"),
            ],
            directory,
            "installed CLI entry point",
        )
        if args.with_pytest:
            run(
                [str(python), "-m", "pip", "install", str(wheel) + "[pytest]"],
                directory,
                "install pytest extra",
            )
            (directory / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
            (directory / "test_installed.py").write_text(
                "from retrieval_regression import Threshold\n"
                "def test_plugin(retrieval_check):\n"
                "    retrieval_check({'q':{'d':1}}, {'q':{'d':1}}, {'q':{'d':1}}, "
                "k=[1], thresholds=[Threshold('recall@1', max_drop=0)])\n",
                encoding="utf-8",
            )
            run(
                [str(python), "-I", "-m", "pytest", "-q", "-c", "pytest.ini", "test_installed.py"],
                directory,
                "installed pytest plugin auto-discovery",
            )
    record = {
        "wheel": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "core_without_dependencies": True,
        "module_entry_point": True,
        "cli_entry_point": True,
        "pytest_plugin": args.with_pytest,
    }
    output = work / "validation" / "wheel-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
