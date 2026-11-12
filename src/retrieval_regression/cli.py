"""Command line: 0 success, 1 failed quality gates, 2 invalid input/configuration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .comparison import compare
from .evaluation import evaluate
from .io import load_json, load_thresholds
from .reporting import write_html, write_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval-regression")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "compare"):
        sub = commands.add_parser(command)
        sub.add_argument("--qrels", required=True)
        sub.add_argument("--k", nargs="+", type=int, default=[5, 10])
        sub.add_argument("--json", dest="json_output")
        if command == "evaluate":
            sub.add_argument("--run", required=True)
        else:
            sub.add_argument("--baseline", required=True)
            sub.add_argument("--candidate", required=True)
            sub.add_argument("--groups")
            sub.add_argument("--config")
            sub.add_argument("--html", dest="html_output")
            sub.add_argument("--bootstrap-samples", type=int, default=0)
            sub.add_argument("--confidence", type=float, default=0.95)
            sub.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        input_paths = {
            Path(getattr(args, name)).resolve()
            for name in ("qrels", "run", "baseline", "candidate", "groups", "config")
            if getattr(args, name, None)
        }
        output_paths = [
            Path(getattr(args, name)).resolve()
            for name in ("json_output", "html_output")
            if getattr(args, name, None)
        ]
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("JSON and HTML output paths must be different")
        if input_paths.intersection(output_paths):
            raise ValueError("output paths must not overwrite input files")
        qrels = load_json(args.qrels)
        if args.command == "evaluate":
            evaluation = evaluate(qrels, load_json(args.run), k=args.k)
            if args.json_output:
                write_json(evaluation, args.json_output)
            print(json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        comparison = compare(
            qrels,
            load_json(args.baseline),
            load_json(args.candidate),
            k=args.k,
            groups=load_json(args.groups) if args.groups else None,
            thresholds=load_thresholds(args.config) if args.config else (),
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
        if args.json_output:
            write_json(comparison, args.json_output)
        if args.html_output:
            write_html(comparison, args.html_output)
        print(json.dumps(comparison.to_dict(), indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if comparison.passed else 1
    except (ValueError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
