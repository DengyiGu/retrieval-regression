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
from .io import load_json, load_qrels, load_run, load_thresholds
from .provenance import fingerprint, make_manifest, verify_inputs, write_manifest
from .reporting import write_html, write_json


def _same_file(left: Path, right: Path) -> bool:
    return left == right or (left.exists() and right.exists() and left.samefile(right))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval-regression")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "compare"):
        sub = commands.add_parser(command)
        sub.add_argument("--qrels", required=True)
        sub.add_argument("--qrels-format", choices=["json", "beir", "trec"], default="json")
        sub.add_argument("--k", nargs="+", type=int, default=[5, 10])
        sub.add_argument("--json", dest="json_output")
        sub.add_argument("--manifest", dest="manifest_output")
        if command == "evaluate":
            sub.add_argument("--run", required=True)
            sub.add_argument("--run-format", choices=["json", "trec"], default="json")
        else:
            sub.add_argument("--baseline", required=True)
            sub.add_argument("--candidate", required=True)
            sub.add_argument("--baseline-format", choices=["json", "trec"], default="json")
            sub.add_argument("--candidate-format", choices=["json", "trec"], default="json")
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
            for name in ("json_output", "html_output", "manifest_output")
            if getattr(args, name, None)
        ]
        if any(
            _same_file(left, right)
            for index, left in enumerate(output_paths)
            for right in output_paths[index + 1 :]
        ):
            raise ValueError("JSON, HTML and manifest output paths must be different")
        if any(_same_file(source, target) for source in input_paths for target in output_paths):
            raise ValueError("output paths must not overwrite input files")
        sources = tuple(
            fingerprint(name, getattr(args, name), getattr(args, name + "_format", "json"))
            for name in ("qrels", "run", "baseline", "candidate", "groups", "config")
            if args.manifest_output and getattr(args, name, None)
        )
        parameters = {"command": args.command, "k": sorted(set(args.k))}
        qrels = load_qrels(args.qrels, format=args.qrels_format)
        if args.command == "evaluate":
            evaluation = evaluate(qrels, load_run(args.run, format=args.run_format), k=args.k)
            if args.manifest_output:
                verify_inputs(sources)
            if args.json_output:
                write_json(evaluation, args.json_output)
            if args.manifest_output:
                write_manifest(
                    make_manifest(evaluation, sources, parameters, version=__version__),
                    args.manifest_output,
                )
            print(json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=True, allow_nan=False))
            return 0
        comparison = compare(
            qrels,
            load_run(args.baseline, format=args.baseline_format),
            load_run(args.candidate, format=args.candidate_format),
            k=args.k,
            groups=load_json(args.groups) if args.groups else None,
            thresholds=load_thresholds(args.config) if args.config else (),
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
        if args.manifest_output:
            verify_inputs(sources)
            parameters.update(
                {
                    "bootstrap_samples": args.bootstrap_samples,
                    "confidence": args.confidence,
                    "seed": args.seed,
                }
            )
        if args.json_output:
            write_json(comparison, args.json_output)
        if args.html_output:
            write_html(comparison, args.html_output)
        if args.manifest_output:
            write_manifest(
                make_manifest(comparison, sources, parameters, version=__version__),
                args.manifest_output,
            )
        print(json.dumps(comparison.to_dict(), indent=2, ensure_ascii=True, allow_nan=False))
        return 0 if comparison.passed else 1
    except (ValueError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
