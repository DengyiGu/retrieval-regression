"""Audit real BEIR judgments and exported TREC runs against the JSON workflow.

First run benchmarks/scifact.py. This audit uses only cached data and local run files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from scifact import obtain_archive, oracle_errors

from retrieval_regression import (
    InputFormatError,
    Threshold,
    __version__,
    compare,
    load_qrels,
    load_run,
)

ROOT = Path(__file__).resolve().parents[1]


def render_run(run, tag):
    return "".join(
        f"{qid} Q0 {doc} {rank} {run[qid][doc]!r} {tag}\n"
        for qid in sorted(run)
        for rank, doc in enumerate(sorted(run[qid], key=lambda d: (-run[qid][d], d)), 1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "reports/scifact")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/scifact.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/formats-audit")
    args = parser.parse_args()
    required = ("qrels", "bm25", "tfidf", "groups")
    if any(not (args.runs_dir / f"{name}.json").exists() for name in required):
        parser.error("run benchmarks/scifact.py first, or set --runs-dir to its output")
    original = {name: json.loads((args.runs_dir / f"{name}.json").read_text()) for name in required}
    data = obtain_archive(args.archive, False)
    args.output.mkdir(parents=True, exist_ok=True)
    beir_path = args.output / "qrels.tsv"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        beir_path.write_bytes(archive.read("scifact/qrels/test.tsv"))
    trec_qrels = args.output / "qrels.trec"
    trec_qrels.write_text(
        "".join(
            f"{q} 0 {d} {int(g)}\n"
            for q, docs in original["qrels"].items()
            for d, g in docs.items()
        ),
        encoding="utf-8",
        newline="\n",
    )
    for name in ("bm25", "tfidf"):
        (args.output / f"{name}.run").write_text(
            render_run(original[name], name), encoding="utf-8", newline="\n"
        )
    qrels = load_qrels(beir_path, format="beir")
    assert qrels == original["qrels"] == load_qrels(trec_qrels, format="trec")
    base = load_run(args.output / "bm25.run", format="trec")
    cand = load_run(args.output / "tfidf.run", format="trec")
    assert base == original["bm25"] and cand == original["tfidf"]
    kwargs = {
        "k": [1, 5, 10],
        "groups": original["groups"],
        "thresholds": [Threshold("ndcg@10", max_drop=0.02)],
        "bootstrap_samples": 1000,
        "seed": 42,
    }
    expected = compare(original["qrels"], original["bm25"], original["tfidf"], **kwargs)
    actual = compare(qrels, base, cand, **kwargs)
    assert actual.to_dict() == expected.to_dict()
    reference_qrels = {q: {d: int(g) for d, g in docs.items()} for q, docs in qrels.items()}
    reference = {
        "bm25": max(oracle_errors(reference_qrels, base, actual.baseline).values()),
        "tfidf": max(oracle_errors(reference_qrels, cand, actual.candidate).values()),
    }
    config = args.output / "gates.json"
    config.write_text('{"thresholds":[{"metric":"ndcg@10","max_drop":0.02}]}', encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "retrieval_regression",
        "compare",
        "--qrels",
        str(beir_path),
        "--qrels-format",
        "beir",
        "--baseline",
        str(args.output / "bm25.run"),
        "--baseline-format",
        "trec",
        "--candidate",
        str(args.output / "tfidf.run"),
        "--candidate-format",
        "trec",
        "--groups",
        str(args.runs_dir / "groups.json"),
        "--config",
        str(config),
        "--k",
        "1",
        "5",
        "10",
        "--bootstrap-samples",
        "1000",
        "--seed",
        "42",
        "--json",
        str(args.output / "comparison.json"),
        "--html",
        str(args.output / "comparison.html"),
        "--manifest",
        str(args.output / "manifest.json"),
    ]
    cli = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert cli.returncode == 1, cli.stderr
    # to_dict contains tuples; a JSON round-trip normalizes those for the CLI comparison.
    assert json.loads(cli.stdout) == json.loads(json.dumps(actual.to_dict()))
    corrupted = args.output / "corrupted.run"
    base_text = (args.output / "bm25.run").read_text(encoding="utf-8")
    rows = base_text.splitlines()
    first, second = rows[0].split(), rows[1].split()
    second[4] = repr(float(first[4]) + 1)
    mixed = rows[0].split()
    mixed[-1] = "another-experiment"
    faults = {
        "duplicate_document": base_text + rows[0] + "\n",
        "mixed_run_tag": base_text + " ".join(mixed) + "\n",
        "score_rank_conflict": rows[0] + "\n" + " ".join(second) + "\n",
        "nonfinite_score": rows[0].replace(first[4], "NaN") + "\n",
    }
    rejected = []
    for name, text in faults.items():
        corrupted.write_text(text, encoding="utf-8")
        try:
            load_run(corrupted, format="trec")
        except InputFormatError as exc:
            rejected.append({"case": name, "line": exc.line_number, "reason": exc.message})
        else:
            raise AssertionError(f"corruption not rejected: {name}")
    omitted = set(sorted(qrels)[::10])
    missing_path = args.output / "missing.run"
    missing_path.write_text(
        render_run({q: v for q, v in base.items() if q not in omitted}, "fault"),
        encoding="utf-8",
        newline="\n",
    )
    faulty = compare(
        qrels,
        base,
        load_run(missing_path, format="trec"),
        k=[10],
        thresholds=[Threshold("ndcg@10", max_drop=0)],
    )
    assert not faulty.passed and faulty.candidate.query_count == len(qrels)
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    state = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    record = {
        "experiment": "beir-trec-adapter-audit",
        "version": __version__,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": state,
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted((ROOT / "src/retrieval_regression").glob("*.py")),
                Path(__file__).resolve(),
            ]
        },
        "dataset": {
            "name": "BEIR SciFact",
            "split": "test",
            "queries": len(qrels),
            "judgments": sum(map(len, qrels.values())),
            "archive_sha256": hashlib.sha256(data).hexdigest(),
        },
        "run_rows": {"bm25": sum(map(len, base.values())), "tfidf": sum(map(len, cand.values()))},
        "formats": ["original BEIR TSV", "TREC qrels", "TREC run", "JSON"],
        "json_api_and_trec_api_and_cli_identical": True,
        "ndcg10": {
            "baseline": actual.baseline.summary["ndcg@10"],
            "candidate": actual.candidate.summary["ndcg@10"],
        },
        "gate_passed": actual.passed,
        "independent_reference_max_error": reference,
        "corruptions_rejected": rejected,
        "omitted_query_fault": {
            "omitted": len(omitted),
            "denominator": faulty.candidate.query_count,
            "gate_detected": not faulty.passed,
        },
        "manifest": manifest,
        "python": platform.python_version(),
        "limitations": [
            "One scientific domain and two lexical runs.",
            "TREC runs are exported from saved JSON; raw qrels use original BEIR bytes.",
            "Strict subset: positive unique ranks, one run tag and integer judgment grades.",
            "Parsers retain full mappings in memory; not a streaming evaluator.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "dataset",
                    "run_rows",
                    "ndcg10",
                    "gate_passed",
                    "independent_reference_max_error",
                    "corruptions_rejected",
                    "omitted_query_fault",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
