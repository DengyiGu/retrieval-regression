"""Seeded synthetic evaluator timing; not a retrieval quality benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
import tracemalloc
from pathlib import Path

from retrieval_regression import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("work/benchmarks/synthetic.json"))
    args = parser.parse_args()
    if min(args.queries, args.documents, args.repeats) < 1:
        parser.error("queries/documents/repeats must be positive")
    rng = random.Random(args.seed)
    qrels = {
        f"q{i}": {f"d{j}": 1 for j in range(min(3, args.documents))} for i in range(args.queries)
    }
    run = {
        qid: {f"d{j}": rng.random() for j in range(args.documents)}
        for i, qid in enumerate(qrels)
        if i % 10 != 0
    }
    evaluate(qrels, run)
    seconds = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        evaluate(qrels, run)
        seconds.append(time.perf_counter() - start)
    tracemalloc.start()
    evaluate(qrels, run)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record = {
        "kind": "synthetic_evaluator_benchmark",
        "queries": args.queries,
        "documents_per_run_query": args.documents,
        "seed": args.seed,
        "repeats": args.repeats,
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "separate_tracemalloc_peak_bytes": peak,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "k": [5, 10],
        "input_sha256": hashlib.sha256(
            json.dumps([qrels, run], sort_keys=True).encode()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
