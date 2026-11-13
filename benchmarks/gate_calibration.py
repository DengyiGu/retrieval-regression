"""Seeded synthetic gate calibration; no GPU, data downloads or third-party packages."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from retrieval_regression import Threshold, compare


def simulate(trials: int, samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    scenarios = {}
    for name, candidate_probability in [("equal_quality", 0.7), ("regression", 0.45)]:
        counts = {"point": Counter(), "bootstrap": Counter()}
        for trial in range(trials):
            qrels = {str(i): {"relevant": 1} for i in range(30)}
            baseline = {q: {"relevant" if rng.random() < 0.7 else "wrong": 1} for q in qrels}
            candidate = {
                q: {"relevant" if rng.random() < candidate_probability else "wrong": 1}
                for q in qrels
            }
            for mode in counts:
                result = compare(
                    qrels,
                    baseline,
                    candidate,
                    k=[1],
                    thresholds=[Threshold("ndcg@1", max_drop=0.02)],
                    gate_mode=mode,
                    bootstrap_samples=samples if mode == "bootstrap" else 0,
                    seed=seed + trial,
                )
                counts[mode][result.status] += 1
        scenarios[name] = {mode: dict(counts[mode]) for mode in counts}
    return {
        "kind": "synthetic_calibration",
        "seed": seed,
        "trials_per_scenario": trials,
        "queries": 30,
        "metric": "ndcg@1",
        "max_drop": 0.02,
        "confidence": 0.95,
        "bootstrap_samples": samples,
        "baseline_hit_probability": 0.7,
        "candidate_hit_probabilities": {"equal_quality": 0.7, "regression": 0.45},
        "independent_query_and_system_bernoulli_draws": True,
        "scenarios": scenarios,
        "limitations": [
            "Synthetic distribution, not a real-world false-alarm guarantee",
            "Inconclusive results block by default; fewer false failures do not mean more passes",
            "Percentile coverage is approximate; repeated runs/selection need separate control",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--output", type=Path, default=Path("reports/gate-calibration.json"))
    args = parser.parse_args()
    result = simulate(args.trials, args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
