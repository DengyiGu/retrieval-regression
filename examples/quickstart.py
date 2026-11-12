"""Run from the repository root after editable installation."""

from pathlib import Path

from retrieval_regression import Threshold, compare
from retrieval_regression.io import load_json
from retrieval_regression.reporting import write_html, write_json

DATA = Path(__file__).parent / "toy"
result = compare(
    load_json(DATA / "qrels.json"),
    load_json(DATA / "baseline.json"),
    load_json(DATA / "candidate.json"),
    k=[1, 3],
    groups=load_json(DATA / "groups.json"),
    thresholds=[Threshold("recall@3", min_value=0.7, max_drop=0.02)],
    bootstrap_samples=500,
    seed=42,
)
print("Baseline:", result.baseline.summary)
print("Candidate:", result.candidate.summary)
print("Gate passed:", result.passed)
write_json(result, "reports/toy-comparison.json")
write_html(result, "reports/toy-comparison.html")
