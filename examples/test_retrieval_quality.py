"""Try: python -m pytest examples/test_retrieval_quality.py -q"""

from pathlib import Path

from retrieval_regression import Threshold
from retrieval_regression.io import load_json

DATA = Path(__file__).parent / "toy"


def test_candidate_does_not_regress(retrieval_check):
    retrieval_check(
        load_json(DATA / "qrels.json"),
        load_json(DATA / "baseline.json"),
        load_json(DATA / "candidate.json"),
        k=[1, 3],
        thresholds=[Threshold("recall@3", min_value=0.7, max_drop=0.02)],
    )
