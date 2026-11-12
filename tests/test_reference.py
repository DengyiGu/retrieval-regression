"""Independent metric oracle on seeded graded judgments with unambiguous rankings."""

import random

import pytest

from retrieval_regression import evaluate

ir = pytest.importorskip("ir_measures")


@pytest.mark.reference
def test_metrics_match_ir_measures_on_seeded_rankings():
    rng = random.Random(20261112)
    qrels, run = {}, {}
    reference_qrels, reference_run = [], []
    for qi in range(20):
        qid = f"q{qi}"
        docs = {f"d{di}": rng.randrange(4) for di in range(15)}
        docs["d0"] = 3
        qrels[qid] = docs
        scores = rng.sample(range(1000), 15)
        run[qid] = {doc: float(score) for doc, score in zip(docs, scores, strict=True)}
        reference_qrels.extend(ir.Qrel(qid, doc, grade) for doc, grade in docs.items())
        reference_run.extend(ir.ScoredDoc(qid, doc, score) for doc, score in run[qid].items())
    result = evaluate(qrels, run, k=[1, 5, 10, 20])
    for cutoff in result.k:
        pairs = [
            ("recall", ir.R @ cutoff),
            ("precision", ir.P @ cutoff),
            ("mrr", ir.RR @ cutoff),
            ("ndcg", ir.nDCG @ cutoff),
        ]
        oracle = ir.calc_aggregate(
            [measure for _, measure in pairs], reference_qrels, reference_run
        )
        for name, measure in pairs:
            assert result.summary[f"{name}@{cutoff}"] == pytest.approx(oracle[measure], abs=1e-12)
