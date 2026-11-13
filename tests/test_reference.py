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


@pytest.mark.reference
def test_standard_file_adapters_match_independent_file_readers(tmp_path):
    from retrieval_regression import load_qrels, load_run

    rng = random.Random(90421)
    qrels_lines, run_lines = [], []
    for qi in range(25):
        qid = f"{qi:04}"
        grades = [2] + [rng.randrange(3) for _ in range(11)]
        scores = rng.sample(range(1000), 12)
        order = sorted(range(12), key=lambda i: -scores[i])
        qrels_lines.extend(f"{qid} 0 d{di} {grade}\n" for di, grade in enumerate(grades))
        run_lines.extend(
            f"{qid} Q0 d{di} {rank} {scores[di]} test\n" for rank, di in enumerate(order, 1)
        )
    qrels_path, run_path = tmp_path / "qrels.txt", tmp_path / "run.txt"
    qrels_path.write_text("".join(qrels_lines), encoding="utf-8")
    run_path.write_text("".join(run_lines), encoding="utf-8")
    actual = evaluate(
        load_qrels(qrels_path, format="trec"), load_run(run_path, format="trec"), k=[1, 5, 10]
    )
    for k in actual.k:
        measures = {
            f"recall@{k}": ir.R @ k,
            f"precision@{k}": ir.P @ k,
            f"mrr@{k}": ir.RR @ k,
            f"ndcg@{k}": ir.nDCG @ k,
        }
        oracle = list(
            ir.iter_calc(
                list(measures.values()),
                ir.read_trec_qrels(str(qrels_path)),
                ir.read_trec_run(str(run_path)),
            )
        )
        names = {measure: name for name, measure in measures.items()}
        for score in oracle:
            assert actual.per_query[score.query_id][names[score.measure]] == pytest.approx(
                score.value, abs=1e-12
            )


@pytest.mark.reference
def test_descending_id_ties_match_trec_eval_provider():
    qrels = {"q": {"a": 0, "z": 1}}
    run = {"q": {"a": 1, "z": 1}}
    # Pin the provider: ir-measures may dispatch other metrics to different engines.
    oracle = ir.providers.registry["pytrec_eval"].calc_aggregate(
        [ir.nDCG @ 1, ir.R @ 1],
        [ir.Qrel("q", "a", 0), ir.Qrel("q", "z", 1)],
        [ir.ScoredDoc("q", "a", 1), ir.ScoredDoc("q", "z", 1)],
    )
    actual = evaluate(qrels, run, k=[1], tie_breaker="doc_id_desc")
    assert actual.summary["ndcg@1"] == oracle[ir.nDCG @ 1] == 1
    assert actual.summary["recall@1"] == oracle[ir.R @ 1] == 1
