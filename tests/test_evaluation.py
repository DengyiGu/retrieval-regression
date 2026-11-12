import math

import pytest

from retrieval_regression import evaluate


def test_hand_calculated_graded_ndcg_and_binary_metrics():
    qrels = {"q": {"a": 3, "b": 1, "n": 0}}
    result = evaluate(qrels, {"q": {"b": 9, "a": 8}}, k=[1, 2, 5])
    assert result.summary["recall@1"] == 0.5
    assert result.summary["precision@5"] == 0.4
    assert result.summary["mrr@1"] == 1
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert result.summary["ndcg@2"] == pytest.approx(expected)
    assert result.summary["ndcg@1"] == pytest.approx(1 / 3)


def test_missing_and_empty_queries_stay_in_denominator():
    qrels = {"answered": {"a": 1}, "missing": {"b": 1}, "empty": {"c": 1}}
    result = evaluate(qrels, {"answered": {"a": 1}, "empty": {}}, k=[1])
    assert result.query_count == 3
    assert result.summary["recall@1"] == pytest.approx(1 / 3)
    assert result.per_query["missing"]["ndcg@1"] == 0
    assert {(d.code, d.query_id) for d in result.diagnostics} == {
        ("missing_query", "missing"),
        ("empty_ranking", "empty"),
    }


def test_wholly_empty_run_scores_zero():
    result = evaluate({"q": {"d": 1}}, {}, k=[1])
    assert all(value == 0 for value in result.summary.values())


def test_ties_are_deterministic_and_documented():
    qrels = {"q": {"a": 1}}
    left = evaluate(qrels, {"q": {"b": 2, "a": 2}}, k=[1])
    right = evaluate(qrels, {"q": {"a": 2, "b": 2}}, k=[1])
    assert left == right
    assert left.summary["recall@1"] == 1


def test_unjudged_documents_are_not_assumed_relevant():
    result = evaluate({"q": {"d": 1}}, {"q": {"unknown": 3, "d": 2}}, k=[1, 2])
    assert result.summary["mrr@2"] == 0.5
    assert result.summary["precision@1"] == 0
    assert result.diagnostics[0].code == "unjudged_documents"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "2", None])
def test_invalid_run_scores_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        evaluate({"q": {"d": 1}}, {"q": {"d": value}})


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), False, "1"])
def test_invalid_relevance_is_rejected(value):
    with pytest.raises(ValueError):
        evaluate({"q": {"d": value}}, {})


@pytest.mark.parametrize(
    "qrels",
    [{}, {"q": {}}, {"q": {"d": 0}}, {"": {"d": 1}}, {"q": {"": 1}}, {1: {"d": 1}}, {"q": []}],
)
def test_invalid_qrels(qrels):
    with pytest.raises(ValueError):
        evaluate(qrels, {})


@pytest.mark.parametrize("k", [[], [0], [-1], [1.0], [True], "10"])
def test_invalid_cutoffs(k):
    with pytest.raises(ValueError):
        evaluate({"q": {"d": 1}}, {}, k=k)


def test_unknown_query_and_malformed_rankings_fail():
    with pytest.raises(ValueError, match="unknown query"):
        evaluate({"q": {"d": 1}}, {"typo": {"d": 1}})
    with pytest.raises(ValueError, match="mapping"):
        evaluate({"q": {"d": 1}}, {"q": ["d", "d"]})


def test_cutoffs_are_sorted_and_deduplicated():
    assert evaluate({"q": {"d": 1}}, {}, k=[10, 1, 10]).k == (1, 10)


def test_extreme_finite_grades_do_not_overflow_ndcg():
    result = evaluate({"q": {"a": 1e308, "b": 1e308}}, {"q": {"a": 2, "b": 1}}, k=[2])
    assert result.summary["ndcg@2"] == 1


def test_inputs_are_not_mutated():
    qrels, run = {"q": {"a": 2}}, {"q": {"a": 1}}
    evaluate(qrels, run, k=[1])
    assert qrels == {"q": {"a": 2}}
    assert run == {"q": {"a": 1}}
