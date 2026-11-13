import copy

import pytest

from retrieval_regression import evaluate, rerank


def test_reranking_batches_across_queries_and_preserves_candidate_set():
    calls = []

    def scorer(pairs):
        calls.append(pairs)
        return (float(len(text)) for _, text in pairs)

    source = {"q2": {"c": 2}, "q1": {"b": 3, "a": 4, "ignored": 1}, "empty": {}}
    original = copy.deepcopy(source)
    result = rerank(
        {"q1": "one", "q2": "two", "empty": "none"},
        {"a": "a", "b": "longer", "c": "ccc"},
        source,
        scorer,
        top_k=2,
        batch_size=2,
    )
    assert result == {"q1": {"a": 1, "b": 6}, "q2": {"c": 3}, "empty": {}}
    assert calls == [[("one", "a"), ("one", "longer")], [("two", "ccc")]]
    assert source == original


def test_tied_candidates_use_same_document_id_rule_as_evaluator():
    result = rerank(
        {"q": "query"}, {"a": "a", "z": "z"}, {"q": {"z": 1, "a": 1}}, lambda pairs: [-2], top_k=1
    )
    assert result == {"q": {"a": -2}}
    assert evaluate({"q": {"a": 1}}, result, k=[1]).summary["recall@1"] == 1


@pytest.mark.parametrize(
    "kwargs", [{"top_k": 0}, {"top_k": True}, {"batch_size": -1}, {"batch_size": 1.5}]
)
def test_bad_batch_and_candidate_settings(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        rerank({"q": "query"}, {"d": "doc"}, {"q": {"d": 1}}, lambda pairs: [1], **kwargs)


@pytest.mark.parametrize(
    "output", [[], [1, 2], [float("nan")], [float("inf")], [True], ["1"], None]
)
def test_invalid_scorer_output_fails_explicitly(output):
    with pytest.raises(ValueError, match="scorer"):
        rerank({"q": "query"}, {"d": "doc"}, {"q": {"d": 1}}, lambda pairs: output)


@pytest.mark.parametrize(
    "queries,corpus,run",
    [
        ({}, {"d": "doc"}, {"q": {"d": 1}}),
        ({"q": ""}, {"d": "doc"}, {"q": {"d": 1}}),
        ({"q": "query"}, {}, {"q": {"d": 1}}),
        ({"q": "query"}, {"d": " "}, {"q": {"d": 1}}),
        ([], {}, {}),
    ],
)
def test_invalid_text_is_rejected_before_scorer_call(queries, corpus, run):
    calls = []
    with pytest.raises(ValueError):
        rerank(queries, corpus, run, lambda pairs: calls.append(pairs))
    assert calls == []


def test_empty_run_does_not_invoke_model_and_invalid_callable_is_rejected():
    assert rerank({}, {}, {}, lambda pairs: pytest.fail("must not call")) == {}
    with pytest.raises(ValueError, match="callable"):
        rerank({}, {}, {}, None)


def test_model_failures_propagate_without_being_misreported_as_bad_scores():
    def scorer(pairs):
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        rerank({"q": "query"}, {"d": "doc"}, {"q": {"d": 1}}, scorer)
