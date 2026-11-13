"""Leakage prevention and negative-sampling contracts; no model download required."""

import pytest

from experiments.reranking.data import normalize_query, sample_pairs, split_queries


def test_group_split_excludes_heldout_text_and_is_order_independent():
    queries = {f"q{i}": f"different question {i}" for i in range(100)}
    queries.update({"duplicate": "  DIFFERENT question 7  ", "heldout": "different QUESTION 9"})
    qrels = {q: {"doc": 1} for q in queries if q != "heldout"}
    split = split_queries(qrels, queries, {"heldout"})
    reversed_split = split_queries(dict(reversed(list(qrels.items()))), queries, {"heldout"})
    assert split == reversed_split
    assert split["excluded_text_overlap"] == ["q9"]
    assert ("q7" in split["train"]) == ("duplicate" in split["train"])
    train_text = {normalize_query(queries[q]) for q in split["train"]}
    val_text = {normalize_query(queries[q]) for q in split["validation"]}
    assert not train_text & val_text
    assert normalize_query(queries["heldout"]) not in train_text | val_text


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_invalid_split_fraction(fraction):
    with pytest.raises(ValueError):
        split_queries({"q": {"d": 1}}, {"q": "question"}, set(), validation_fraction=fraction)


def test_overlapping_query_ids_and_insufficient_groups_fail():
    with pytest.raises(ValueError, match="IDs overlap"):
        split_queries({"q": {"d": 1}}, {"q": "question"}, {"q"})
    with pytest.raises(ValueError, match="non-empty"):
        split_queries({"q": {"d": 1}}, {"q": ""}, set())
    with pytest.raises(ValueError, match="not enough"):
        split_queries({"q": {"d": 1}}, {"q": "question"}, set())


@pytest.mark.parametrize("mode", ["random_negative", "hard_negative"])
def test_all_known_positives_excluded_from_negative_pool_and_counts_match(mode):
    qrels = {"q": {"p1": 1, "p2": 2}}
    docs = {"p1", "p2", *{f"n{i}" for i in range(12)}}
    candidates = {"q": {doc: index for index, doc in enumerate(sorted(docs))}}
    pairs = sample_pairs(qrels, candidates, docs, mode=mode)
    assert len(pairs) == 8
    positive = [p for p in pairs if p["label"] == 1]
    negative = [p for p in pairs if p["label"] == 0]
    assert {p["docid"] for p in positive} == {"p1", "p2"}
    assert len({p["docid"] for p in negative}) == 6
    assert not {p["docid"] for p in negative} & {"p1", "p2"}
    assert all(p["source"].startswith("unjudged_") for p in negative)
    assert pairs == sample_pairs(qrels, candidates, docs, mode=mode)


def test_hard_negatives_follow_retrieval_rank_and_explicit_negative_is_labeled():
    pairs = sample_pairs(
        {"q": {"p": 1, "n1": 0}},
        {"q": {"n1": 4, "p": 5, "n2": 3}},
        ["p", "n1", "n2"],
        mode="hard_negative",
        negatives_per_positive=1,
    )
    assert pairs[-1] == {"qid": "q", "docid": "n1", "label": 0, "source": "judged_negative"}


@pytest.mark.parametrize("mode", ["random_negative", "hard_negative"])
def test_insufficient_negatives_fail_instead_of_changing_experiment(mode):
    with pytest.raises(ValueError, match="not enough"):
        sample_pairs({"q": {"p": 1}}, {"q": {"p": 1}}, ["p"], mode=mode)
    with pytest.raises(ValueError, match="positive documents"):
        sample_pairs({"q": {"p": 1}}, {"q": {}}, [], mode=mode)


def test_unknown_mode_and_invalid_count_rejected():
    with pytest.raises(ValueError, match="unknown"):
        sample_pairs({}, {}, [], mode="typo")
    with pytest.raises(ValueError, match="positive"):
        sample_pairs({}, {}, [], mode="hard_negative", negatives_per_positive=0)
