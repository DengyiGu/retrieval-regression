import pytest

from retrieval_regression import Threshold, compare

QRELS = {"q1": {"a": 1}, "q2": {"b": 1}}
BASE = {"q1": {"a": 2}, "q2": {"b": 2}}
CAND = {"q1": {"wrong": 2}, "q2": {"b": 2}}


def test_absolute_gate_drop_and_group_breakdown():
    result = compare(
        QRELS,
        BASE,
        CAND,
        k=[1],
        groups={"q1": "hard"},
        thresholds=[
            Threshold("recall@1", max_drop=0.5),
            Threshold("recall@1", min_value=0.1, group="hard"),
        ],
    )
    assert result.delta["recall@1"] == -0.5
    assert result.groups["hard"]["query_count"] == 1
    assert result.groups["__ungrouped__"]["candidate"]["recall@1"] == 1
    assert result.gates[0].passed
    assert not result.gates[1].passed
    assert not result.passed


def test_no_thresholds_is_a_comparison_without_enforcement():
    result = compare(QRELS, BASE, {}, k=[1])
    assert result.passed and not result.gates
    assert result.delta["recall@1"] == -1


@pytest.mark.parametrize(
    "threshold",
    [
        Threshold("missing@1", max_drop=0),
        Threshold("recall@1"),
        Threshold("recall@1", min_value=1.1),
        Threshold("recall@1", max_drop=-1),
        Threshold("recall@1", max_drop=float("nan")),
        Threshold("recall@1", min_value=True),
        Threshold("recall@1", max_drop=0, group="unknown"),
    ],
)
def test_invalid_thresholds_fail_closed(threshold):
    with pytest.raises(ValueError):
        compare(QRELS, BASE, CAND, k=[1], thresholds=[threshold])


def test_invalid_groups():
    with pytest.raises(ValueError, match="unknown query"):
        compare(QRELS, BASE, CAND, groups={"typo": "group"})
    with pytest.raises(ValueError, match="non-empty"):
        compare(QRELS, BASE, CAND, groups={"q1": ""})


def test_paired_bootstrap_is_reproducible_and_does_not_change_gates():
    first = compare(QRELS, BASE, CAND, k=[1], bootstrap_samples=150, seed=7)
    second = compare(QRELS, BASE, CAND, k=[1], bootstrap_samples=150, seed=7)
    assert first == second
    ci = first.confidence_intervals["recall@1"]
    assert ci.low == -1 and ci.high == 0
    assert first.passed


def test_identical_runs_have_zero_delta_and_zero_width_interval():
    result = compare(QRELS, BASE, BASE, k=[1], bootstrap_samples=20)
    assert all(delta == 0 for delta in result.delta.values())
    assert all(ci.low == ci.high == 0 for ci in result.confidence_intervals.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bootstrap_samples": -1},
        {"bootstrap_samples": True},
        {"confidence": 1},
        {"confidence": float("nan")},
        {"seed": "7"},
    ],
)
def test_invalid_statistics_arguments(kwargs):
    with pytest.raises(ValueError):
        compare(QRELS, BASE, CAND, **kwargs)
