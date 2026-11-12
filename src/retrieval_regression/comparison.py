"""Paired comparisons, category breakdowns, and explicit quality gates."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from .evaluation import evaluate
from .models import ComparisonResult, ConfidenceInterval, GateResult, Threshold
from .validation import _identifier, _number


def _percentile(values: list[float], p: float) -> float:
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def compare(
    qrels: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    *,
    k: Sequence[int] = (5, 10),
    groups: Mapping[str, str] | None = None,
    thresholds: Sequence[Threshold] = (),
    bootstrap_samples: int = 0,
    confidence: float = 0.95,
    seed: int = 0,
) -> ComparisonResult:
    """Compare runs on identical queries; max_drop is an absolute metric drop.

    Optional bootstrap CIs resample query-level paired differences. They are descriptive,
    are not p-values, and do not change quality-gate decisions.
    """
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 0
    ):
        raise ValueError("bootstrap_samples must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    confidence = _number(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    base = evaluate(qrels, baseline, k=k)
    cand = evaluate(qrels, candidate, k=k)
    metrics = tuple(base.summary)
    qids = tuple(base.per_query)
    delta = {metric: cand.summary[metric] - base.summary[metric] for metric in metrics}
    per_query_delta = {
        qid: {
            metric: cand.per_query[qid][metric] - base.per_query[qid][metric] for metric in metrics
        }
        for qid in qids
    }
    grouped: dict[str, dict[str, Any]] = {}
    if groups is not None:
        if not isinstance(groups, Mapping):
            raise ValueError("groups must map query IDs to category names")
        for qid, group in groups.items():
            if qid not in base.per_query:
                raise ValueError(f"groups contains unknown query ID {qid!r}")
            _identifier(group, f"groups[{qid!r}]")
        buckets: dict[str, list[str]] = {}
        for qid in qids:
            buckets.setdefault(groups.get(qid, "__ungrouped__"), []).append(qid)
        for group, members in sorted(buckets.items()):
            b = {
                m: math.fsum(base.per_query[q][m] for q in members) / len(members) for m in metrics
            }
            c = {
                m: math.fsum(cand.per_query[q][m] for q in members) / len(members) for m in metrics
            }
            grouped[group] = {
                "query_count": len(members),
                "baseline": b,
                "candidate": c,
                "delta": {m: c[m] - b[m] for m in metrics},
            }
    gates: list[GateResult] = []
    for threshold in thresholds:
        if not isinstance(threshold, Threshold):
            raise ValueError("thresholds must contain Threshold objects")
        _identifier(threshold.metric, "threshold metric")
        if threshold.group is not None:
            _identifier(threshold.group, "threshold group")
        if threshold.metric not in metrics:
            raise ValueError(f"unknown/uncomputed threshold metric {threshold.metric!r}")
        if threshold.min_value is None and threshold.max_drop is None:
            raise ValueError("a threshold needs min_value and/or max_drop")
        minimum = None if threshold.min_value is None else _number(threshold.min_value, "min_value")
        drop = (
            None
            if threshold.max_drop is None
            else _number(threshold.max_drop, "max_drop", nonnegative=True)
        )
        if minimum is not None and not 0 <= minimum <= 1:
            raise ValueError("min_value must be between 0 and 1")
        if drop is not None and drop > 1:
            raise ValueError("max_drop must be between 0 and 1")
        if threshold.group is None:
            before, after = base.summary[threshold.metric], cand.summary[threshold.metric]
        else:
            if threshold.group not in grouped:
                raise ValueError(f"unknown threshold group {threshold.group!r}")
            before = grouped[threshold.group]["baseline"][threshold.metric]
            after = grouped[threshold.group]["candidate"][threshold.metric]
        passed = (minimum is None or after + 1e-12 >= minimum) and (
            drop is None or after + 1e-12 >= before - drop
        )
        gates.append(
            GateResult(threshold.metric, threshold.group, before, after, minimum, drop, passed)
        )
    intervals: dict[str, ConfidenceInterval] = {}
    if bootstrap_samples:
        for metric in metrics:
            rng = random.Random(seed)
            differences = [per_query_delta[qid][metric] for qid in qids]
            means = sorted(
                math.fsum(rng.choices(differences, k=len(qids))) / len(qids)
                for _ in range(bootstrap_samples)
            )
            alpha = (1 - confidence) / 2
            intervals[metric] = ConfidenceInterval(
                _percentile(means, alpha),
                _percentile(means, 1 - alpha),
                confidence,
                bootstrap_samples,
            )
    return ComparisonResult(
        base,
        cand,
        delta,
        per_query_delta,
        grouped,
        tuple(gates),
        all(gate.passed for gate in gates),
        intervals,
        seed,
    )
