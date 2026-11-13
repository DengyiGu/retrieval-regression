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
    gate_mode: str = "point",
    min_queries: int = 20,
    multiple_testing: str = "bonferroni",
    negative_grades: str = "error",
    no_positive: str = "error",
    tie_breaker: str = "doc_id_asc",
) -> ComparisonResult:
    """Compare runs on identical queries; max_drop is an absolute metric drop.

    Point mode applies budgets to observed means. Bootstrap mode applies budgets
    to percentile intervals and returns inconclusive when evidence is insufficient.
    Queries must be independent sampling units; coverage is approximate.
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
    if gate_mode not in {"point", "bootstrap"}:
        raise ValueError("gate_mode must be point or bootstrap")
    if multiple_testing not in {"none", "bonferroni"}:
        raise ValueError("multiple_testing must be none or bonferroni")
    if isinstance(min_queries, bool) or not isinstance(min_queries, int) or min_queries < 2:
        raise ValueError("min_queries must be an integer >= 2")
    if gate_mode == "bootstrap" and bootstrap_samples < 1000:
        raise ValueError("bootstrap gates require bootstrap_samples >= 1000")
    base = evaluate(
        qrels,
        baseline,
        k=k,
        negative_grades=negative_grades,
        no_positive=no_positive,
        tie_breaker=tie_breaker,
    )
    cand = evaluate(
        qrels,
        candidate,
        k=k,
        negative_grades=negative_grades,
        no_positive=no_positive,
        tie_breaker=tie_breaker,
    )
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
    buckets: dict[str, list[str]] = {}
    if groups is not None:
        if not isinstance(groups, Mapping):
            raise ValueError("groups must map query IDs to category names")
        for qid, group in groups.items():
            if qid not in qrels:
                raise ValueError(f"groups contains unknown query ID {qid!r}")
            _identifier(group, f"groups[{qid!r}]")
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
    cache: dict[tuple[float, ...], list[float]] = {}

    def interval(values: tuple[float, ...], level: float) -> ConfidenceInterval:
        if values not in cache:
            rng = random.Random(seed)
            cache[values] = sorted(
                math.fsum(rng.choices(values, k=len(values))) / len(values)
                for _ in range(bootstrap_samples)
            )
        alpha = (1 - level) / 2
        return ConfidenceInterval(
            _percentile(cache[values], alpha),
            _percentile(cache[values], 1 - alpha),
            level,
            bootstrap_samples,
        )

    bounds = sum(
        (t.min_value is not None) + (t.max_drop is not None)
        for t in thresholds
        if isinstance(t, Threshold)
    )
    gate_confidence = (
        1 - (1 - confidence) / max(1, bounds) if (multiple_testing == "bonferroni") else confidence
    )
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
        gate_members = qids if threshold.group is None else tuple(buckets[threshold.group])
        evidence: dict[str, ConfidenceInterval] = {}
        status = "pass" if passed else "fail"
        reason = "Observed means compared with absolute budgets"
        if gate_mode == "bootstrap":
            targets: dict[str, float] = {}
            if minimum is not None:
                evidence["min_value"] = interval(
                    tuple(cand.per_query[q][threshold.metric] for q in gate_members),
                    gate_confidence,
                )
                targets["min_value"] = minimum
            if drop is not None:
                evidence["max_drop"] = interval(
                    tuple(per_query_delta[q][threshold.metric] for q in gate_members),
                    gate_confidence,
                )
                targets["max_drop"] = -drop
            if len(gate_members) < min_queries:
                status = "inconclusive"
                reason = f"Only {len(gate_members)} queries; min_queries={min_queries}"
            elif any(ci.high + 1e-12 < targets[name] for name, ci in evidence.items()):
                status = "fail"
                reason = "An interval is entirely below its budget boundary"
            elif all(ci.low + 1e-12 >= targets[name] for name, ci in evidence.items()):
                status = "pass"
                reason = "All interval lower bounds meet their budget boundaries"
            else:
                status = "inconclusive"
                reason = "An interval overlaps its budget boundary"
        gates.append(
            GateResult(
                threshold.metric,
                threshold.group,
                before,
                after,
                minimum,
                drop,
                status == "pass",
                status,
                len(gate_members),
                evidence,
                reason,
            )
        )
    intervals: dict[str, ConfidenceInterval] = {}
    if bootstrap_samples:
        intervals = {
            m: interval(tuple(per_query_delta[q][m] for q in qids), confidence) for m in metrics
        }
        for group, members in buckets.items():
            grouped[group]["confidence_intervals"] = {
                m: {
                    "low": ci.low,
                    "high": ci.high,
                    "confidence": ci.confidence,
                    "samples": ci.samples,
                }
                for m in metrics
                for ci in [interval(tuple(per_query_delta[q][m] for q in members), confidence)]
            }
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
        gate_mode,
        multiple_testing,
    )
