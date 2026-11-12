"""Public result and threshold types; scores are always candidate minus baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Qrels = dict[str, dict[str, float]]
Run = dict[str, dict[str, float]]


@dataclass(frozen=True)
class Diagnostic:
    code: str
    query_id: str
    message: str


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float]
    per_query: dict[str, dict[str, float]]
    diagnostics: tuple[Diagnostic, ...]
    k: tuple[int, ...]
    query_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.0", "kind": "evaluation", **asdict(self)}


@dataclass(frozen=True)
class Threshold:
    metric: str
    min_value: float | None = None
    max_drop: float | None = None
    group: str | None = None


@dataclass(frozen=True)
class GateResult:
    metric: str
    group: str | None
    baseline: float
    candidate: float
    min_value: float | None
    max_drop: float | None
    passed: bool


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    confidence: float
    samples: int


@dataclass(frozen=True)
class ComparisonResult:
    baseline: EvaluationResult
    candidate: EvaluationResult
    delta: dict[str, float]
    per_query_delta: dict[str, dict[str, float]]
    groups: dict[str, dict[str, Any]]
    gates: tuple[GateResult, ...]
    passed: bool
    confidence_intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.0", "kind": "comparison", **asdict(self)}
