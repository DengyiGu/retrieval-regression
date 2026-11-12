"""Input validation shared by the Python API and JSON readers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .models import Diagnostic, Qrels, Run


def _identifier(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: IDs must be non-empty strings")
    return value


def _number(value: Any, location: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{location}: expected a finite number, got {value!r}")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{location}: expected a finite number") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(
            f"{location}: expected a finite {'non-negative ' if nonnegative else ''}number"
        )
    return number


def validate_qrels(value: Any) -> Qrels:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("qrels must be a non-empty mapping of query IDs to relevance judgments")
    result: Qrels = {}
    for qid, judgments in value.items():
        _identifier(qid, "qrels query")
        if not isinstance(judgments, Mapping):
            raise ValueError(f"qrels[{qid!r}] must be a mapping")
        docs = {
            _identifier(doc, f"qrels[{qid!r}] document"): _number(
                grade, f"qrels[{qid!r}][{doc!r}]", nonnegative=True
            )
            for doc, grade in judgments.items()
        }
        if not any(grade > 0 for grade in docs.values()):
            raise ValueError(f"qrels[{qid!r}] must contain at least one positive judgment")
        result[qid] = docs
    return result


def validate_run(value: Any, qrels: Qrels) -> tuple[Run, tuple[Diagnostic, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError("run must be a mapping of query IDs to document scores")
    result: Run = {}
    diagnostics: list[Diagnostic] = []
    for qid, scores in value.items():
        _identifier(qid, "run query")
        if qid not in qrels:
            raise ValueError(f"run contains unknown query ID {qid!r}")
        if not isinstance(scores, Mapping):
            raise ValueError(f"run[{qid!r}] must be a mapping")
        result[qid] = {
            _identifier(doc, f"run[{qid!r}] document"): _number(score, f"run[{qid!r}][{doc!r}]")
            for doc, score in scores.items()
        }
    for qid in sorted(qrels):
        if qid not in result:
            diagnostics.append(Diagnostic("missing_query", qid, "Missing query scored as zero"))
        elif not result[qid]:
            diagnostics.append(Diagnostic("empty_ranking", qid, "Empty ranking scored as zero"))
        unjudged = set(result.get(qid, {})) - set(qrels[qid])
        if unjudged:
            diagnostics.append(
                Diagnostic(
                    "unjudged_documents",
                    qid,
                    f"{len(unjudged)} retrieved documents have no judgment; "
                    "treated as non-relevant",
                )
            )
    return result, tuple(diagnostics)


def validate_k(k: Sequence[int]) -> tuple[int, ...]:
    if isinstance(k, (str, bytes)) or not k:
        raise ValueError("k must contain at least one positive integer")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in k):
        raise ValueError("k must contain only positive integers")
    return tuple(sorted(set(k)))
