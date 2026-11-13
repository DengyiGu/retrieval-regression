"""Retrieval metrics over a fixed, explicit qrels query universe."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .models import Diagnostic, EvaluationResult
from .validation import validate_k, validate_qrels, validate_run


def evaluate(
    qrels: Mapping[str, Mapping[str, float]],
    run: Mapping[str, Mapping[str, float]],
    *,
    k: Sequence[int] = (5, 10),
    negative_grades: str = "error",
    no_positive: str = "error",
    tie_breaker: str = "doc_id_asc",
) -> EvaluationResult:
    """Macro-average over ALL qrels queries, including missing/empty run queries.

    Ties default to ascending IDs; descending IDs are optional.
    Positive grades are relevant for binary metrics.
    NDCG uses linear graded gain, matching trec_eval's ndcg_cut convention.
    """
    cutoffs = validate_k(k)
    if negative_grades not in {"error", "clip", "drop"}:
        raise ValueError("negative_grades must be error, clip or drop")
    if no_positive not in {"error", "keep", "drop"}:
        raise ValueError("no_positive must be error, keep or drop")
    if tie_breaker not in {"doc_id_asc", "doc_id_desc"}:
        raise ValueError("tie_breaker must be doc_id_asc or doc_id_desc")
    original = validate_qrels(
        qrels, allow_negative=negative_grades != "error", require_positive=False
    )
    judgments = {}
    policy_diagnostics: list[Diagnostic] = []
    for qid, docs in original.items():
        clean = {}
        for doc, grade in docs.items():
            if grade < 0:
                policy_diagnostics.append(
                    Diagnostic(
                        "negative_grade_" + negative_grades,
                        qid,
                        f"Document {doc!r}, grade {grade}: {negative_grades}",
                    )
                )
                if negative_grades == "drop":
                    continue
                grade = 0.0
            clean[doc] = grade
        if not any(grade > 0 for grade in clean.values()):
            if no_positive == "error":
                raise ValueError(f"qrels[{qid!r}] must contain at least one positive judgment")
            policy_diagnostics.append(
                Diagnostic(
                    "no_positive_" + no_positive,
                    qid,
                    "No positive judgments: "
                    + (
                        "excluded from denominator"
                        if no_positive == "drop"
                        else "included with all metrics zero"
                    ),
                )
            )
            if no_positive == "drop":
                continue
        judgments[qid] = clean
    if not judgments:
        raise ValueError("qrels policies excluded every query")
    # Validate against the original universe before removing explicitly dropped topics.
    scores, diagnostics = validate_run(run, original)
    diagnostics = tuple(policy_diagnostics) + tuple(
        d for d in diagnostics if d.query_id in judgments
    )
    per_query: dict[str, dict[str, float]] = {}
    for qid in sorted(judgments):
        docs = judgments[qid]
        ranking = sorted(scores.get(qid, {}), reverse=tie_breaker == "doc_id_desc")
        ranking.sort(key=lambda doc: -scores[qid][doc])
        positive_count = sum(grade > 0 for grade in docs.values())
        scale = max(docs.values(), default=0.0) or 1.0
        ideal = sorted((grade / scale for grade in docs.values()), reverse=True)
        values: dict[str, float] = {}
        for cutoff in cutoffs:
            top = ranking[:cutoff]
            hits = [rank for rank, doc in enumerate(top, 1) if docs.get(doc, 0) > 0]
            dcg = math.fsum(
                docs.get(doc, 0) / scale / math.log2(rank + 1) for rank, doc in enumerate(top, 1)
            )
            idcg = math.fsum(
                grade / math.log2(rank + 1) for rank, grade in enumerate(ideal[:cutoff], 1)
            )
            values[f"recall@{cutoff}"] = len(hits) / positive_count if positive_count else 0.0
            values[f"precision@{cutoff}"] = len(hits) / cutoff
            values[f"mrr@{cutoff}"] = 1 / hits[0] if hits else 0.0
            values[f"ndcg@{cutoff}"] = min(1.0, dcg / idcg) if idcg else 0.0
        per_query[qid] = values
    metrics = next(iter(per_query.values()))
    summary = {
        metric: math.fsum(values[metric] for values in per_query.values()) / len(per_query)
        for metric in metrics
    }
    return EvaluationResult(
        summary,
        per_query,
        diagnostics,
        cutoffs,
        len(per_query),
        len(original),
        {
            "negative_grades": negative_grades,
            "no_positive": no_positive,
            "tie_breaker": tie_breaker,
        },
    )
