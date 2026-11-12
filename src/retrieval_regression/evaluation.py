"""Retrieval metrics over a fixed, explicit qrels query universe."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .models import EvaluationResult
from .validation import validate_k, validate_qrels, validate_run


def evaluate(
    qrels: Mapping[str, Mapping[str, float]],
    run: Mapping[str, Mapping[str, float]],
    *,
    k: Sequence[int] = (5, 10),
) -> EvaluationResult:
    """Macro-average over ALL qrels queries, including missing/empty run queries.

    Ties use ascending document IDs. Positive grades are relevant for binary metrics.
    NDCG uses linear graded gain, matching trec_eval's ndcg_cut convention.
    """
    cutoffs = validate_k(k)
    judgments = validate_qrels(qrels)
    scores, diagnostics = validate_run(run, judgments)
    per_query: dict[str, dict[str, float]] = {}
    for qid in sorted(judgments):
        docs = judgments[qid]
        ranking = sorted(scores.get(qid, {}), key=lambda doc: (-scores[qid][doc], doc))
        positive_count = sum(grade > 0 for grade in docs.values())
        scale = max(docs.values())
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
            values[f"recall@{cutoff}"] = len(hits) / positive_count
            values[f"precision@{cutoff}"] = len(hits) / cutoff
            values[f"mrr@{cutoff}"] = 1 / hits[0] if hits else 0.0
            values[f"ndcg@{cutoff}"] = min(1.0, dcg / idcg)
        per_query[qid] = values
    metrics = next(iter(per_query.values()))
    summary = {
        metric: math.fsum(values[metric] for values in per_query.values()) / len(per_query)
        for metric in metrics
    }
    return EvaluationResult(summary, per_query, diagnostics, cutoffs, len(per_query))
