"""Batch reranking through a caller-supplied scorer, without a model dependency."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from .models import Run
from .validation import _number, validate_run_structure

Scorer = Callable[[Sequence[tuple[str, str]]], Iterable[float]]


def rerank(
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
    run: Mapping[str, Mapping[str, float]],
    scorer: Scorer,
    *,
    top_k: int = 50,
    batch_size: int = 32,
) -> Run:
    """Score the top candidates; return only that candidate set with new finite scores.

    Candidate ordering uses score descending then document ID ascending. Original scores are
    never mixed with model scores. Missing queries stay missing; empty rankings stay empty.
    Every selected text is checked before calling the scorer. The caller owns model loading,
    text truncation, device placement and external scorer configuration.
    """
    for name, value in (("top_k", top_k), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(queries, Mapping) or not isinstance(corpus, Mapping):
        raise ValueError("queries and corpus must be mappings of IDs to text")
    if not callable(scorer):
        raise ValueError("scorer must be callable")
    scores = validate_run_structure(run)
    tasks: list[tuple[str, str, str, str]] = []
    result: Run = {}
    for qid in sorted(scores):
        if qid not in queries:
            raise ValueError(f"run contains unknown query ID {qid!r}")
        query = queries[qid]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query {qid!r} needs non-empty text")
        result[qid] = {}
        selected = sorted(scores[qid], key=lambda doc: (-scores[qid][doc], doc))[:top_k]
        for doc in selected:
            if doc not in corpus:
                raise ValueError(f"missing corpus text for document {doc!r}")
            text = corpus[doc]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"document {doc!r} needs non-empty text")
            tasks.append((qid, doc, query, text))
    for offset in range(0, len(tasks), batch_size):
        batch = tasks[offset : offset + batch_size]
        output = scorer([(query, text) for _, _, query, text in batch])
        try:
            values = list(output)
        except TypeError as exc:
            raise ValueError("scorer must return an iterable of finite scores") from exc
        if len(values) != len(batch):
            raise ValueError(f"scorer returned {len(values)} scores for {len(batch)} pairs")
        for (qid, doc, _, _), score in zip(batch, values, strict=True):
            result[qid][doc] = _number(score, f"scorer[{qid!r}][{doc!r}]")
    return result
