# Evaluation protocol and assumptions

## Query universe

Let U be all query IDs in the validated qrels. Every aggregate metric is the arithmetic mean
over U. A missing or empty ranking contributes zero. Both runs use exactly U. Unknown run
query IDs are rejected. Each query must have at least one positive judgment; queries without
one are rejected rather than silently dropped or assigned an arbitrary metric.

## Rankings and metrics

Rank by descending finite score, then ascending document ID for ties. Let rel(d) be the
judged grade (zero for unjudged documents), and let relevant mean rel(d) > 0.

- Recall@K = number of relevant retrieved documents in the first K / all judged relevant documents.
- Precision@K = number of relevant retrieved documents in the first K / K, even for a short ranking.
- MRR@K = reciprocal rank of the first relevant document within K, or zero if none exists.
- DCG@K = sum of rel(d_i) / log2(i+1), i starting at 1. IDCG sorts all judgments by grade.
  NDCG@K = DCG / IDCG. Gains are linear, not `2**rel - 1`. Scaling all grades by their
  maximum internally avoids overflow without changing the ratio.

Do not remove query/document self matches by default. Relevance grades are supplied by the
user; their correctness and completeness are not inferred by this library.

## Input integrity

Reject duplicate JSON keys, empty/invalid IDs, negative grades, booleans, NaN/Infinity,
unknown queries, malformed mappings, unknown config fields, and uncomputed threshold metrics.
Duplicate IDs already overwritten in Python dictionaries cannot be recovered.

Unjudged documents are treated as non-relevant and reported. This is an evaluation convention,
not evidence that they are actually irrelevant. Incompletely pooled judgments may favor some
retrievers; report this limitation in external experiments.

## Comparisons and categories

All deltas are candidate minus baseline. Category results macro-average the queries belonging
to that category, not documents or category sizes. Groups are user supplied; the library does
not invent query categories or automatically diagnose causes. Overall improvement can coexist
with category regressions. Choose important categories and thresholds before inspecting a
held-out result when using the outcome for a research claim.

## Quality gates

A gate can require a minimum candidate value and/or limit the absolute decline from baseline.
Every configured bound must pass, allowing a numerical tolerance of 1e-12. An unknown metric
or category is an input error. Gates are deterministic policies, not statistical tests.

## Paired bootstrap

For each metric, compute one difference per query. Draw Q differences with replacement,
average them, and repeat B times. Report interpolated percentile bounds at the configured
confidence level. The same seed reproduces the sampled query indices across metrics.

Assumptions: queries are reasonably independent and representative of the target query
population. Correlated questions, tiny categories, benchmark selection, model pretraining
contamination, and repeated tuning can invalidate broad interpretations. Current intervals
are overall only, do not correct multiple comparisons, do not establish causality, and never
change gate decisions. Small B gives unstable tail estimates; the toy uses B=500 only for a demo.

## Independent correctness checks

`tests/test_reference.py` compares seeded, untied graded rankings with `ir-measures`, including
cutoffs longer than a ranking. Install `.[reference]` to run it. Missing-query denominators,
empty rankings, ties, and malformed inputs are checked separately with hand-calculated tests
because external tools differ in those policies. No agreement claim extends beyond tested cases.

References:
- [BEIR evaluation implementation](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)
- [ir-measures documentation](https://ir-measur.es/en/latest/)
- [trec_eval](https://github.com/usnistgov/trec_eval)
