# Evaluation protocol and assumptions

## Query universe

Let U be all query IDs in the validated qrels. Every aggregate metric is the arithmetic mean
over U. A missing or empty ranking contributes zero. Both runs use exactly U. Unknown run
query IDs are rejected. The strict default requires a positive judgment. Explicit `negative_grades="clip"|"drop"`
handles negative markers before `no_positive="keep"|"drop"` handles zero-positive topics.
Keeping assigns all four metrics zero; dropping excludes the topic from both runs and groups.
Both input and evaluated counts and per-topic diagnostics are retained. A run query outside
the original qrels is still an error. Topics absent from qrels cannot be inferred; supply an
explicit empty judgment mapping for them if they belong in your chosen universe.

## Rankings and metrics

Rank by descending finite score, then ascending document ID for ties by default. `tie_breaker="doc_id_desc"` uses descending
IDs, matching trec_eval's tie order. Tied scores can otherwise produce different metrics from
trec_eval / the ir-measures pytrec_eval provider. TREC declared ranks do not override scores. Let rel(d) be the
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

Reject duplicate JSON keys, empty/invalid IDs, negative grades under the default policy, booleans, NaN/Infinity,
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

A gate requires a minimum candidate value and/or limits absolute decline from baseline.
Tolerance is 1e-12. Unknown metrics/categories are errors. `gate_mode="point"` (default)
applies bounds to observed means; descriptive intervals do not affect this mode.

`gate_mode="bootstrap"` applies bounds to percentile intervals. A `max_drop=d` bound examines
the paired delta distribution against -d; `min_value=m` examines the candidate mean against m.
For each bound, a lower endpoint >= boundary passes, an upper endpoint < boundary fails,
and an overlapping interval is inconclusive. Any failed bound makes the gate fail; otherwise
all bounds must pass. Fewer than `min_queries=20` in that gate's actual group yields
inconclusive, even for zero-width intervals. This floor is configurable (>=2), not a guarantee.

Overall status is fail if any gate fails, otherwise inconclusive if any gate is inconclusive,
otherwise pass. No gates yields `status="not_gated"` and legacy `passed=True`.
An inconclusive comparison has `passed=False`; CLI exit 3 and pytest failure block by default.
Explicit `on_inconclusive="warn"` permits continuation while retaining the inconclusive verdict.
A mix of failed and inconclusive gates still fails; warning mode never suppresses failures.

## Paired bootstrap and multiplicity

For each metric, draw Q paired query differences with replacement, average, repeat B times,
and interpolate percentile endpoints. Candidate minimum bounds resample candidate query values.
Groups are resampled within their own query sets. The same seed uses the same sampled query
indices across metrics of the same size. At least 1,000 samples are required for statistical
gates; 5,000+ is preferable for tail stability, particularly with many bounds.

Descriptive overall and group intervals use the requested confidence without adjustment.
Gate evidence defaults to `multiple_testing="bonferroni"`: with H configured bounds
(overall + groups, minimum + drop), each gate interval uses confidence 1-(1-confidence)/H.
`multiple_testing="none"` disables this adjustment. This nominal adjustment does not require
independent metrics/groups, but practical coverage still depends on bootstrap approximation.
It does not control repeated CI runs, model selection or post-hoc thresholds.
The library reports intervals, not p-values or exact false-alarm control.

Queries should be independent and representative sampling units. Correlated questions,
sparse/degenerate samples, benchmark selection, pretraining contamination and repeated tuning
can make percentile intervals too optimistic. Twenty observations are not automatically enough;
identical observed runs establish agreement on this sample, not on unseen queries.
Failure to find a regression does not prove non-inferiority. See the [calibration and data
evidence](EVIDENCE.md), including the large inconclusive fraction in the small A/A simulation.

## Independent correctness checks

`tests/test_reference.py` compares seeded, untied graded rankings with `ir-measures`, including
cutoffs longer than a ranking. Install `.[reference]` to run it. Missing-query denominators,
empty rankings, ties, and malformed inputs are checked separately with hand-calculated tests
because external tools differ in those policies. A separate tied-score test checks the optional
descending-ID mode against the pytrec_eval provider. No agreement claim extends beyond tested cases.

References:
- [BEIR evaluation implementation](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)
- [ir-measures documentation](https://ir-measur.es/en/latest/)
- [trec_eval](https://github.com/usnistgov/trec_eval)
