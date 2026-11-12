# API and data formats

## Inputs

```json
{"q1": {"doc-a": 2, "doc-b": 0}, "q2": {"doc-c": 1}}
```

Qrels map query IDs to document relevance grades. IDs must be non-empty strings. Grades must
be finite, real, non-negative numbers (booleans rejected). Every query needs a positive grade.
The project recommends small integer grades for interoperability with reference tools.

Runs use the same nested shape, with finite retrieval scores instead of grades:

```json
{"q1": {"doc-b": 0.95, "doc-a": 0.8}, "q2": {"doc-c": 0.7}}
```

Scores need not be probabilities and may be negative. Missing queries and empty dictionaries
are valid failures and score zero. Unknown query IDs are errors. Unjudged document IDs are
valid and generate diagnostics. Mappings cannot preserve duplicate document IDs: duplicate
JSON keys are rejected during loading, but a Python dict whose duplicates were already
overwritten cannot be diagnosed retrospectively. List-style rankings are not accepted.

## Evaluation

```python
from retrieval_regression import evaluate

result = evaluate(qrels, run, k=[1, 5, 10])
```

Returns `EvaluationResult` with `summary`, `per_query`, `diagnostics`, normalized `k`, and
`query_count`. Metric keys are lower-case `recall@K`, `precision@K`, `mrr@K`, `ndcg@K`.
`result.to_dict()` adds `schema_version="1.0"` and `kind="evaluation"`.

## Comparison

```python
from retrieval_regression import Threshold, compare

result = compare(
    qrels,
    baseline,
    candidate,
    k=[5, 10],
    groups={"q1": "api", "q2": "setup"},
    thresholds=[Threshold("recall@10", min_value=0.8, max_drop=0.02)],
    bootstrap_samples=1000,
    confidence=0.95,
    seed=42,
)
```

`ComparisonResult`: `baseline`, `candidate`, `delta`, `per_query_delta`, `groups`, `gates`,
`passed`, `confidence_intervals`, `seed`. `to_dict()` uses `kind="comparison"` and schema 1.0.
Input dictionaries are not mutated. Returned nested dictionaries are regular Python objects;
do not modify results before writing reports if you want to preserve the original evidence.

Categories are mutually exclusive in this alpha: each query maps to one string. When groups
are supplied, missing category labels go into `__ungrouped__`; reserve this name for that bucket.

`Threshold(metric, min_value=None, max_drop=None, group=None)` requires at least one bound.
Bounds are finite and within [0, 1]. `max_drop` is an absolute difference. Both bounds must
hold when both are supplied. A 1e-12 tolerance avoids boundary failures due to roundoff.
A group-scoped threshold needs an existing category; metric cutoffs must be included in `k`.
Unknown thresholds fail closed. No thresholds means `passed=True`, but no quality was enforced.

## JSON threshold configuration

```json
{
  "thresholds": [
    {"metric": "recall@10", "min_value": 0.8, "max_drop": 0.02},
    {"metric": "ndcg@10", "max_drop": 0.01, "group": "api"}
  ]
}
```

Unknown keys are rejected, including spelling mistakes in bound names. Groups use a separate
JSON object of query ID to category. JSON files may contain a UTF-8 BOM.

## Reports and CLI

`write_json(result, path)` supports evaluation and comparison results. `write_html(result,
path)` supports comparisons. Parent directories are created as needed. Existing output files
are replaced. Text is escaped in HTML. JSON includes per-query evidence, diagnostics, gates,
category summaries, and optional confidence intervals.

Run `python -m retrieval_regression --help`, then `compare --help` or `evaluate --help`.
Comparison flags: `--qrels`, `--baseline`, `--candidate`, `--k`, `--config`, `--groups`,
`--json`, `--html`, `--bootstrap-samples`, `--confidence`, `--seed`.
Evaluation flags: `--qrels`, `--run`, `--k`, `--json`.
The CLI always prints result JSON on a valid invocation. Exit 1 is an expected result for a
failed gate, not a program crash. Invalid data/configuration yields exit 2 and an error message.

The HTML gate status is `NOT CONFIGURED` when no thresholds were supplied. The existing Python
`passed=True` value in that case means no gate failed; inspect `gates` before treating it as a
quality verdict. CLI stdout uses ASCII JSON escapes so Unicode IDs survive legacy terminal
encodings; report files retain UTF-8 text. CLI output paths must differ from input files and
from each other, including symbolic-link and hard-link aliases present at validation time.
