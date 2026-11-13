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


## File-format profiles

`load_qrels(path, *, format="json")` and `load_run(path, *, format="json")` return validated
nested mappings and are exported at package level. They do not guess from a filename extension.
JSON retains its existing fractional-grade support. `load_run` checks structure; unknown query
IDs are rejected later by `evaluate`/`compare` against the selected qrels universe.

| Input | Format name | Accepted rows |
| --- | --- | --- |
| JSON qrels/run | `json` (default) | Existing nested mapping contract |
| BEIR qrels | `beir` | Exact header `query-id<TAB>corpus-id<TAB>score`, then three tab-separated fields |
| TREC qrels | `trec` | Four whitespace-separated fields: `qid 0 docid grade` |
| TREC run | `trec` | Six whitespace-separated fields: `qid Q0 docid rank score tag` |

Text formats accept UTF-8 with optional BOM, LF/CRLF, blank lines, interleaved queries and
out-of-order rows. IDs are preserved as strings and must be whitespace-free tokens. Comments,
quoted/multiline fields and format guessing are not supported. BEIR/TREC judgment grades must
be non-negative decimal integers; negative unjudged markers such as `-1` are rejected rather
than silently relabeled. Every judged query still requires a positive grade.

Run scores may be finite negative or exponential-notation numbers. Rank labels must be positive
1-based decimal integers and unique within a query. Gaps are allowed, for example a filtered
export retaining ranks 2 and 8. Scores must be non-increasing when sorted by declared rank.
The whole file must use one tag. Repeated query/document pairs are always errors, even when
values agree. Empty run files are valid and score all judged queries as missing.

Ranks are validation metadata. The evaluator always orders by score descending, then document
ID ascending; declared ranks do not override a tied score. This strict profile rejects certain
historical TREC files accepted by permissive tools (zero-based/repeated ranks, mixed tags,
nonzero qrels iteration, or score/rank contradictions). Convert those files explicitly under a
known policy rather than expecting silent repairs.

`InputFormatError` is a public `ValueError` subclass with `path`, `line_number` and `message`.
Row failures include physical file/line coordinates; whole-file failures may have no line.
Low-level parser functions are also available in `retrieval_regression.adapters`:
`read_beir_qrels`, `read_trec_qrels`, `read_trec_run`.

CLI flags: `--qrels-format {json,beir,trec}` on both commands; `--run-format {json,trec}` on
`evaluate`; independent `--baseline-format` and `--candidate-format` on `compare`. Defaults
remain JSON. Groups and threshold configuration remain JSON.

Sources: [BEIR loader](https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py),
[NIST TREC run guidelines](https://ir.nist.gov/trec-covid/round5.html). This library's strict
validation policy is documented above; NIST evaluation itself sorts by scores and ignores ranks.

## Evaluation manifests

Both CLI commands accept `--manifest PATH`. This is a separate JSON artifact with schema 1.0,
kind `evaluation_manifest`; existing result schema 1.0 is unchanged. It records input roles,
base filenames, formats, byte lengths and SHA-256 hashes, package/Python versions, effective
cutoffs, bootstrap configuration when applicable, query count and whether gates were configured.
Config and group files are fingerprinted too. The `result_sha256` hashes the result serialized
as UTF-8 JSON with sorted keys, ASCII escapes and compact `,`/`:` separators.

Absolute input paths are excluded. A manifest is evidence about supplied files and evaluator
settings; it does not embed those files or automatically capture the external retriever/model
configuration. Retain source inputs and model configuration separately.

Inputs are hashed before parsing and rehashed after evaluation, before any report is written.
A content mismatch aborts with exit 2. These before/after checks detect changing inputs in normal
workflows; they do not provide an atomic snapshot against adversarial concurrent writers.
Manifest destinations participate in the same input/output alias protection as JSON and HTML.
The extra file passes occur only when a manifest is requested.


## Batched neural or custom reranking

```python
from retrieval_regression import rerank

new_run = rerank(queries, corpus, baseline_run, scorer, top_k=50, batch_size=32)
```

`queries` and `corpus` map string IDs to text. `scorer` receives a sequence of `(query_text,
document_text)` pairs and returns an iterable with exactly one finite real score per pair.
External model loading, device choice and truncation belong to the scorer. Generators and
NumPy score arrays are accepted; tensor-returning scorers should convert outputs to numbers.

The top `top_k` candidates per query are chosen using original scores and the documented ID
tie break. Scoring batches can span queries. The returned `Run` contains only these candidates
with new scores; it is a mapping, not a sorted list. Query omissions and empty rankings retain
their meaning, and the original inputs are not mutated. Reranking cannot retrieve relevant
documents absent from its candidate pool. Keep the same qrels universe during evaluation.

Selected text is checked before model calls; short/nonfinite scorer output raises a clear error,
while exceptions raised inside the external scorer propagate. `top_k` and `batch_size` must be
positive integers. The core imports no torch, Transformers, or hosted-model SDK.

The source checkout includes optional `experiments.reranking` training/inference and
`examples.neural_search` modules. Their models and datasets are opt-in local resources; they
are research examples, not provider integrations installed with the runtime wheel.
