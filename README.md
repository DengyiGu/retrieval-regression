# Retrieval Regression

**Catch retrieval quality regressions before they reach your RAG application.**

A local-first Python library for comparing search results, finding query-level regressions,
and enforcing quality thresholds in automated tests. No paid APIs, LLM judges, GPU, or runtime
dependencies are required for the core library.

**Status:** working alpha (`0.3.0`). Install from this repository; no PyPI release is claimed.

## The problem

You change an embedding model, chunking strategy, index, or reranker. The overall score improves,
but an important query category gets worse. Or a run silently omits difficult queries and the
reported average becomes misleading.

Retrieval Regression evaluates both runs over the **same complete set of judged queries**,
shows what changed, and lets you fail a test when a configured quality budget is exceeded.
It evaluates retrieved document rankings, not generated-answer correctness.

## When to use it

Use this library when you already have relevance judgments and exported baseline/candidate
rankings, and want a repeatable pass/fail check in CI. For example, accept an index change only
if overall NDCG@10 drops by no more than 0.02 and an important query category keeps Recall@10
above its minimum. Keep corpus IDs and judgment semantics stable across the two runs.

Established tools such as [ir-measures](https://ir-measur.es/en/latest/) already provide a much
broader metric catalog, and [BEIR](https://github.com/beir-cellar/beir) provides datasets and
retrieval benchmarks. This project's focus is a small regression-testing workflow: strict inputs,
paired differences, category gates, pytest failures, and an offline report. The metric formulas
are standard. If you only need metric calculation, an existing evaluator may be sufficient.

Changing chunk boundaries or document IDs requires corresponding judgment mapping; this library
does not create that mapping. It also needs external judgments and cannot assess generated answers.

## Install from source

Python 3.10+ is required. These commands work in Windows PowerShell and POSIX shells:

```console
git clone https://github.com/DengyiGu/retrieval-regression.git
cd retrieval-regression
python -m venv .venv
```

On Windows, use the environment directly without changing PowerShell execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,reference]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe examples/quickstart.py
```

On Linux/macOS:

```bash
.venv/bin/python -m pip install -e '.[dev,reference]'
.venv/bin/python -m pytest -q
.venv/bin/python examples/quickstart.py
```

For runtime-only use, install with `python -m pip install .`. To use the pytest fixture,
install `python -m pip install '.[pytest]'`. Installation downloads dependencies unless they
are cached; evaluation and the bundled examples run offline after installation.

## Python API

```python
from retrieval_regression import Threshold, compare
from retrieval_regression.reporting import write_html

qrels = {"install": {"doc-a": 2}, "api": {"doc-b": 1}}
baseline = {"install": {"doc-a": 0.9}, "api": {"wrong": 0.9, "doc-b": 0.8}}
candidate = {"install": {"doc-a": 0.9}, "api": {"doc-b": 0.9}}

result = compare(
    qrels,
    baseline,
    candidate,
    k=[1, 5],
    groups={"install": "setup", "api": "reference"},
    thresholds=[Threshold("recall@5", min_value=0.9, max_drop=0.02)],
    bootstrap_samples=1000,
    seed=42,
)

print(result.delta)  # candidate minus baseline
print(result.passed)  # True
print(result.per_query_delta["api"])
write_html(result, "reports/comparison.html")
```

Use `evaluate(qrels, run, k=[1, 5])` for a single run. See [API and data formats](docs/API.md).

## Command line

After activating the environment, `rr` and `retrieval-regression` are aliases for the same CLI.
You can also invoke it with the environment's Python:

```console
python -m retrieval_regression compare --qrels examples/toy/qrels.json --baseline examples/toy/baseline.json --candidate examples/toy/candidate.json --groups examples/toy/groups.json --config examples/toy/gates.json --k 1 3 --json reports/comparison.json --html reports/comparison.html
```

Use the virtual environment's Python path if it is not activated. The command writes reports
and prints JSON to stdout. Replace `candidate.json` with `regressed.json` to demonstrate a
failed gate. Exit codes: **0** success, **1** failed quality gate, **2** invalid input/configuration.
No configured thresholds means comparison only, not quality enforcement.

## Standard retrieval files and reproducible runs

Load an existing BEIR test split and TREC exports directly:

```python
from retrieval_regression import load_qrels, load_run, compare

qrels = load_qrels("examples/formats/qrels.tsv", format="beir")
baseline = load_run("examples/formats/baseline.run", format="trec")
candidate = load_run("examples/formats/candidate.run", format="trec")
result = compare(qrels, baseline, candidate, k=[1, 3])
```

The CLI accepts separate formats for each input, so migration can mix JSON and TREC runs:

```console
python -m retrieval_regression compare --qrels examples/formats/qrels.tsv --qrels-format beir --baseline examples/formats/baseline.run --baseline-format trec --candidate examples/formats/candidate.run --candidate-format trec --config examples/toy/gates.json --groups examples/toy/groups.json --k 1 3 --json reports/comparison.json --html reports/comparison.html --manifest reports/manifest.json
```

Parsers reject duplicate judgments/documents/ranks, mixed run tags, invalid headers and
non-finite scores with file/line diagnostics. Ranks are checked against scores; scoring still
uses descending scores and ascending document IDs for ties. The adapters intentionally support
[a documented strict profile](docs/API.md#file-format-profiles), rather than every historical
TREC variant. Missing queries remain in the evaluation denominator.

An optional manifest records SHA-256/size/format for every input (including gates and groups),
evaluator/Python versions, parameters and a canonical result fingerprint. It contains base
filenames rather than machine-specific absolute paths. Input contents are rechecked before
reports are written. Keep the original input files with the manifest to reproduce a run.

## Neural reranking and local model experiments

`rerank(queries, corpus, run, scorer, top_k=50, batch_size=32)` connects candidate rankings to
any pair scorer that returns one numeric score per query/document text pair. It has no model
provider dependency. The optional [neural experiment](experiments/reranking/README.md) provides
an actual Transformer scorer, model fine-tuning and local search:

```console
python -m pip install -e '.[benchmark,neural]'
python -m experiments.reranking.run --download
python -m examples.neural_search --query "Does smoking increase cancer risk?" --dataset scifact
```

Install a suitable CUDA PyTorch build for GPU use. Downloads are opt-in; after caching, model
inference/training runs locally. The experiment compares a pinned pretrained MiniLM reranker
with random-negative and hard-negative fine-tuning, selects on a training-derived validation
split, and reports every variant on SciFact and NFCorpus held-out tests.

[The measured neural report](docs/reports/2026-11-12-neural.md) includes actual training,
model-quality and GPU-memory results. Small single-seed experiments support local observations;
they do not establish universal gains. Dataset and pretrained-weight terms remain separate
from the code license; weights and raw texts are not redistributed.

## Download the adapted model

The [v0.3.0 model release](https://github.com/DengyiGu/retrieval-regression/releases/tag/v0.3.0)
provides `scifact-minilm-reranker-v0.3.0.zip` and `SHA256SUMS`. Extract the archive to obtain
`scifact-minilm-reranker/`; weights and tokenizer are complete, so retraining is optional.
See the [model card](experiments/reranking/MODEL_CARD.md) for loading, attribution and limitations.
With the corpus cached, search using the downloaded checkpoint:

```console
python -m examples.neural_search --checkpoint scifact-minilm-reranker --query "your English query"
```

Code and experiments stay in this repository; model binaries are Release assets rather than
Git history. The released seed-13 checkpoint was selected on validation after one additional
seed check. Both checked runs improved SciFact NDCG@10; broader improvements are not established.

## pytest integration

The optional plugin registers the `retrieval_check` fixture:

```python
from retrieval_regression import Threshold


def test_retrieval_quality(retrieval_check):
    retrieval_check(
        {"q": {"relevant-doc": 1}},
        {"q": {"relevant-doc": 1.0}},
        {"q": {"relevant-doc": 1.0}},
        k=[5],
        thresholds=[Threshold("recall@5", max_drop=0.02)],
    )
```

The fixture requires an explicit threshold and fails with metric details when a gate fails.
Try `python -m pytest examples/test_retrieval_quality.py -q`.

## Available now

- Recall@K, Precision@K, MRR@K, and graded NDCG@K.
- Batched, model-independent reranking with caller-supplied scoring.
- Optional local neural inference, controlled fine-tuning ablation and a usable search example.
- Fixed-query macro averaging, including missing and empty rankings as zero.
- Strict JSON, original BEIR qrels TSV, TREC qrels and TREC run loading.
- File/line errors for duplicate rows, invalid ranks, mixed tags and score/rank conflicts.
- Optional input/result fingerprints for portable, auditable evaluation records.
- Deterministic tie handling and diagnostics for unjudged retrieved documents.
- Overall and user-defined category comparisons, with absolute quality-drop budgets.
- Optional seeded paired-bootstrap intervals for overall metric deltas.
- Local HTML/JSON reports, Python API, CLI, and pytest integration.
- Hand-calculated tests and optional independent checks against `ir-measures`.

## Evaluation conventions

- Every query in qrels is evaluated. Each must have at least one positive judgment.
- Retrieval scores sort descending; ties break by ascending document ID.
- Positive relevance grades count as relevant for binary metrics.
- NDCG uses **linear graded gain**, matching `trec_eval`'s `ndcg_cut` convention.
- Unjudged documents count as non-relevant; incomplete judgments limit interpretation.
- Precision divides by K even if fewer than K documents are returned.
- Unknown query IDs and invalid configuration raise errors instead of being silently ignored.
- `max_drop=0.02` means **two absolute percentage points**, not a 2% relative change.

Full definitions, limitations, and statistical assumptions: [evaluation protocol](docs/EVALUATION.md).

## Project map

```text
src/retrieval_regression/    Metrics, validation, comparisons, reports, CLI, pytest plugin
tests/                      Correctness, integration, and independent-reference checks
examples/                   Offline toy data and runnable usage
benchmarks/                 SciFact validation and synthetic evaluator benchmark
docs/                       Requirements, architecture, research protocol and roadmap
```

## Development and research

Start with [development setup](docs/DEVELOPMENT.md), [requirements](docs/REQUIREMENTS.md),
and the [roadmap](docs/ROADMAP.md). Run `python scripts/check.py` after changes.
The synthetic example demonstrates behavior; **it is not evidence of real-world retrieval gains**.
No benchmark speedups, model improvements, or external adoption are claimed.

A public SciFact experiment is included in `benchmarks/scifact.py`; install `.[benchmark]`
and run `python benchmarks/scifact.py --download` once, then rerun offline without that flag.
See [the validation report](docs/reports/2026-11-12-validation.md) for measured outcomes and
[the follow-up review](docs/reports/2026-11-12-review.md) for audit fixes and remaining limits.
The [format audit](docs/reports/2026-11-12-formats.md) checks original BEIR judgments and
60,000 exported TREC rows against JSON results and an independent evaluator. It also records
synthetic measurements up to 10,000 queries. Parsers retain full mappings in memory;
large-run streaming and broader model validation remain on the roadmap. A separate
[neural experiment](docs/reports/2026-11-12-neural.md) adds SciFact/NFCorpus evaluation.
See [research protocol](docs/RESEARCH.md).

## Maintainer

[Dengyi Gu](https://github.com/DengyiGu).

## License

[MIT](LICENSE).
