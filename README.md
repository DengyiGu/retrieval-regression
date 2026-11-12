# Retrieval Regression

**Catch retrieval quality regressions before they reach your RAG application.**

A local-first Python library for comparing search results, finding query-level regressions,
and enforcing quality thresholds in automated tests. No paid APIs, LLM judges, GPU, or runtime
dependencies are required for the core library.

**Status:** working alpha (`0.1.0`). Install from this repository; no PyPI release is claimed.

## The problem

You change an embedding model, chunking strategy, index, or reranker. The overall score improves,
but an important query category gets worse. Or a run silently omits difficult queries and the
reported average becomes misleading.

Retrieval Regression evaluates both runs over the **same complete set of judged queries**,
shows what changed, and lets you fail a test when a configured quality budget is exceeded.
It evaluates retrieved document rankings, not generated-answer correctness.

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
- Fixed-query macro averaging, including missing and empty rankings as zero.
- Strict JSON loading: duplicate keys and non-finite numbers are rejected.
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
benchmarks/                 Reproducible synthetic evaluator benchmark
docs/                       Requirements, architecture, research protocol and roadmap
```

## Development and research

Start with [development setup](docs/DEVELOPMENT.md), [requirements](docs/REQUIREMENTS.md),
and the [roadmap](docs/ROADMAP.md). Run `python scripts/check.py` after changes.
The synthetic example demonstrates behavior; **it is not evidence of real-world retrieval gains**.
No benchmark speedups, model improvements, or external adoption are claimed.

A public SciFact experiment is included in `benchmarks/scifact.py`; install `.[benchmark]`
and run `python benchmarks/scifact.py --download` once, then rerun offline without that flag.
See [the validation report](docs/reports/2026-11-12-validation.md) for measured outcomes.
Ranking-file adapters and large-run scalability work remain on the roadmap. See [research protocol](docs/RESEARCH.md).

## License

[MIT](LICENSE).
