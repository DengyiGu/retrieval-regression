# Product requirements

## Purpose and target users

Help RAG/search engineers verify that changes to a retriever, index, or reranker do not
silently reduce judged retrieval quality. The core runs offline on a CPU with no cloud
resources, API keys, model calls, or runtime dependencies.

## Functional requirements and acceptance

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| F1 | Fixed query universe | Missing/empty runs score zero over the full qrels denominator |
| F2 | Correct ranking metrics | Hand-worked examples plus independent `ir-measures` agreement |
| F3 | Strict inputs | Reject invalid IDs, duplicate JSON keys, NaN/Infinity, unknown queries |
| F4 | Actionable comparisons | Overall and per-query deltas use candidate minus baseline |
| F5 | Categories | User-supplied query-to-category mapping; unknown queries rejected |
| F6 | Quality gates | Absolute minimum and/or maximum drop, optionally category-scoped |
| F7 | CLI integration | Exit 0/1/2 for success/regression/invalid inputs |
| F8 | Reports | JSON schema version and standalone escaped HTML |
| F9 | pytest plugin | Fixture requires thresholds and exposes understandable failures |
| F10 | Optional uncertainty | Seeded paired bootstrap on common query set; assumptions documented |

## Non-functional requirements

- Python 3.10+; stdlib-only runtime core; optional test/reference dependencies.
- Windows is a first-class local development target. CI also exercises Linux.
- No network calls or telemetry in the core library, tests, toy examples, or reports.
- Public docs, APIs, errors, and code comments use English.
- Stable ordering, explicit metric semantics, meaningful type hints, and actionable exceptions.
- No silent filtering, query sampling, grade conversion, or threshold relaxation.
- Preserve source/run files; generated artifacts go under ignored `reports/` or `work/`.
- Large datasets/models are optional, downloaded only for an explicitly selected experiment.

## Initial boundaries

Input is nested JSON/Python mappings with document IDs and scores, not raw documents.
The library does not run a vector database, train models, judge generated answers, infer
root causes, or detect every kind of dataset leakage. It reports measured differences;
users determine why a pipeline changed.

## Completion criteria for the alpha

Runnable API/CLI/plugin, documented conventions, meaningful tests, independent metric parity,
source and wheel builds, example pass/fail paths, and honest limitations. The next research
milestone requires public-data experiments and traceable artifacts, not a README claim.
