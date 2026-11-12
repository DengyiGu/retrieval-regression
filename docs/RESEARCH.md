# Research and benchmark protocol

## Questions this project can answer

- Does a pipeline change reduce judged retrieval quality overall or in a chosen category?
- Does a missing-query/export defect inflate a conventional reported average?
- Are metric outputs consistent with an independent reference under matching conventions?
- What time/memory does this evaluator require at a measured scale?

The library does not prove a new retrieval algorithm, generated-answer correctness, or general
model superiority. Avoid broad research claims from a single public test split.

## Data and sources

Start with public, labeled retrieval data such as BEIR SciFact. Use the complete test-query
set and fixed corpus. Prefer primary sources:

- https://github.com/beir-cellar/beir
- https://github.com/allenai/scifact
- https://github.com/allenai/scifact/blob/master/LICENSE.md
- https://aclanthology.org/2020.emnlp-main.609/

Original SciFact licensing distinguishes claims/annotations and abstracts. Mirror metadata
may differ. Keep downloaded data local; cite the original terms and mirror discrepancy in the
report. The MIT code license does not relicense external data.

## Experiment design

Write the comparison configuration before computing the reported test result. Use the same
corpus, query set, preprocessing, ranking depth, metric cutoffs, and tie policy. Record all
hyperparameters. A comparison of TF-IDF and BM25 is a diagnostic example, not a claim that
this library improves either retriever. Do not tune a model on reported test labels.

Use independently established relevance judgments with documented provenance. For optional model experiments, pin a model revision and document possible
benchmark contamination and inference settings. Small lexical baselines are sufficient for
initial library correctness validation and require no GPU.

## Required experiment record

Store commands and raw machine-readable results. Include:

- Experiment identifier, purpose, execution date/time, Git state and random seeds.
- Source URLs, download hashes, license notes, split, corpus/query/judgment counts.
- Dataset validation, missing queries, unjudged documents, duplicates and rejection behavior.
- Retrieval implementation/version, preprocessing, parameters, depth and metric conventions.
- Baseline/candidate metrics, query-level deltas, gates and optional uncertainty.
- Independent reference discrepancies and how tie conventions were aligned.
- Exact software versions, measured wall time, hardware if performance is discussed.
- Negative outcomes, known limits, and a reproducible next step.

Raw datasets and detailed local artifacts stay under ignored `data/`, `reports/`, or `work/`.
Compact derived evidence and reports can be committed under `docs/reports/` without corpus text.
Synthetic datasets and injected missing-query faults must be labeled separately from natural
retrieval outcomes. Never remove difficult queries to improve a reported average.
