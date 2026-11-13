# Roadmap and execution order

## M0 — Working alpha

Implemented: fixed-universe metrics, strict JSON validation, paired comparisons, category
summaries, absolute quality gates, optional bootstrap intervals, HTML/JSON, CLI, pytest plugin,
offline examples, independent reference checks, and documentation.
Final verification evidence lives in `docs/reports/2026-11-12-validation.md`, not this checklist.

## M1 — Public-data validation and correctness audit

- [x] Complete a traceable SciFact/BEIR test-split experiment with fixed lexical baselines.
- [x] Verify all reported metrics against an independent implementation.
- [x] Inject clearly labeled missing-query failures and prove the gate detects them.
- [x] Publish a report with source/hash/configuration, all results, failures, and limitations.

Acceptance: another person can rerun one documented command, recover matching metric values,
and distinguish natural retriever differences from deliberately injected failures.

## M2 — Interoperable input adapters — delivered in 0.2.0

Implemented: strict BEIR TSV and TREC qrels/run loaders, public Python dispatch, per-input
CLI format selection, file/line errors, duplicate judgments/documents/ranks, tag and score/rank
validation. Optional manifests retain input/result fingerprints without machine-specific paths.

Evidence: original SciFact BEIR bytes and 60,000 TREC run rows produce exactly matching JSON,
API and CLI results. Independent reference checks and corruption/missing-query injection pass.
See docs/reports/2026-11-12-formats.md for conventions, commands and limitations.

## M3 — Usability and scalability

Completed: synthetic evaluation at 100/1,000/10,000 queries and 100 documents per present query.
Recorded time and traced allocations; no evaluator optimization is claimed.

Remaining: profile deeper runs and inspect result/report overhead before optimizing. Record time and peak
memory, including per-query result overhead. Investigate top-K selection, result/report size,
report pagination, and eventual streaming only when measurements justify them.
Delivered in 0.4.0: category intervals and bounded CLI summaries.
Remaining: multi-label categories and paginated/streamed reporting.

Acceptance: report exact tested scales and tradeoffs; no universal performance claim from one laptop.

## M4 — Broader evidence and release readiness

Delivered in 0.3.0: pinned neural reranking, two controlled one-epoch adaptation conditions,
SciFact and external NFCorpus tests, validation-only selection, and a local search command.
The experiment removed two exact normalized train/test query overlaps before training.

Next: repeat across seeds, expand query/language coverage, and compare training losses and
candidate depths using held-out validation. Pin model
revision and configuration, keep test labels out of selection, and report negative outcomes.
Add API compatibility fixtures, user feedback, and documented release notes.

Acceptance: repeatable external examples, clean wheel installation, passing platform CI,
reviewed limitations and packaging metadata. PyPI publication is not automatic.

## Task selection

Prioritize reusable retrieval/model functions and measured experiments. Keep research/model
training optional and separate from the runtime core. Expand candidate generation, training
objectives and multilingual evaluation through evidence-backed iterations; avoid unrelated
infrastructure that does not improve these workflows.

## M5 - Statistical regression decisions - delivered in 0.4.0 source

Three-state percentile-bootstrap gates, per-category evidence, nominal Bonferroni adjustment,
explicit inconclusive CI policy and seeded small-sample A/A/regression calibration.
Next: benchmark power and coverage across query dependence/sparsity and investigate alternative
interval methods before making broad statistical guarantees.
