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

## M2 — Interoperable input adapters

Implement BEIR qrels TSV and TREC run parsing as separate adapters. Validate duplicate
query/document pairs, malformed headers/rows, ranks and finite scores; retain string IDs.
Add format-specific tests and CLI selection. Do not make silent format guesses.

Acceptance: public run files can be compared without custom conversion code, preserving the
same query-universe and metric policies. Keep raw-data licenses separate from project code.

## M3 — Usability and scalability

Profile 100/1,000/10,000-query synthetic evaluations before optimizing. Record time and peak
memory, including per-query result overhead. Investigate top-K selection, result/report size,
report pagination, and eventual streaming only when measurements justify them.
Add multi-label categories or uncertainty within categories only with explicit semantics.

Acceptance: report exact tested scales and tradeoffs; no universal performance claim from one laptop.

## M4 — Broader evidence and release readiness

Use another public domain and a fixed neural retriever/reranker when justified. Pin model
revision and configuration, keep test labels out of selection, and report negative outcomes.
Add API compatibility fixtures, user feedback, and documented release notes.

Acceptance: repeatable external examples, clean wheel installation, passing platform CI,
reviewed limitations and packaging metadata. PyPI publication is not automatic.

## Task selection

Finish the earliest incomplete milestone before adding a dashboard, LLM judge, vector database, or training framework. Those are not requirements for this library.
