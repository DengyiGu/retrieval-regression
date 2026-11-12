Retrieval Regression

Regression testing for search and RAG retrieval pipelines.

Retrieval Regression is a Python library designed to help developers answer a practical question:

Did my retrieval changes improve quality—or silently break existing queries?

It compares baseline and candidate retrieval results against relevance judgments, identifies query-level regressions, and produces actionable reports.

Planned Features

Standard retrieval metrics, including Recall@K, MRR@K, and NDCG@K.

Baseline-versus-candidate comparisons with query-level breakdowns.

Validation for missing queries, duplicate document IDs, and invalid scores.

Evaluation by user-defined query categories.

Configurable quality thresholds and pytest integration.

Local HTML and JSON reports.

Examples using public retrieval benchmarks.

Design Goals

Run locally without paid APIs or an LLM judge.

Work with retrieval results from different frameworks.

Make evaluation assumptions and edge cases explicit.

Provide reproducible examples and independently verified metrics.

Project Status

Early development. The features above describe the planned initial release.
