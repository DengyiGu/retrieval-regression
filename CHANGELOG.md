# Changelog

## 0.3.0 — Unreleased alpha

- Model-independent batched reranking with candidate-score separation and returned Run mappings.
- Optional local MiniLM neural reranking, controlled random/hard-negative fine-tuning and
  validation-only model selection, evaluated on SciFact and NFCorpus held-out queries.
- Normalized query-group splitting and removal of two exact train/test text overlaps.
- Local neural search example using pretrained or validation-selected adapted checkpoints.
- Measured model results, actual GPU training/inference and independent metric verification.

## 0.2.0 — Unreleased alpha

- Explicit BEIR TSV and TREC qrels/run loaders with physical-line diagnostics.
- Duplicate-row/rank, consistent-tag and score/rank validation with documented strict profiles.
- Independent input format selection for CLI comparisons and backward-compatible JSON defaults.
- Optional portable manifests fingerprint inputs, configuration and results; changing input
  content aborts before outputs are written.
- Real SciFact format-conversion audit, independent file-reader parity tests and synthetic
  measurements through 10,000 queries. No neural model improvement or throughput speedup is claimed.

## 0.1.0 — Unreleased alpha

- Local, dependency-free retrieval metrics with an explicit query universe.
- Strict data validation, paired comparisons, quality gates and category summaries.
- Optional seeded bootstrap intervals, JSON/HTML, CLI and pytest integration.
- Offline examples, independent metric checks, developer documentation.

Audit fixes: reject overflowing JSON floats and hard-linked output collisions, label ungated
HTML reports explicitly, and preserve Unicode query IDs on legacy-encoded CLI stdout. CI now
also verifies installation and plugin discovery from the built wheel in a fresh environment.

This version is available from source; no PyPI release is claimed.
