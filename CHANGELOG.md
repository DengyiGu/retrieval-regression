# Changelog

## 0.1.0 — Unreleased alpha

- Local, dependency-free retrieval metrics with an explicit query universe.
- Strict data validation, paired comparisons, quality gates and category summaries.
- Optional seeded bootstrap intervals, JSON/HTML, CLI and pytest integration.
- Offline examples, independent metric checks, developer documentation.

Audit fixes: reject overflowing JSON floats and hard-linked output collisions, label ungated
HTML reports explicitly, and preserve Unicode query IDs on legacy-encoded CLI stdout. CI now
also verifies installation and plugin discovery from the built wheel in a fresh environment.

This version is available from source; no PyPI release is claimed.
