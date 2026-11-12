# Architecture

## Data flow

```text
Python mappings / strict JSON
        -> validation -> evaluate every judged query -> aggregate metrics
        -> paired comparison -> category summaries -> explicit quality gates
        -> versioned JSON / standalone HTML / CLI exit code / pytest failure
```

## Modules

- `validation.py`: identifiers, finite real numbers, qrels/run contracts and diagnostics.
- `evaluation.py`: deterministic ranking and binary/linear-gain metrics; no I/O.
- `comparison.py`: paired deltas, category summaries, thresholds, optional bootstrap intervals.
- `models.py`: typed public dataclasses and JSON serialization boundary.
- `io.py`: strict JSON parser and narrowly validated threshold configuration.
- `reporting.py`: local reports with escaped user text; no external resources.
- `cli.py`: argument parsing, file orchestration, exit-code contract.
- `pytest_plugin.py`: optional adapter; the core never imports pytest.

The computational API is independent of CLI, pytest, model frameworks, and report rendering.
Do not add torch, a vector database, a web server, or an LLM SDK to core dependencies.
Optional adapters should translate inputs into the same internal contracts.

## Design decisions

1. **qrels determines queries.** Never use run keys or the intersection of two runs as the denominator.
2. **Unknown query IDs are errors.** Accidental dataset mixing should not silently produce a score.
3. **Positive judgments are required.** Queries with no positive judgments are rejected explicitly.
4. **Linear NDCG gains.** Independent parity with `trec_eval` conventions is easier to audit.
5. **Document-ID tie break.** Ordering is deterministic, though tied-run results may differ from
   tools with other tie conventions. Reference parity tests therefore avoid score ties.
6. **No runtime dependencies.** Transparent correctness and installation are prioritized over
   large-run throughput in the alpha. Validate scale before adding a dependency.
7. **Explicit enforcement.** A plain comparison may have no gates; the pytest fixture must have one.

## Complexity and current limits

For Q queries with R retrieved documents and J judgments each, sorting is roughly
O(Q * (R log R + J log J)). Comparisons retain per-query data for both runs. Bootstrap adds
O(B * Q * M) work for B samples and M metrics. JSON, reports, and results are materialized in
memory; million-query runs are outside the current scalability claim. HTML includes all query
rows. Confidence intervals are currently overall only, not category-specific.

## Extensibility

Add BEIR/TREC adapters as isolated parsers; preserve duplicate detection and identifier types.
A future streaming API needs a separate result/report contract rather than silently dropping
per-query evidence. Schema changes need migration notes and backwards-compatibility tests.
