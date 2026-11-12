# Validation and public-data audit — 2026-11-12

## Outcome

The working alpha passed its local correctness/integration checks and independently matched
retrieval metrics on the complete BEIR SciFact test-query set. A real lexical retriever change
failed its preconfigured quality gate, and a deliberately injected missing-query export fault
was detected. These outcomes validate the evaluation workflow; they do not establish a new
retrieval algorithm or universal model improvement.

## Dataset and provenance

- Dataset: BEIR-formatted SciFact, scientific claim-to-abstract retrieval, **test** split.
- Measured counts: **5,183 documents, 300 test queries, 339 relevance judgments**.
- Original project: https://github.com/allenai/scifact
- Original paper: https://aclanthology.org/2020.emnlp-main.609/
- BEIR release: https://github.com/beir-cellar/beir
- Archive: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
- Archive SHA-256: `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
- Archive MD5: `5f7d1de60b170fc8027bb7898e2efca1`, checked against BEIR's published value.
- Data and results were downloaded/generated locally on November 12, 2026.

Licensing: the original project's license distinguishes claims/evidence annotations (CC BY 4.0)
and abstracts (ODC-By 1.0): https://github.com/allenai/scifact/blob/master/LICENSE.md.
The BEIR Hugging Face mirror lists CC BY-SA 4.0 and another mirror has differing metadata.
Those labels are not treated as a blanket relicensing of the original components. Raw corpus
text and downloaded archives are not committed or redistributed with this MIT-licensed code.

## Fixed experimental setup

No test-label tuning or model training was performed. Both systems use the full same corpus,
complete same query universe, and the same lowercase English word analyzer with stopword
removal and tokens of at least two characters. Retrieval depth is 100; K is 1, 5, and 10.

- Baseline: `rank_bm25.BM25Okapi`, k1=1.5, b=0.75, epsilon=0.25.
- Candidate: scikit-learn TF-IDF, sublinear TF and L2 normalization; cosine-style dot-product ranking.
- Categories: <=12 analyzed query tokens is short (256 queries), otherwise long (44 queries).
- Gate: candidate NDCG@10 must not fall more than **0.02 absolute** below baseline.
- Uncertainty: 1,000 paired bootstrap samples, seed 42, 95% descriptive intervals.
- Ties: ascending document IDs; unjudged documents receive zero gain.

These are fixed lexical baselines, not the official BEIR leaderboard implementations/configurations.
The candidate is not assumed to be better merely because it is named candidate.

## Measured retrieval results

| Metric | BM25 baseline | TF-IDF candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.531944 | 0.461111 | -0.070833 |
| Recall@5 | 0.719278 | 0.727556 | +0.008278 |
| Recall@10 | 0.791444 | 0.774333 | -0.017111 |
| Precision@10 | 0.088333 | 0.086000 | -0.002333 |
| MRR@10 | 0.636794 | 0.588540 | -0.048254 |
| NDCG@5 | 0.643795 | 0.611213 | -0.032582 |
| NDCG@10 | 0.669693 | 0.627965 | -0.041727 |

The NDCG@10 gate **failed as intended**: a 0.041727 drop exceeds the 0.02 budget.
Recall@5 improves slightly while ranking-sensitive metrics decline. A single improved metric
would therefore be insufficient evidence to accept this change under the configured policy.

NDCG@10 deltas by predefined category: short queries -0.038833; long queries -0.058565.
The overall paired-bootstrap interval is [-0.060093, -0.022954]. This describes uncertainty
under the stated query-sampling assumptions, not causality or general model superiority.
There is no multiple-comparison correction, and category intervals are not implemented.

## Independent metric agreement

All 12 aggregate metrics for each natural run were checked against `ir-measures`/
`pytrec-eval-terrier`. Maximum absolute error for both runs was **6.661338147750939e-16**.
Before reference evaluation, retrieval scores were replaced with unique rank scores to
preserve the library's documented tie ordering. No ranking was otherwise changed.
Missing-query handling was checked separately because external tools differ in denominator policy.

## Deliberately injected failure

Remove every tenth query in sorted ID order from the BM25 run: **30 of 300 queries omitted**.
This is a labeled fault-injection experiment, not a naturally observed model defect.

| Calculation | NDCG@10 |
| --- | ---: |
| Original complete BM25 run | 0.669693 |
| Faulty run, correct full-query denominator | 0.606759 |
| Faulty run, incorrect survivor-only denominator | 0.674177 |

The survivor-only average misleadingly appears higher than the original complete run.
The library retains all 300 queries, records the 30 omissions, and fails the zero-drop gate.
This demonstrates why query-universe integrity is part of evaluation correctness.

## Code audit findings and fixes

1. **Output-file collision:** the CLI could replace qrels/run/config inputs if the report path
   matched an input. It now resolves paths first and rejects collisions, including identical
   JSON/HTML outputs. Integration tests confirm original input content is preserved.
2. **Coverage initialization:** pytest plugin discovery imports modules before pytest-cov starts,
   and subprocesses were initially unmeasured. Coverage now starts before pytest and instruments
   Python subprocesses. This fixes the measurement setup, not an underlying metric defect.
3. **Threshold validation:** invalid metric/group types are validated explicitly rather than
   leaking incidental container exceptions. Unknown bounds/metrics/categories fail closed.
4. **Dataset HTTPS trust:** the initial Python download failed certificate verification on this
   machine. The optional benchmark now uses the system certificate store through `truststore`;
   certificate verification remains enabled and the archive checksum is checked.

No mismatch was found in the independently checked metric cases. This is a scoped audit,
not proof that every future input or environment is correct.

## Local verification

The local checks use Windows and Python 3.11.3. Current test results and final packaging smoke
checks are recorded in [the verification summary](validation-summary.json). The check suite covers hand calculations,
missing/empty queries, score ties, invalid values, strict JSON, gates, bootstrap reproducibility,
HTML escaping, CLI exit paths, plugin discovery, and independent metric agreement.

A synthetic evaluator-only run with 1,000 queries, 100 retrieved documents per present query,
10% omitted queries, K=[5,10], seed 42 and 3 repeats measured a median **0.073733 seconds**.
A separate tracemalloc pass recorded **4,411,204 bytes** of peak traced allocation. The timed
runs exclude input construction and do not run tracemalloc. This measures one local scale,
not overall process memory, a speedup, or a cross-hardware performance guarantee.

## Reproduce

From the repository root, using the project environment:

```console
python -m pip install -e '.[dev,reference,benchmark]'
python scripts/check.py
python benchmarks/scifact.py --download
python benchmarks/scifact.py
python benchmarks/benchmark_evaluator.py --queries 1000 --documents 100 --repeats 3
```

On Windows substitute `.\.venv\Scripts\python.exe` for `python` when the environment is not activated.
Only the first SciFact invocation requires network access. The second verifies cached-data
reproducibility and regenerates local results without downloading.

A readable local index of all 300 questions and relevant-document IDs was additionally saved
as `data/scifact-test-questions.csv`; the original questions remain in the cached public archive.

Local full artifacts: `reports/scifact/{comparison.json,comparison.html,bm25.json,tfidf.json,
qrels.json,groups.json,injected-missing-query.json,summary.json}`.
Committed compact evidence:

- [Dataset, source-file fingerprints, software versions, configuration and aggregate results](scifact-summary.json)
- [All 300 query-level deltas](scifact-query-deltas.json)
- [Synthetic evaluator measurement](synthetic-evaluator-summary.json)
- [Observed local and remote verification summary](validation-summary.json)
- [Clean-wheel smoke verification](wheel-smoke.json)

The experiment was executed on uncommitted implementation work. Its recorded base Git commit
and dirty flag are retained honestly; per-source-file SHA-256 fingerprints identify the tested
implementation independently of the eventual bootstrap commit.

## Limitations and next work

One English scientific dataset and two lexical baselines are insufficient to establish broad
retrieval quality. Relevance judgments are incomplete. No embedding/reranker inference, training,
answer evaluation, streaming million-query run, or cross-platform execution was tested locally.
The [actual GitHub CI run](https://github.com/DengyiGu/retrieval-regression/actions/runs/43195875088) for implementation commit
`5518e88` passed all nine jobs: Windows/Linux on Python 3.10-3.13, plus independent-reference
and package validation. These are observed results, not merely a workflow configuration.
PyPI publication has not occurred.

Next: add strict BEIR/TREC adapters, validate another domain and a pinned neural retriever where
useful, profile larger scales, and collect feedback from actual users before expanding scope.
