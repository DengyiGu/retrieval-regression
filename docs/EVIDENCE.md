# Evidence and limits

This guide separates useful findings from routine development checks. Source tests and the
CI badge cover implementation checks; temporary task logs remain local.

## Noisy quality gates

`python benchmarks/gate_calibration.py` repeats 300 trials per scenario on 30 independent
queries. Each system's NDCG@1 is a Bernoulli hit: baseline probability .70; candidate .70
for equal quality or .45 for regression. The absolute drop budget is .02, confidence .95,
bootstrap draws 2,000, fixed seed 20261113. These are synthetic systems, not model benchmarks.

| Scenario | Point pass / fail | Bootstrap pass / fail / inconclusive |
| --- | --- | --- |
| Equal quality (A/A distribution) | 163 / 137 | 15 / 7 / 278 |
| True .25 expected decline | 7 / 293 | 0 / 142 / 158 |

[Machine-readable results](reports/gate-calibration.json). This independently constructed A/A
simulation illustrates the review's concern; it does not reproduce its unspecified noise.
Interval decisions reduced definite false failures to 7/300 (2.3%) in this particular setup,
but most A/A trials remained inconclusive. The regression case shows the power cost: only
142/300 definite detections at this sample size, with 158 inconclusive. Blocking inconclusive
results still blocks 285/300 A/A trials; warning mode is an explicit operational tradeoff,
not evidence of safety. Choose larger representative query sets for tight budgets.

Percentile bootstrap coverage and the Bonferroni adjustment are approximate here. Sparse or
degenerate samples, correlated queries and repeated test-set tuning require more care; these
numbers are not a general false-alarm bound. See [decision rules](EVALUATION.md).

## Real qrels compatibility

Only judgment files and the Robust topic list were downloaded, not document corpora.
Inspection date: 2026-11-13. Negative grades were preserved during parsing and explicitly
clipped or dropped for scoring; zero-positive behavior was separately tested with small fixtures.

| Upstream file | Observed contents | Relevant behavior |
| --- | --- | --- |
| [BEIR TREC-COVID test.tsv](https://huggingface.co/datasets/BeIR/trec-covid-qrels/resolve/main/test.tsv) | 50 queries, 66,336 judgments; grades -1: 2, 0: 41,661, 1: 10,456, 2: 14,217 | Strict rejects; explicit negative policies load/evaluate successfully |
| [NIST Robust04 qrels](https://trec.nist.gov/data/robust/qrels.robust2004.txt) | 249 queries, 311,410 judgments; all represented queries have positives | No all-zero topic was found in this actual file |
| [NIST Robust04 topics](https://trec.nist.gov/data/robust/04.testset.gz) | 250 topics; topic 672 is absent from qrels | Absent topics are different from represented all-zero topics |

SHA-256 of exact downloaded qrels bytes:
- TREC-COVID: `10669ab7d526cb04f52079139fd88c3d467a0776441b046567f540582798982b`
- Robust04: `f8f2c972d3c710d85daa7ead02daf4ffe2bbe3647c9f3904500182f43ddbf4c3`

The evaluator cannot infer topics missing from qrels. If all 250 topics belong to a chosen
universe, explicitly add the missing topic with an empty mapping and choose `no_positive`.
Clipping negative markers to zero differs semantically from preserving them as unjudged;
choose a policy appropriate to the collection. This library does not implement infAP.

## Score ties and external agreement

[trec_eval's comparator](https://github.com/usnistgov/trec_eval/blob/main/form_res_rels.c)
breaks score ties by descending document ID. The legacy default here is ascending, so tied
results can differ. Optional `doc_id_desc` is tested against the independent pytrec_eval
provider. This does not claim parity for every external tool, metric or denominator policy.

## Neural reranking: local gains, limited transfer

The [pinned MiniLM experiment](reports/2026-11-12-neural.md) uses 22,713,601 parameters,
fixed top-50 BM25 candidates, one adaptation epoch and validation-only selection. Two
random-negative seeds were checked; this is a small demonstration, not broad model research.

- SciFact NDCG@10: pretrained .667796; random-negative seed 42 .698973; selected seed 13 .699263.
- NFCorpus NDCG@10: pretrained .332514; seed 42 .334350; selected seed 13 .333319.
- NFCorpus paired intervals span zero; selected seed 13 also lowers NDCG@1 (.469040 vs .486068).
- Hard-negative sampling did not beat random negatives in this fixed experiment.

The defensible conclusion is a local SciFact gain plus limited transfer evidence. The project
is useful as an evaluation/regression workflow; neither the small model nor a single short
training run establishes a novel reranking method. [Model card](../experiments/reranking/MODEL_CARD.md)
and [seed comparison](reports/model-release.md) retain model/data attribution and limitations.

## Further reproducible evidence

- [SciFact lexical experiment](reports/2026-11-12-validation.md): fixed complete query universe
  and a missing-query fault expose misleading averages.
- [Standard-format experiment](reports/2026-11-12-formats.md): original BEIR and exported TREC
  representations agree under matching conventions; recorded scale reaches 10,000 queries.
- [Neural machine-readable record](reports/neural-summary.json) and
  [seed record](reports/model-release-check.json): fixed configuration and negative outcomes.

Historical experiment artifacts retain their original evaluator version and schema; current
features do not retroactively change those measurements. Large-scale streaming, clustered
query bootstrap and broader neural generalization remain unimplemented.
