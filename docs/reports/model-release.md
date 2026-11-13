# Model release check

A short follow-up retains the original seed-42 run and adds one seed-13 run of the previously
validation-selected random-negative condition. The split is unchanged: 727 train, 80 validation,
two exact train/test text matches excluded. Sampling/shuffle/dropout seed changes; model,
one-epoch schedule, candidate pool and sequence length remain fixed.

| Seed | Validation NDCG@10 | SciFact test NDCG@10 | NFCorpus test NDCG@10 |
| --- | ---: | ---: | ---: |
| 42 | 0.677952 | 0.698973 | 0.334350 |
| 13 | 0.683297 | 0.699263 | 0.333319 |

The pretrained references are 0.667796 (SciFact) and 0.332514 (NFCorpus). Both adapted runs
improve the primary SciFact metric. Their external NDCG@10 values are close to the pretrained
reference and do not establish a reliable general improvement. The previously documented
external top-rank tradeoff remains: the released model's NFCorpus NDCG@1 is 0.469040 versus
0.486068 pretrained. No fresh unseen test set or extensive stability study is claimed.

Seed 13 was selected on validation NDCG@10 before reporting its test scores. The checkpoint's
safetensors SHA-256 and exact follow-up results are in [the compact record](model-release-check.json).
It was reloaded successfully for corpus search before packaging.

The [v0.3.0 release](https://github.com/DengyiGu/retrieval-regression/releases/tag/v0.3.0)
contains weights, tokenizer, model card, Apache-2.0 license, attribution notice and provenance
inside one zip; `SHA256SUMS` is a separate asset. The [model card](../../experiments/reranking/MODEL_CARD.md)
shows loading without training. Code remains MIT; upstream model/tokenizer terms are retained.
Raw datasets, training pairs, local instructions and credentials are not distributed.


## Publication verification

The release is public and contains the 83,464,572-byte model archive and a separate checksum
file. The public archive was downloaded without authentication, its SHA-256 verified, and the
extracted model/tokenizer successfully loaded for two short inference examples. Exact artifact
hashes are published with the Release; current source checks are linked from the README.
The model is labeled a research preview; no separate repository or Git-history weight blobs
were introduced.
