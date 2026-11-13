---
language: en
license: apache-2.0
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
tags:
- reranking
- minilm
- scifact
---

# SciFact MiniLM Reranker

A 22.7M-parameter English Cross Encoder adapted to scientific query/document relevance.
Adaptation by [Dengyi Gu](https://github.com/DengyiGu). Base model:
[cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.

## Download and load

The complete model is attached to the [v0.3.0 release](https://github.com/DengyiGu/retrieval-regression/releases/tag/v0.3.0).
Download `scifact-minilm-reranker-v0.3.0.zip`, optionally compare it with `SHA256SUMS`, and extract it.
The extracted `scifact-minilm-reranker` directory contains weights and the tokenizer; training
is not required to use the released checkpoint.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = "scifact-minilm-reranker"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    path, local_files_only=True, use_safetensors=True, attn_implementation="eager"
).eval()
inputs = tokenizer(
    ["Where is Berlin located?", "Where is Berlin located?"],
    ["Berlin is a city in Germany.", "Tokyo is a city in Japan."],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)
with torch.inference_mode():
    scores = model(**inputs).logits[:, 0].tolist()
print(scores)
```

Higher logits rank documents as more relevant within a query; scores are not calibrated
probabilities. For corpus search in the source checkout:

```console
python -m examples.neural_search --checkpoint scifact-minilm-reranker --query "your English query"
```

That command also needs the cached SciFact corpus; `python benchmarks/scifact.py --download`
prepares it. Install the checkout's `benchmark,neural` extras and an appropriate PyTorch build.

## Training and selection

Fine-tuning used SciFact's official training split. Two normalized query-text matches with the
original test queries were excluded. The fixed remaining split has 727 training and 80 validation
queries; duplicate normalized text stays in one partition. One epoch used 3,336 pairs (834
positive, 2,502 random weak negatives), AdamW at 2e-5, batch 16, max length 256 and float32.
Known positive documents were excluded from the negative pool.

Two runs used seeds 42 and 13 on the same split. Seed 13 was selected strictly by validation
NDCG@10 (0.683297 versus 0.677952). Test scores did not determine the selected seed.

| Run | SciFact test NDCG@10 | NFCorpus test NDCG@10 |
| --- | ---: | ---: |
| Original pretrained model | 0.667796 | 0.332514 |
| Adaptation, seed 42 | 0.698973 | 0.334350 |
| Released adaptation, seed 13 | 0.699263 | 0.333319 |

All systems rerank the same top-50 BM25 candidates. SciFact test contains 300 queries; NFCorpus
test contains 323. Scores are conditional on those candidate pools and 256-token truncation.
The two-seed check reuses previously observed test sets; it is not a fresh unseen evaluation.

## Intended use and limits

This is an exploratory scientific-retrieval checkpoint, not a general-purpose improvement
claim. Two seeds and one epoch are a small check. NFCorpus NDCG@10 changes are small; top-rank
metrics can worsen, including NDCG@1 dropping from 0.486068 to 0.469040 for the released model.
Most negative labels are unjudged, and semantic or pretraining overlap is not ruled out.
English retrieval was evaluated; other languages were not. Retain task-specific evaluation.

Training/evaluation code, fixed configuration, detailed results and model hashes are in
[retrieval-regression](https://github.com/DengyiGu/retrieval-regression). The archive includes
`release-check.json` and `model_info.json` for the exact artifact and the two-seed results.

## Sources and license

Weights/tokenizer retain Apache-2.0 terms and upstream attribution; see `LICENSE` and `NOTICE.txt`.
The repository's MIT code license does not relicense pretrained weights or raw data.
SciFact source: https://github.com/allenai/scifact . NFCorpus evaluation source:
https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ . No corpus text or training pairs are
redistributed in this artifact.
