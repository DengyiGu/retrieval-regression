# Local neural reranking experiment

Compare a pinned pretrained Cross Encoder with one-epoch adaptation using random or BM25
hard negatives. This optional experiment uses the library's batched `rerank` interface;
the library runtime itself has no torch/model dependency.

## Run

Create a separate Python 3.11 environment, install a suitable CUDA-enabled PyTorch build using
[official instructions](https://pytorch.org/get-started/locally/), then from the checkout run:

```console
python -m pip install -e '.[benchmark,neural]'
python -m experiments.reranking.run --download
```

`--download` permits the initial public dataset/model downloads. Subsequent runs need no network.
Default device is CUDA; use `--device cpu` explicitly for CPU execution. Use a different
`--output` for a new experiment; `--resume` continues the same configuration/source snapshot.
A normal fresh virtual environment is recommended. A local run may also reuse installed GPU
libraries read-only through a project-local environment with `--system-site-packages`.

## Fixed design

`config.json` records the exact model revision and file hashes, seed, candidate depth, sequence
length, optimizer settings and sampling conditions. The model is
[cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
a pretrained English reranker released under Apache-2.0. Load only pinned local safetensors
with built-in Transformers code; training checkpoints stay under ignored experiment outputs.

SciFact official training queries are grouped by normalized text. Exact normalized overlap
with the official test queries is removed before the seeded training/validation split.
All positive judgments for a training query are excluded from negative sampling. Each condition
uses the same positive pairs and negative count, seed and hyperparameters. Most negatives are
unjudged rather than confirmed irrelevant; the saved pair records label this distinction.

Each variant is evaluated on the training-derived validation split. Selection uses NDCG@10
there only, with ties favoring earlier variants in the fixed configuration. All variants are
then reported on SciFact test and NFCorpus test, including regressions. Test relevance labels
are not used to construct training pairs, tune parameters or select a variant. BM25 uses the
full corpus without relevance labels and supplies the same top-50 candidates to all models.

The experiment saves split IDs, sampled pairs, checkpoints, validation scores, selected variant,
full test runs, comparisons, bootstrap intervals, timing and allocated GPU-memory measurements.
Archive, configuration, checkpoint and source hashes identify the inputs and implementation.
One seed and one epoch are an exploratory ablation, not a universal ranking of sampling methods.

## Use the model for local search

After the experiment, query either cached corpus using the validation-selected model:

```console
python -m examples.neural_search --query "Does smoking increase cancer risk?" --dataset scifact
python -m examples.neural_search --query "What are the benefits of exercise?" --dataset nfcorpus --model pretrained
```

Results show document IDs/titles, snippets, neural scores and previous BM25 ranks. These are
retrieval demonstrations, not validated factual or medical answers. The models and corpora
are English; multilingual behavior is not established.

## Data provenance and terms

- BEIR archive URLs and MD5 values: https://github.com/beir-cellar/beir/wiki/Datasets-available
- SciFact: https://github.com/allenai/scifact and its original component-specific license terms.
- NFCorpus: https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ . Its source page permits
  academic use and refers other uses of NutritionFacts data to the original terms. This is a
  local research experiment; no raw texts, archives or model weights are redistributed.

Review original component terms for your intended use; the repository's MIT code license does
not relicense external data or pretrained weights.
