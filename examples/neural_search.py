"""Search a cached corpus with BM25 candidates and a locally trained reranker.

Run from the checkout: python -m examples.neural_search --query "your English question"
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from experiments.reranking.run import bm25_run
from experiments.reranking.training import ModelScorer, load_model
from retrieval_regression import rerank
from retrieval_regression.io import load_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--checkpoint", type=Path, help="Load a downloaded local model directory")
    parser.add_argument("--dataset", choices=["scifact", "nfcorpus"], default="scifact")
    parser.add_argument(
        "--model",
        choices=["selected", "pretrained", "random_negative", "hard_negative"],
        default="selected",
    )
    parser.add_argument("--experiment", type=Path, default=ROOT / "reports/neural-reranking")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.experiment / "config.json"
    config = load_json(
        config_path if config_path.exists() else ROOT / "experiments/reranking/config.json"
    )
    if args.checkpoint:
        variant, checkpoint = "local_checkpoint", args.checkpoint
    else:
        variant = (
            load_json(args.experiment / "selection.json")["selected"]
            if args.model == "selected"
            else args.model
        )
        checkpoint = (
            ROOT / "data/models" / config["revision"]
            if variant == "pretrained"
            else args.experiment / variant
        )
    with zipfile.ZipFile(ROOT / "data" / f"{args.dataset}.zip") as archive:
        rows = [
            json.loads(line)
            for line in archive.read(f"{args.dataset}/corpus.jsonl").decode().splitlines()
        ]
    corpus = {row["_id"]: row.get("title", "") + " " + row["text"] for row in rows}
    titles = {row["_id"]: row.get("title", "") for row in rows}
    queries = {"query": args.query}
    baseline = bm25_run(corpus, queries, queries, config["candidates"])
    model, tokenizer = load_model(checkpoint, args.device)
    scored = rerank(
        queries,
        corpus,
        baseline,
        ModelScorer(model, tokenizer, args.device, config["max_length"]),
        top_k=config["candidates"],
        batch_size=config["inference_batch_size"],
    )
    original_ranks = {doc: rank for rank, doc in enumerate(baseline["query"], 1)}
    ranked = sorted(scored["query"], key=lambda d: (-scored["query"][d], d))
    result = {
        "query": args.query,
        "dataset": args.dataset,
        "model_variant": variant,
        "model_revision": config["revision"],
        "results": [
            {
                "rank": rank,
                "docid": doc,
                "title": titles[doc],
                "bm25_rank": original_ranks[doc],
                "neural_score": scored["query"][doc],
                "snippet": corpus[doc][:350],
            }
            for rank, doc in enumerate(ranked[: args.top_k], 1)
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
