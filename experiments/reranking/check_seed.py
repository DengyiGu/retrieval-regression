"""One additional training seed on the original fixed split; select by validation only."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from experiments.reranking.data import read_dataset, sample_pairs, sha256
from experiments.reranking.run import dump, score_model
from experiments.reranking.training import load_model, seed_everything, train_model
from retrieval_regression import evaluate
from retrieval_regression.io import load_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/neural-reranking")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/model-release-check")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    import torch

    original = load_json(args.source / "summary.json")
    config = dict(original["config"])
    if args.seed == config["seed"]:
        raise ValueError("choose a seed different from the original experiment")
    split = load_json(args.source / "split.json")
    candidates = load_json(args.source / "scifact-candidates.json")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config["seed"] = args.seed
    dump(output / "config.json", {**config, "split_seed": original["config"]["seed"]})
    corpus, queries, judgments = read_dataset(ROOT / "data/scifact.zip", "scifact", "train", output)
    train_qrels = {qid: judgments[qid] for qid in split["train"]}
    val_qrels = {qid: judgments[qid] for qid in split["validation"]}
    val_candidates = {qid: candidates[qid] for qid in val_qrels}
    checkpoint = output / f"random-negative-seed-{args.seed}"
    seed_everything(args.seed)
    if (output / "training.json").exists():
        training = load_json(output / "training.json")
    else:
        pairs = sample_pairs(
            train_qrels,
            candidates,
            corpus,
            mode="random_negative",
            negatives_per_positive=config["negatives_per_positive"],
            seed=args.seed,
        )
        dump(output / "training-pairs.json", pairs)
        model, tokenizer = load_model(ROOT / "data/models" / config["revision"], "cuda")
        training = train_model(model, tokenizer, pairs, queries, corpus, config, "cuda")
        checkpoint.mkdir(exist_ok=True)
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        dump(output / "training.json", training)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    val_run, _ = score_model(checkpoint, corpus, queries, val_candidates, config, "cuda")
    validation = evaluate(val_qrels, val_run, k=[1, 5, 10]).summary
    old = original["validation_and_training"]["random_negative"]["validation"]
    selected_seed = (
        args.seed if validation["ndcg@10"] > old["ndcg@10"] else original["config"]["seed"]
    )
    selected_path = checkpoint if selected_seed == args.seed else args.source / "random_negative"
    dump(
        output / "selection.json",
        {
            "seed": selected_seed,
            "criterion": "SciFact validation NDCG@10 only",
            "checkpoint_name": selected_path.name,
        },
    )
    print("validation:", validation["ndcg@10"], "selected seed:", selected_seed, flush=True)
    scores = {}
    for name in config["datasets"]:
        corpus, queries, qrels = read_dataset(ROOT / "data" / f"{name}.zip", name, "test", output)
        baseline = load_json(args.source / f"{name}-test-bm25.json")
        result, timing = score_model(checkpoint, corpus, queries, baseline, config, "cuda")
        dump(output / f"{name}-run.json", result)
        scores[name] = {"metrics": evaluate(qrels, result, k=[1, 5, 10]).summary, "timing": timing}
        print(name, "NDCG@10:", scores[name]["metrics"]["ndcg@10"], flush=True)
    both = {
        str(original["config"]["seed"]): {
            "validation": old,
            "tests": {
                name: data["models"]["random_negative"]["metrics"]
                for name, data in original["datasets"].items()
            },
        },
        str(args.seed): {
            "validation": validation,
            "tests": {name: data["metrics"] for name, data in scores.items()},
        },
    }
    gates = {
        "scifact_both_seeds_improve_over_pretrained": all(
            r["tests"]["scifact"]["ndcg@10"]
            > original["datasets"]["scifact"]["models"]["pretrained"]["metrics"]["ndcg@10"]
            for r in both.values()
        ),
        "nfcorpus_no_ndcg10_drop_exceeding_002": all(
            r["tests"]["nfcorpus"]["ndcg@10"]
            >= original["datasets"]["nfcorpus"]["models"]["pretrained"]["metrics"]["ndcg@10"] - 0.02
            for r in both.values()
        ),
    }
    record = {
        "experiment": "one-additional-seed-release-check",
        "seeds": both,
        "fixed_split": {
            "seed": original["config"]["seed"],
            "split_sha256": sha256(args.source / "split.json"),
        },
        "selected_seed": selected_seed,
        "selected_checkpoint_name": selected_path.name,
        "selected_model_sha256": sha256(selected_path / "model.safetensors"),
        "training": training,
        "checks": gates,
        "passed": all(gates.values()),
        "sampling_strategy": "random_negative; selected in the original validation experiment",
        "limitations": [
            "Two seeds only, not a comprehensive stability study.",
            "Previously observed test sets are reused; no fresh unseen test is claimed.",
            "The two seeds share a fixed split but differ in sampling, shuffle and dropout.",
            "NFCorpus top-rank tradeoffs remain documented in the original report.",
        ],
    }
    dump(output / "summary.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
