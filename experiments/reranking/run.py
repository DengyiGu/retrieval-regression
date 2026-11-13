"""Run a fixed neural reranking ablation and evaluate on held-out domains."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from retrieval_regression import Threshold, __version__, compare, evaluate, rerank
from retrieval_regression.io import load_json
from retrieval_regression.reporting import write_html, write_json

from .data import read_dataset, sample_pairs, sha256, split_queries
from .resources import prepare_resources
from .training import ModelScorer, load_model, seed_everything, synchronize, train_model

ROOT = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def bm25_run(corpus, queries, qids, depth):
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer

    analyzer = TfidfVectorizer(stop_words="english").build_analyzer()
    docids = sorted(corpus)
    model = BM25Okapi([analyzer(corpus[d]) for d in docids], k1=1.5, b=0.75, epsilon=0.25)
    run = {}
    for qid in sorted(qids):
        scores = model.get_scores(analyzer(queries[qid]))
        indices = sorted(range(len(docids)), key=lambda i: (-scores[i], docids[i]))[:depth]
        run[qid] = {docids[i]: float(scores[i]) for i in indices}
    return run


def score_model(path, corpus, queries, candidates, config, device):
    import torch

    model, tokenizer = load_model(path, device)
    scorer = ModelScorer(model, tokenizer, device, config["max_length"])
    # Warm one fixed batch; it is not included in the timed pass.
    qid = next(iter(candidates))
    doc = next(iter(candidates[qid]))
    scorer([(queries[qid], corpus[doc])])
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    start = time.perf_counter()
    result = rerank(
        queries,
        corpus,
        candidates,
        scorer,
        top_k=config["candidates"],
        batch_size=config["inference_batch_size"],
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    record = {
        "pairs": sum(map(len, result.values())),
        "wall_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else None,
    }
    del scorer, model, tokenizer
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports/neural-reranking")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu explicitly")
    config = load_json(args.config)
    seed_everything(config["seed"])
    model_path = prepare_resources(config, ROOT, download=args.download)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        *sorted(Path(__file__).parent.glob("*.py")),
        *sorted((ROOT / "src/retrieval_regression").glob("*.py")),
    ]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sources}
    identity = {
        "config_sha256": sha256(args.config),
        "source_sha256": source_hashes,
        "archives": {name: sha256(ROOT / "data" / f"{name}.zip") for name in config["datasets"]},
    }
    plan_path = output / "identity.json"
    if plan_path.exists():
        if not args.resume or load_json(plan_path) != identity:
            raise ValueError("output already exists or experiment changed; choose a new --output")
    else:
        dump(plan_path, identity)
        dump(output / "config.json", config)
    train_corpus, queries, training_qrels = read_dataset(
        ROOT / "data/scifact.zip", "scifact", "train", output
    )
    _, _, test_qrels = read_dataset(ROOT / "data/scifact.zip", "scifact", "test", output)
    split = split_queries(
        training_qrels,
        queries,
        set(test_qrels),
        seed=config["seed"],
        validation_fraction=config["validation_fraction"],
    )
    dump(output / "split.json", split)
    train_qrels = {q: training_qrels[q] for q in split["train"]}
    validation_qrels = {q: training_qrels[q] for q in split["validation"]}
    candidate_path = output / "scifact-candidates.json"
    if candidate_path.exists():
        candidates = load_json(candidate_path)
    else:
        candidates = bm25_run(
            train_corpus, queries, set(training_qrels) | set(test_qrels), config["candidates"]
        )
        dump(candidate_path, candidates)
    validation_candidates = {q: candidates[q] for q in validation_qrels}
    print("query partitions:", {key: len(value) for key, value in split.items()}, flush=True)
    progress_path = output / "progress.json"
    progress = load_json(progress_path) if progress_path.exists() else {}
    for variant in config["variants"]:
        checkpoint = model_path if variant == "pretrained" else output / variant
        if variant in progress:
            print("resuming completed variant:", variant, flush=True)
            continue
        seed_everything(config["seed"])
        training = None
        if variant != "pretrained":
            pairs = sample_pairs(
                train_qrels,
                candidates,
                train_corpus,
                mode=variant,
                negatives_per_positive=config["negatives_per_positive"],
                seed=config["seed"],
            )
            dump(output / f"{variant}-pairs.json", pairs)
            model, tokenizer = load_model(model_path, args.device)
            print(f"training {variant}: {len(pairs)} pairs", flush=True)
            training = train_model(
                model, tokenizer, pairs, queries, train_corpus, config, args.device
            )
            checkpoint.mkdir(exist_ok=True)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            del model, tokenizer
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        run, timing = score_model(
            checkpoint, train_corpus, queries, validation_candidates, config, args.device
        )
        dump(output / f"{variant}-validation-run.json", run)
        evaluation = evaluate(validation_qrels, run, k=[1, 5, 10])
        progress[variant] = {
            "validation": evaluation.summary,
            "training": training,
            "validation_timing": timing,
            "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        }
        dump(progress_path, progress)
        print(variant, "validation NDCG@10:", evaluation.summary["ndcg@10"], flush=True)
    selected = max(
        config["variants"],
        key=lambda name: progress[name]["validation"][config["selection_metric"]],
    )
    dump(
        output / "selection.json",
        {
            "selected": selected,
            "criterion": config["selection_metric"],
            "partition": "SciFact training-derived validation only",
        },
    )
    print("validation-selected variant:", selected, flush=True)
    datasets = {}
    for name in config["datasets"]:
        corpus, eval_queries, qrels = read_dataset(
            ROOT / "data" / f"{name}.zip", name, "test", output
        )
        base_path = output / f"{name}-test-bm25.json"
        if name == "scifact":
            baseline = {q: candidates[q] for q in qrels}
        elif base_path.exists():
            baseline = load_json(base_path)
        else:
            baseline = bm25_run(corpus, eval_queries, qrels, config["candidates"])
        dump(base_path, baseline)
        runs, model_results = {}, {}
        for variant in config["variants"]:
            cache = output / f"{name}-{variant}-run.json"
            timing_path = output / f"{name}-{variant}-timing.json"
            if cache.exists() and timing_path.exists():
                runs[variant], timing = load_json(cache), load_json(timing_path)
            else:
                checkpoint = model_path if variant == "pretrained" else output / variant
                runs[variant], timing = score_model(
                    checkpoint, corpus, eval_queries, baseline, config, args.device
                )
                dump(cache, runs[variant])
                dump(timing_path, timing)
            model_results[variant] = {
                "metrics": evaluate(qrels, runs[variant], k=[1, 5, 10]).summary,
                "inference": timing,
            }
            print(
                name, variant, "NDCG@10:", model_results[variant]["metrics"]["ndcg@10"], flush=True
            )
        comparisons = {}
        for label, before, after in [
            ("bm25_to_pretrained", baseline, runs["pretrained"]),
            ("pretrained_to_random", runs["pretrained"], runs["random_negative"]),
            ("pretrained_to_hard", runs["pretrained"], runs["hard_negative"]),
        ]:
            result = compare(
                qrels,
                before,
                after,
                k=[1, 5, 10],
                thresholds=[Threshold("ndcg@10", max_drop=config["max_drop"])],
                bootstrap_samples=config["bootstrap_samples"],
                seed=config["seed"],
            )
            write_json(result, output / f"{name}-{label}.json")
            write_html(result, output / f"{name}-{label}.html")
            comparisons[label] = {
                "delta": result.delta,
                "passed": result.passed,
                "ndcg10_interval": vars(result.confidence_intervals["ndcg@10"]),
            }
        datasets[name] = {
            "queries": len(qrels),
            "documents": len(corpus),
            "judgments": sum(map(len, qrels.values())),
            "bm25": evaluate(qrels, baseline, k=[1, 5, 10]).summary,
            "models": model_results,
            "comparisons": comparisons,
        }
    summary = {
        "experiment": "neural-reranking-and-negative-sampling-ablation",
        "version": __version__,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "identity": identity,
        "split_counts": {k: len(v) for k, v in split.items()},
        "excluded_text_overlap_qids": split["excluded_text_overlap"],
        "validation_and_training": progress,
        "selected_variant": selected,
        "datasets": datasets,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "packages": {
                n: version(n) for n in ["transformers", "numpy", "scikit-learn", "rank-bm25"]
            },
        },
        "limitations": [
            "One seed and one epoch per sampling strategy.",
            "Most sampled negatives are unjudged, not confirmed irrelevant.",
            "Only exact normalized query overlap is removed; semantic overlap may remain.",
            "Pretraining-data overlap cannot be ruled out.",
            "Top-50 BM25 candidates and max 256 tokens constrain the experiment.",
            "Timings cover local tokenization/scoring, not a production service.",
            "NFCorpus source terms distinguish academic use; raw data is not redistributed.",
        ],
    }
    dump(output / "summary.json", summary)
    print("Finished; local summary:", output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
