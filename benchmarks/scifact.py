"""Reproducible CPU-only SciFact retrieval and evaluator audit.

Download is opt-in. Raw data stays in ignored data/; reports in ignored reports/.
BM25 and TF-IDF parameters below are fixed before reading test results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import ssl
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import ir_measures as ir
import truststore
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval_regression import Threshold, compare, evaluate
from retrieval_regression.reporting import write_html, write_json

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
EXPECTED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
ROOT = Path(__file__).resolve().parents[1]


def obtain_archive(path: Path, download: bool) -> bytes:
    if not path.exists():
        if not download:
            raise ValueError(
                "dataset not cached; rerun with --download for the public SciFact archive"
            )
        request = urllib.request.Request(URL, headers={"User-Agent": "retrieval-regression/0.1"})
        with urllib.request.urlopen(
            request, timeout=60, context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ) as response:
            data = response.read(50 * 1024 * 1024 + 1)
        if len(data) > 50 * 1024 * 1024:
            raise ValueError("archive exceeds the 50 MiB download limit")
        if hashlib.md5(data, usedforsecurity=False).hexdigest() != EXPECTED_MD5:
            raise ValueError("archive does not match the BEIR-published MD5 checksum")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.md5(data, usedforsecurity=False).hexdigest() != EXPECTED_MD5:
        raise ValueError("cached archive checksum mismatch")
    return data


def load_dataset(data: bytes):
    # Read only known members. Do not extract paths or execute downloaded code.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = {}
        for name in ("scifact/corpus.jsonl", "scifact/queries.jsonl", "scifact/qrels/test.tsv"):
            if archive.getinfo(name).file_size > 100 * 1024 * 1024:
                raise ValueError("dataset member exceeds 100 MiB")
            members[name] = archive.read(name).decode("utf-8")
    corpus, queries, qrels = {}, {}, {}
    for row in map(json.loads, members["scifact/corpus.jsonl"].splitlines()):
        identifier = row["_id"]
        if identifier in corpus:
            raise ValueError("duplicate corpus document")
        corpus[identifier] = row.get("title", "") + " " + row["text"]
    for row in map(json.loads, members["scifact/queries.jsonl"].splitlines()):
        if row["_id"] in queries:
            raise ValueError("duplicate query")
        queries[row["_id"]] = row["text"]
    reader = csv.DictReader(io.StringIO(members["scifact/qrels/test.tsv"]), delimiter="\t")
    for row in reader:
        qid, docid, grade = row["query-id"], row["corpus-id"], int(row["score"])
        if qid not in queries or docid not in corpus:
            raise ValueError("judgment references unknown query/document")
        if docid in qrels.setdefault(qid, {}):
            raise ValueError("duplicate relevance judgment")
        qrels[qid][docid] = grade
    if len(corpus) != 5183 or len(qrels) != 300:
        raise ValueError("unexpected SciFact release dimensions")
    return corpus, {qid: queries[qid] for qid in sorted(qrels)}, qrels, members


def top_run(qids, docids, score_rows, depth):
    result = {}
    for qid, scores in zip(qids, score_rows, strict=True):
        ranked = sorted(zip(docids, scores, strict=True), key=lambda pair: (-pair[1], pair[0]))
        result[qid] = {doc: float(score) for doc, score in ranked[:depth]}
    return result


def oracle_errors(qrels, run, result):
    judgments = [
        ir.Qrel(qid, doc, grade) for qid, docs in qrels.items() for doc, grade in docs.items()
    ]
    scored = []
    for qid, docs in run.items():
        ranked = sorted(docs, key=lambda doc: (-docs[doc], doc))
        # Normalize scores to explicit ranks so oracle tie policy cannot change the ordering.
        scored.extend(
            ir.ScoredDoc(qid, doc, float(len(ranked) - rank)) for rank, doc in enumerate(ranked)
        )
    mapping = {}
    for cutoff in result.k:
        mapping.update(
            {
                f"recall@{cutoff}": ir.R @ cutoff,
                f"precision@{cutoff}": ir.P @ cutoff,
                f"mrr@{cutoff}": ir.RR @ cutoff,
                f"ndcg@{cutoff}": ir.nDCG @ cutoff,
            }
        )
    reference = ir.calc_aggregate(list(mapping.values()), judgments, scored)
    differences = {
        name: abs(result.summary[name] - reference[metric]) for name, metric in mapping.items()
    }
    if max(differences.values()) > 1e-12:
        raise ValueError(f"independent reference disagreement: {differences}")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/scifact.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/scifact")
    args = parser.parse_args()
    started = time.perf_counter()
    data = obtain_archive(args.archive, args.download)
    corpus, queries, qrels, members = load_dataset(data)
    docids, qids = sorted(corpus), sorted(queries)
    texts = [corpus[doc] for doc in docids]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    analyzer = vectorizer.build_analyzer()
    begin = time.perf_counter()
    document_matrix = vectorizer.fit_transform(texts)
    query_matrix = vectorizer.transform([queries[qid] for qid in qids])
    tfidf = top_run(
        qids,
        docids,
        ((query_matrix[i] @ document_matrix.T).toarray().ravel() for i in range(len(qids))),
        100,
    )
    tfidf_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    bm25_model = BM25Okapi([analyzer(text) for text in texts], k1=1.5, b=0.75, epsilon=0.25)
    bm25 = top_run(
        qids, docids, (bm25_model.get_scores(analyzer(queries[qid])) for qid in qids), 100
    )
    bm25_seconds = time.perf_counter() - begin
    # Category split is specified by query length, not chosen from observed quality.
    groups = {
        qid: "short_query" if len(analyzer(queries[qid])) <= 12 else "long_query" for qid in qids
    }
    result = compare(
        qrels,
        bm25,
        tfidf,
        k=[1, 5, 10],
        groups=groups,
        thresholds=[Threshold("ndcg@10", max_drop=0.02)],
        bootstrap_samples=1000,
        seed=42,
    )
    reference_errors = {
        "bm25": oracle_errors(qrels, bm25, result.baseline),
        "tfidf": oracle_errors(qrels, tfidf, result.candidate),
    }
    removed = qids[::10]
    faulty = {qid: docs for qid, docs in bm25.items() if qid not in set(removed)}
    fault_result = compare(
        qrels, bm25, faulty, k=[10], thresholds=[Threshold("ndcg@10", max_drop=0)]
    )
    if fault_result.passed:
        raise ValueError("injected missing-query fault did not trigger its gate")
    naive_survivor = evaluate({qid: qrels[qid] for qid in faulty}, faulty, k=[10])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("qrels", qrels), ("bm25", bm25), ("tfidf", tfidf), ("groups", groups)):
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    write_json(result, args.output / "comparison.json")
    write_html(result, args.output / "comparison.html")
    write_json(fault_result, args.output / "injected-missing-query.json")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    summary = {
        "experiment": "scifact-lexical-retrieval-and-evaluator-audit",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(
                [*ROOT.glob("src/retrieval_regression/*.py"), Path(__file__).resolve()]
            )
        },
        "git_head": git.stdout.strip(),
        "git_dirty": bool(dirty.stdout.strip()),
        "dataset": {
            "source_url": URL,
            "archive_sha256": hashlib.sha256(data).hexdigest(),
            "archive_md5": EXPECTED_MD5,
            "split": "test",
            "documents": len(corpus),
            "queries": len(qrels),
            "judgments": sum(map(len, qrels.values())),
            "member_sha256": {
                name: hashlib.sha256(text.encode()).hexdigest() for name, text in members.items()
            },
        },
        "configuration": {
            "depth": 100,
            "k": [1, 5, 10],
            "seed": 42,
            "bm25": {
                "implementation": "rank_bm25.BM25Okapi",
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
            },
            "tfidf": {"sublinear_tf": True, "norm": "l2", "stop_words": "english"},
            "shared_analyzer": "sklearn word tokens of length >=2, lowercase, English stopwords",
            "group_rule": "short_query <=12 analyzed tokens, otherwise long_query",
            "bootstrap_samples": 1000,
            "max_absolute_ndcg10_drop": 0.02,
        },
        "metrics": {
            "bm25": result.baseline.summary,
            "tfidf": result.candidate.summary,
            "delta": result.delta,
        },
        "gates_passed": result.passed,
        "independent_reference_max_abs_error": {
            name: max(values.values()) for name, values in reference_errors.items()
        },
        "injected_fault": {
            "kind": "deliberately omit every tenth query in sorted ID order",
            "omitted_queries": len(removed),
            "gate_detected": not fault_result.passed,
            "full_query_count": fault_result.candidate.query_count,
            "faulty_ndcg10": fault_result.candidate.summary["ndcg@10"],
            "incorrect_survivor_only_ndcg10": naive_survivor.summary["ndcg@10"],
        },
        "timing_seconds": {
            "tfidf_build_and_retrieve": tfidf_seconds,
            "bm25_build_and_retrieve": bm25_seconds,
            "whole_script": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in (
                    "retrieval-regression",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "rank-bm25",
                    "ir-measures",
                    "pytrec-eval-terrier",
                    "truststore",
                )
            },
        },
        "limitations": [
            "One English scientific retrieval dataset; no neural retriever or training.",
            "Fixed untuned lexical baselines; not official BEIR leaderboard configurations.",
            "Incomplete judgments; unjudged documents are assigned zero relevance.",
            "Injected faults demonstrate detection, not natural model improvement.",
            "Wall times are local observations, not controlled performance comparisons.",
            "Original and mirror license metadata differ; raw data is not redistributed.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
