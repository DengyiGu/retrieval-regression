"""Dataset integrity and sampling; importing this module requires no ML framework."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
import zipfile
from pathlib import Path

from retrieval_regression import load_qrels


def normalize_query(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split_queries(qrels, queries, heldout_ids, *, seed=42, validation_fraction=0.1):
    """Group identical normalized texts; remove held-out text overlap before splitting."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if set(qrels) & set(heldout_ids):
        raise ValueError("training and held-out query IDs overlap")
    if any(
        not isinstance(queries[q], str) or not queries[q].strip()
        for q in set(qrels) | set(heldout_ids)
    ):
        raise ValueError("query text must be non-empty")
    heldout_text = {normalize_query(queries[q]) for q in heldout_ids}
    removed = sorted(q for q in qrels if normalize_query(queries[q]) in heldout_text)
    train, validation = [], []
    for qid in sorted(set(qrels) - set(removed)):
        text = normalize_query(queries[qid])
        value = int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:8], 16) / 2**32
        (validation if value < validation_fraction else train).append(qid)
    if not train or not validation:
        raise ValueError("not enough query groups for non-empty training and validation sets")
    return {"train": train, "validation": validation, "excluded_text_overlap": removed}


def sample_pairs(qrels, candidates, corpus_ids, *, mode, negatives_per_positive=3, seed=42):
    """Select weak negatives without replacement, excluding ALL known positives per query."""
    if mode not in {"random_negative", "hard_negative"}:
        raise ValueError("unknown negative sampling mode")
    if (
        isinstance(negatives_per_positive, bool)
        or not isinstance(negatives_per_positive, int)
        or negatives_per_positive <= 0
    ):
        raise ValueError("negatives_per_positive must be positive")
    universe = set(corpus_ids)
    pairs = []
    for qid, judgments in sorted(qrels.items()):
        positives = {doc for doc, grade in judgments.items() if grade > 0}
        if not positives or not positives <= universe:
            raise ValueError("each query needs positive documents in the corpus")
        count = negatives_per_positive * len(positives)
        if mode == "random_negative":
            pool = sorted(universe - positives)
            rng = random.Random(f"{seed}:{qid}")
            if len(pool) < count:
                raise ValueError("not enough negative candidates")
            negatives = rng.sample(pool, count)
        else:
            run = candidates[qid]
            pool = sorted(
                (doc for doc in run if doc not in positives), key=lambda doc: (-run[doc], doc)
            )
            if len(pool) < count or any(doc not in universe for doc in pool):
                raise ValueError("not enough valid hard-negative candidates")
            negatives = pool[:count]
        pairs.extend(
            {"qid": qid, "docid": doc, "label": 1, "source": "positive_judgment"}
            for doc in sorted(positives)
        )
        pairs.extend(
            {
                "qid": qid,
                "docid": doc,
                "label": 0,
                "source": "judged_negative" if doc in judgments else "unjudged_" + mode,
            }
            for doc in negatives
        )
    return pairs


def read_dataset(path, name, split, output):
    """Read known BEIR archive members, checking duplicate IDs and judgment references."""
    with zipfile.ZipFile(path) as archive:
        members = {}
        for suffix in ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv"):
            entry = archive.getinfo(f"{name}/{suffix}")
            if entry.file_size > 100 * 1024 * 1024:
                raise ValueError("dataset member exceeds 100 MiB")
            members[suffix] = archive.read(entry)
    corpus, queries = {}, {}
    for raw in members["corpus.jsonl"].decode("utf-8").splitlines():
        row = json.loads(raw)
        if row["_id"] in corpus:
            raise ValueError("duplicate corpus ID")
        corpus[row["_id"]] = row.get("title", "") + " " + row["text"]
    for raw in members["queries.jsonl"].decode("utf-8").splitlines():
        row = json.loads(raw)
        if row["_id"] in queries:
            raise ValueError("duplicate query ID")
        queries[row["_id"]] = row["text"]
    output.mkdir(parents=True, exist_ok=True)
    qrels_path = output / f"{name}-{split}-qrels.tsv"
    qrels_path.write_bytes(members[f"qrels/{split}.tsv"])
    qrels = load_qrels(qrels_path, format="beir")
    if not set(qrels) <= set(queries) or any(
        not set(docs) <= set(corpus) for docs in qrels.values()
    ):
        raise ValueError("judgment references an unknown query or document")
    return corpus, queries, qrels


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
