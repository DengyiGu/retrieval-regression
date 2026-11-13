"""Strict, line-aware adapters for BEIR judgments and TREC ranking files."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path

from .models import Qrels, Run


class InputFormatError(ValueError):
    """Malformed file, with a physical line number when a row is responsible."""

    def __init__(self, path: str | Path, line_number: int | None, message: str) -> None:
        self.path = str(path)
        self.line_number = line_number
        self.message = message
        location = self.path if line_number is None else f"{self.path}:{line_number}"
        super().__init__(f"{location}: {message}")


def _rows(path: str | Path, *, tabs: bool = False) -> Iterator[tuple[int, list[str]]]:
    try:
        with Path(path).open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield line_number, line.rstrip("\r\n").split("\t") if tabs else line.split()
    except UnicodeError as exc:
        raise InputFormatError(path, None, "file must use UTF-8 encoding") from exc


def _token(value: str, path: str | Path, line: int, field: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise InputFormatError(path, line, f"{field} must be a non-empty whitespace-free token")
    return value


def _integer(value: str, path: str | Path, line: int, field: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise InputFormatError(path, line, f"{field} must be a non-negative decimal integer")
    try:
        return int(value)
    except ValueError as exc:
        raise InputFormatError(path, line, f"{field} integer is too large") from exc


def _score(value: str, path: str | Path, line: int) -> float:
    try:
        score = float(value)
    except ValueError as exc:
        raise InputFormatError(path, line, "score must be a finite number") from exc
    if not math.isfinite(score):
        raise InputFormatError(path, line, "score must be a finite number")
    return score


def _judgments(path: str | Path, *, beir: bool) -> Qrels:
    result: Qrels = {}
    first_lines: dict[str, int] = {}
    rows = _rows(path, tabs=beir)
    if beir:
        header = next(rows, None)
        if header is None or header[1] != ["query-id", "corpus-id", "score"]:
            raise InputFormatError(
                path,
                None if header is None else header[0],
                "expected BEIR header: query-id\tcorpus-id\tscore",
            )
    for line, columns in rows:
        if len(columns) != (3 if beir else 4):
            raise InputFormatError(path, line, f"expected {3 if beir else 4} columns")
        if beir:
            qid, doc, text_grade = columns
        else:
            qid, iteration, doc, text_grade = columns
            if iteration != "0":
                raise InputFormatError(path, line, "TREC qrels iteration must be 0")
        qid = _token(qid, path, line, "query ID")
        doc = _token(doc, path, line, "document ID")
        grade = _integer(text_grade, path, line, "relevance grade")
        try:
            number = float(grade)
        except OverflowError as exc:
            raise InputFormatError(
                path, line, "relevance grade exceeds finite numeric range"
            ) from exc
        docs = result.setdefault(qid, {})
        if doc in docs:
            raise InputFormatError(path, line, f"duplicate judgment for {qid!r}/{doc!r}")
        docs[doc] = number
        first_lines.setdefault(qid, line)
    if not result:
        raise InputFormatError(path, None, "qrels must contain relevance judgments")
    for qid, docs in result.items():
        if not any(grade > 0 for grade in docs.values()):
            raise InputFormatError(
                path, first_lines[qid], f"query {qid!r} needs at least one positive judgment"
            )
    return result


def read_beir_qrels(path: str | Path) -> Qrels:
    """Read headered three-column BEIR TSV with non-negative integer grades."""
    return _judgments(path, beir=True)


def read_trec_qrels(path: str | Path) -> Qrels:
    """Read four-column qid/0/doc/grade TREC qrels; reject negative/unjudged grades."""
    return _judgments(path, beir=False)


def read_trec_run(path: str | Path) -> Run:
    """Read one six-column run; validate ranks but retain score-based ranking semantics.

    Ranks must be positive and unique per query; gaps and interleaved rows are allowed.
    Scores must be non-increasing in rank order. Tied scores retain the evaluator's ID tie
    rule, not the file's declared tie order. Every row must have the same run tag.
    """
    result: Run = {}
    ranks: dict[str, dict[int, tuple[float, int]]] = {}
    tag: str | None = None
    for line, columns in _rows(path):
        if len(columns) != 6:
            raise InputFormatError(path, line, "expected 6 TREC run columns")
        qid, placeholder, doc, text_rank, text_score, row_tag = columns
        if placeholder != "Q0":
            raise InputFormatError(path, line, "second TREC run column must be Q0")
        rank = _integer(text_rank, path, line, "rank")
        if rank == 0:
            raise InputFormatError(path, line, "rank must be positive (1-based)")
        score = _score(text_score, path, line)
        if tag is not None and row_tag != tag:
            raise InputFormatError(path, line, f"mixed run tags: expected {tag!r}, got {row_tag!r}")
        tag = row_tag
        docs = result.setdefault(qid, {})
        if doc in docs:
            raise InputFormatError(path, line, f"duplicate run document for {qid!r}/{doc!r}")
        positions = ranks.setdefault(qid, {})
        if rank in positions:
            raise InputFormatError(path, line, f"duplicate rank {rank} for query {qid!r}")
        docs[doc] = score
        positions[rank] = score, line
    for qid, positions in ranks.items():
        previous = math.inf
        for rank in sorted(positions):
            score, line = positions[rank]
            if score > previous:
                raise InputFormatError(path, line, f"scores increase with rank for query {qid!r}")
            previous = score
    return result
