"""Opt-in content fingerprints for reproducible command-line evaluations."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ComparisonResult, EvaluationResult


@dataclass(frozen=True)
class InputFingerprint:
    role: str
    path: Path
    format: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        # Local absolute paths are deliberately omitted from portable manifests.
        return {
            "role": self.role,
            "name": self.path.name,
            "format": self.format,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def fingerprint(role: str, path: str | Path, format: str) -> InputFingerprint:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return InputFingerprint(role, source, format, digest.hexdigest(), size)


def verify_inputs(inputs: tuple[InputFingerprint, ...]) -> None:
    """Reject changed input content before any CLI report is written."""
    for before in inputs:
        after = fingerprint(before.role, before.path, before.format)
        if (before.sha256, before.size_bytes) != (after.sha256, after.size_bytes):
            raise ValueError(f"input changed during evaluation: {before.path}")


def make_manifest(
    result: ComparisonResult | EvaluationResult,
    inputs: tuple[InputFingerprint, ...],
    parameters: Mapping[str, Any],
    *,
    version: str,
) -> dict[str, Any]:
    canonical = json.dumps(
        result.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema_version": "1.0",
        "kind": "evaluation_manifest",
        "software": {"retrieval_regression": version, "python": platform.python_version()},
        "inputs": [source.to_dict() for source in inputs],
        "parameters": dict(parameters),
        "result_sha256": hashlib.sha256(canonical).hexdigest(),
        "result_hash_encoding": "UTF-8 JSON; sorted keys; ASCII escapes; compact separators",
        "query_count": (
            result.baseline if isinstance(result, ComparisonResult) else result
        ).query_count,
        "quality_gated": isinstance(result, ComparisonResult) and bool(result.gates),
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
