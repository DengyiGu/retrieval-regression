"""Strict JSON input: reject duplicate keys rather than silently losing rankings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Threshold


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r}")


def load_json(path: str | Path) -> Any:
    """Read UTF-8 (including BOM), disallowing duplicate keys and NaN/Infinity."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def load_thresholds(path: str | Path) -> tuple[Threshold, ...]:
    config = load_json(path)
    if not isinstance(config, dict) or set(config) != {"thresholds"}:
        raise ValueError("config must contain exactly one 'thresholds' array")
    if not isinstance(config["thresholds"], list):
        raise ValueError("thresholds must be an array")
    result: list[Threshold] = []
    for entry in config["thresholds"]:
        if (
            not isinstance(entry, dict)
            or "metric" not in entry
            or set(entry) - {"metric", "min_value", "max_drop", "group"}
        ):
            raise ValueError(
                "each threshold needs metric and only min_value/max_drop/group options"
            )
        if not isinstance(entry["metric"], str):
            raise ValueError("threshold metric must be a string")
        if "group" in entry and entry["group"] is not None and not isinstance(entry["group"], str):
            raise ValueError("threshold group must be a string or null")
        result.append(Threshold(**entry))
    return tuple(result)
