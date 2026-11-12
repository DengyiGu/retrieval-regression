"""Auto-discovered pytest integration; importing the core never imports pytest."""

from __future__ import annotations

from typing import Any

import pytest

from .comparison import compare
from .models import ComparisonResult


class RetrievalCheck:
    """Callable fixture: compare runs and fail with an actionable gate summary."""

    def __call__(self, *args: Any, **kwargs: Any) -> ComparisonResult:
        if not kwargs.get("thresholds"):
            raise ValueError("retrieval_check requires at least one explicit threshold")
        result = compare(*args, **kwargs)
        failures = [
            f"{gate.group or 'overall'} {gate.metric}: "
            f"{gate.baseline:.6f} -> {gate.candidate:.6f} "
            f"(min={gate.min_value}, max_drop={gate.max_drop})"
            for gate in result.gates
            if not gate.passed
        ]
        if failures:
            pytest.fail("Retrieval quality gates failed:\n" + "\n".join(failures), pytrace=False)
        return result


@pytest.fixture
def retrieval_check() -> RetrievalCheck:
    """Compare qrels/baseline/candidate using explicit Threshold objects."""
    return RetrievalCheck()
