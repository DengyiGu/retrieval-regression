"""Deterministic, local regression testing for retrieval results."""

from .adapters import InputFormatError
from .comparison import compare
from .evaluation import evaluate
from .io import load_qrels, load_run
from .models import ComparisonResult, EvaluationResult, Threshold

__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "InputFormatError",
    "Threshold",
    "compare",
    "evaluate",
    "load_qrels",
    "load_run",
]
__version__ = "0.2.0"
