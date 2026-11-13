"""Deterministic, local regression testing for retrieval results."""

from .adapters import InputFormatError
from .comparison import compare
from .evaluation import evaluate
from .io import load_qrels, load_run
from .models import ComparisonResult, EvaluationResult, Threshold
from .reranking import rerank

__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "InputFormatError",
    "Threshold",
    "compare",
    "evaluate",
    "load_qrels",
    "load_run",
    "rerank",
]
__version__ = "0.4.0"
