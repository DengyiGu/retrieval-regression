"""Deterministic, local regression testing for retrieval results."""

from .comparison import compare
from .evaluation import evaluate
from .models import ComparisonResult, EvaluationResult, Threshold

__all__ = ["ComparisonResult", "EvaluationResult", "Threshold", "compare", "evaluate"]
__version__ = "0.1.0"
