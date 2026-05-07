"""
touchstone — Bias-aware LLM judge with multi-model consensus.
"""

from .judge import Judge
from .panel import Panel
from .rubric import Rubric, FACTUALITY, HELPFULNESS, SAFETY, RAG_QUALITY
from .models import JudgeResult, PanelResult, ComparisonResult
from .providers import AnthropicProvider, OpenAIProvider, OllamaProvider
from .calibration import Calibration

__version__ = "0.1.0"
__all__ = [
    "Judge",
    "Panel",
    "Rubric",
    "FACTUALITY",
    "HELPFULNESS",
    "SAFETY",
    "RAG_QUALITY",
    "JudgeResult",
    "PanelResult",
    "ComparisonResult",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "Calibration",
]
