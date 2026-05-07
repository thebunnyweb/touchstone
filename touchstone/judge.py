from __future__ import annotations

import json
import re
from typing import Union

from .bias import detect_self_preference, verbosity_adjust, position_bias_adjust
from .models import JudgeResult, CriterionScore, ComparisonResult
from .providers.base import BaseProvider
from .rubric import Rubric

_SCORE_PROMPT = """\
You are a calibrated LLM evaluator. Score the following AI-generated output.

<output>
{output}
</output>

{rubric_section}

Respond with a single JSON object and nothing else:
{{
  "score": <float 0.0-1.0, overall weighted score>,
  "confidence": <float 0.0-1.0, how confident you are in your judgment>,
  "reasoning": "<2-3 sentences explaining your score>",
  "criteria_scores": [
    {{"name": "<criterion name>", "score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
  ]
}}

Scoring guide: 1.0 = perfect, 0.8 = good with minor issues, 0.6 = acceptable, 0.4 = poor, 0.0 = completely fails.
Be calibrated — avoid clustering around 0.7-0.8. Penalize real problems clearly."""

_COMPARE_PROMPT = """\
You are a calibrated LLM evaluator. Compare two AI-generated outputs (A and B).

<output_a>
{output_a}
</output_a>

<output_b>
{output_b}
</output_b>

{rubric_section}

Respond with a single JSON object and nothing else:
{{
  "winner": "<'a', 'b', or 'tie'>",
  "score_a": <float 0.0-1.0>,
  "score_b": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentences>"
}}"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response:\n{text}")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from judge: {e}\nRaw: {text}") from e


def _resolve_provider(model: str | None, provider: BaseProvider | None) -> BaseProvider:
    if provider is not None:
        return provider
    if model is None:
        model = "claude-sonnet-4-6"
    lower = model.lower()
    if "claude" in lower:
        from .providers.anthropic import AnthropicProvider
        return AnthropicProvider(model=model)
    if any(k in lower for k in ("gpt", "o1", "o3", "o4")):
        from .providers.openai import OpenAIProvider
        return OpenAIProvider(model=model)
    if "gemini" in lower:
        from .providers.openai import OpenAIProvider
        return OpenAIProvider(model=model, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    from .providers.ollama import OllamaProvider
    return OllamaProvider(model=model)


class Judge:
    """
    Single-model LLM judge with automatic bias detection and correction.

    Usage::

        judge = Judge(model="claude-sonnet-4-6")
        result = await judge.score(output, rubric=["factual", "concise"])
        print(result.adjusted_score, result.bias_flags)
    """

    def __init__(
        self,
        model: str | None = None,
        provider: BaseProvider | None = None,
        threshold: float = 0.7,
        evaluated_model: str | None = None,
    ) -> None:
        self._provider = _resolve_provider(model, provider)
        self._threshold = threshold
        self._evaluated_model = evaluated_model

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    async def score(
        self,
        output: str,
        rubric: Union[Rubric, list[str], None] = None,
    ) -> JudgeResult:
        """Score a single output. Returns bias-corrected JudgeResult."""
        if rubric is None:
            rubric = Rubric().add("overall quality", "Accuracy, helpfulness, and clarity")
        elif isinstance(rubric, list):
            rubric = Rubric.from_list(rubric)

        prompt = _SCORE_PROMPT.format(
            output=output,
            rubric_section=rubric.to_prompt_section(),
        )

        raw_text = await self._provider.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = _parse_json(raw_text)

        raw_score = float(raw.get("score", 0.5))
        bias_flags: list[str] = []

        if detect_self_preference(self._provider.model_id, self._evaluated_model):
            bias_flags.append("self_preference")

        adjusted_score, verbose = verbosity_adjust(raw_score, output)
        if verbose:
            bias_flags.append("verbosity")

        criteria_scores = [
            CriterionScore(
                name=cs.get("name", ""),
                score=float(cs.get("score", 0.5)),
                reasoning=cs.get("reasoning", ""),
            )
            for cs in raw.get("criteria_scores", [])
        ]

        return JudgeResult(
            score=raw_score,
            adjusted_score=adjusted_score,
            reasoning=raw.get("reasoning", ""),
            confidence=float(raw.get("confidence", 1.0)),
            criteria_scores=criteria_scores,
            bias_flags=bias_flags,
            passed=adjusted_score >= self._threshold,
            model=self._provider.model_id,
        )

    async def compare(
        self,
        output_a: str,
        output_b: str,
        rubric: Union[Rubric, list[str], None] = None,
    ) -> ComparisonResult:
        """
        Pairwise comparison with automatic position-bias correction.
        Runs A vs B and B vs A, then reconciles.
        """
        if rubric is None:
            rubric = Rubric().add("overall quality", "Accuracy, helpfulness, and clarity")
        elif isinstance(rubric, list):
            rubric = Rubric.from_list(rubric)

        async def _run(a: str, b: str) -> dict:
            prompt = _COMPARE_PROMPT.format(
                output_a=a, output_b=b,
                rubric_section=rubric.to_prompt_section(),
            )
            text = await self._provider.complete(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
            return _parse_json(text)

        import asyncio
        r_ab, r_ba = await asyncio.gather(_run(output_a, output_b), _run(output_b, output_a))

        winner, confidence, bias_detected = position_bias_adjust(
            score_ab=float(r_ab.get("score_a", 0.5)),
            score_ba=float(r_ba.get("score_a", 0.5)),
            winner_ab=r_ab.get("winner", "tie"),
            winner_ba=r_ba.get("winner", "tie"),
        )

        return ComparisonResult(
            winner=winner,
            confidence=confidence,
            reasoning=r_ab.get("reasoning", ""),
            score_a=float(r_ab.get("score_a", 0.5)),
            score_b=float(r_ab.get("score_b", 0.5)),
            position_bias_detected=bias_detected,
        )
