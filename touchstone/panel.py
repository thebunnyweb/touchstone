from __future__ import annotations

import asyncio
from typing import Union

from .bias import compute_agreement
from .judge import Judge, _resolve_provider
from .models import JudgeResult, PanelResult
from .providers.base import BaseProvider
from .rubric import Rubric


class Panel:
    """
    Multi-model judge panel. Runs evals in parallel, computes consensus,
    detects inter-judge disagreement, and flags disputed outputs.

    Usage::

        panel = Panel(models=["claude-sonnet-4-6", "gpt-4o"])
        result = await panel.evaluate(output, rubric=["accuracy", "tone"])
        print(result.consensus_score, result.agreement, result.disputed)
    """

    def __init__(
        self,
        models: list[str] | None = None,
        providers: list[BaseProvider] | None = None,
        threshold: float = 0.7,
        dispute_threshold: float = 0.7,
        evaluated_model: str | None = None,
    ) -> None:
        if providers:
            self._judges = [
                Judge(provider=p, threshold=threshold, evaluated_model=evaluated_model)
                for p in providers
            ]
        elif models:
            self._judges = [
                Judge(model=m, threshold=threshold, evaluated_model=evaluated_model)
                for m in models
            ]
        else:
            raise ValueError("Provide either models= or providers=")
        self._threshold = threshold
        self._dispute_threshold = dispute_threshold

    async def evaluate(
        self,
        output: str,
        rubric: Union[Rubric, list[str], None] = None,
    ) -> PanelResult:
        """Evaluate output across all panel judges in parallel."""
        results: list[JudgeResult] = await asyncio.gather(
            *[j.score(output, rubric) for j in self._judges]
        )

        scores = [r.adjusted_score for r in results]
        consensus = round(sum(scores) / len(scores), 4)
        agreement = compute_agreement(scores)
        disputed = agreement < self._dispute_threshold

        individual = {r.model: r for r in results}

        score_summary = ", ".join(f"{r.model}={r.adjusted_score:.2f}" for r in results)
        reasoning = f"Panel of {len(results)} judges. Scores: {score_summary}. Agreement: {agreement:.2f}."
        if disputed:
            reasoning += " ⚠ High disagreement — consider human review."

        return PanelResult(
            consensus_score=consensus,
            agreement=agreement,
            disputed=disputed,
            individual_results=individual,
            reasoning=reasoning,
            passed=consensus >= self._threshold,
        )
