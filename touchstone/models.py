from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    name: str
    score: float
    reasoning: str


class JudgeResult(BaseModel):
    score: float = Field(description="Raw score from judge (0-1)")
    adjusted_score: float = Field(description="Bias-corrected score (0-1)")
    reasoning: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    criteria_scores: list[CriterionScore] = Field(default_factory=list)
    bias_flags: list[str] = Field(default_factory=list)
    passed: bool
    model: str

    @property
    def weighted_score(self) -> float:
        return self.adjusted_score


class PanelResult(BaseModel):
    consensus_score: float = Field(description="Mean adjusted score across all judges")
    agreement: float = Field(description="Inter-judge agreement (0-1, higher = more agreement)")
    disputed: bool = Field(description="True when agreement < 0.7")
    individual_results: dict[str, JudgeResult]
    reasoning: str
    passed: bool

    @property
    def models(self) -> list[str]:
        return list(self.individual_results.keys())


class ComparisonResult(BaseModel):
    winner: str = Field(description="'a', 'b', or 'tie'")
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    score_a: float
    score_b: float
    position_bias_detected: bool = False
