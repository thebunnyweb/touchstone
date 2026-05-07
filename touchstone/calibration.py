from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .judge import Judge
    from .rubric import Rubric


_DEFAULT_STORE = Path(".touchstone") / "calibration.json"


class Calibration:
    """
    Tracks judge score drift over time.

    Baseline a judge on a fixed dataset, then re-run periodically to detect
    when a model has been silently updated and its scoring behavior has shifted.

    Usage::

        cal = Calibration(judge, store=".touchstone/calibration.json")
        await cal.baseline(samples, rubric)   # run once, store scores
        report = await cal.check(samples, rubric)  # compare against baseline
        print(report.drift_score)  # 0.0 = no drift, 1.0 = complete shift
    """

    def __init__(self, judge: "Judge", store: str | Path = _DEFAULT_STORE) -> None:
        self._judge = judge
        self._store = Path(store)

    async def baseline(self, samples: list[str], rubric: "Rubric | list[str] | None" = None) -> None:
        """Score all samples and store as baseline."""
        scores = []
        for sample in samples:
            result = await self._judge.score(sample, rubric)
            scores.append(result.adjusted_score)

        self._store.parent.mkdir(parents=True, exist_ok=True)
        data = self._load()
        model = self._judge.model_id
        data[model] = {
            "scores": scores,
            "mean": round(sum(scores) / len(scores), 4),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(samples),
        }
        self._store.write_text(json.dumps(data, indent=2))
        print(f"Baseline recorded for {model}: mean={data[model]['mean']:.3f} over {len(samples)} samples.")

    async def check(self, samples: list[str], rubric: "Rubric | list[str] | None" = None) -> "DriftReport":
        """Compare current scores against stored baseline."""
        model = self._judge.model_id
        data = self._load()

        if model not in data:
            raise ValueError(f"No baseline for {model}. Run calibrate.baseline() first.")

        baseline = data[model]
        current_scores = []
        for sample in samples:
            result = await self._judge.score(sample, rubric)
            current_scores.append(result.adjusted_score)

        current_mean = sum(current_scores) / len(current_scores)
        baseline_mean = baseline["mean"]
        drift = abs(current_mean - baseline_mean)

        return DriftReport(
            model=model,
            baseline_mean=baseline_mean,
            current_mean=round(current_mean, 4),
            drift_score=round(drift, 4),
            drifted=drift > 0.05,
            baseline_recorded_at=baseline["recorded_at"],
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def _load(self) -> dict:
        if self._store.exists():
            return json.loads(self._store.read_text())
        return {}


class DriftReport:
    def __init__(
        self,
        model: str,
        baseline_mean: float,
        current_mean: float,
        drift_score: float,
        drifted: bool,
        baseline_recorded_at: str,
        checked_at: str,
    ) -> None:
        self.model = model
        self.baseline_mean = baseline_mean
        self.current_mean = current_mean
        self.drift_score = drift_score
        self.drifted = drifted
        self.baseline_recorded_at = baseline_recorded_at
        self.checked_at = checked_at

    def __repr__(self) -> str:
        status = "DRIFTED ⚠" if self.drifted else "stable ✓"
        return (
            f"DriftReport({self.model} | {status} | "
            f"baseline={self.baseline_mean:.3f} → current={self.current_mean:.3f} | "
            f"drift={self.drift_score:.3f})"
        )
