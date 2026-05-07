from __future__ import annotations

# Model family fingerprints for self-preference detection
_FAMILIES: dict[str, list[str]] = {
    "claude": ["claude"],
    "gpt": ["gpt-4", "gpt-3.5", "o1", "o3", "o4"],
    "gemini": ["gemini"],
    "llama": ["llama", "meta-llama"],
    "mistral": ["mistral", "mixtral"],
    "command": ["command-r"],
}


def _family(model: str) -> str | None:
    lower = model.lower()
    for family, patterns in _FAMILIES.items():
        if any(p in lower for p in patterns):
            return family
    return None


def detect_self_preference(judge_model: str, evaluated_model: str | None) -> bool:
    """True when judge and evaluated model are from the same family."""
    if not evaluated_model:
        return False
    return _family(judge_model) == _family(evaluated_model) is not None


def verbosity_adjust(score: float, output: str, baseline_words: int = 150) -> tuple[float, bool]:
    """
    Penalize unusually long outputs that may have inflated scores due to verbosity bias.
    Returns (adjusted_score, was_flagged).
    """
    word_count = len(output.split())
    ratio = word_count / max(baseline_words, 1)

    if ratio <= 2.5:
        return score, False

    # Soft penalty: 1% per 10 words over 2.5x baseline, capped at 8%
    excess = ratio - 2.5
    penalty = min(0.08, excess * 0.004)
    return round(max(0.0, score - penalty), 4), True


def position_bias_adjust(
    score_ab: float,
    score_ba: float,
    winner_ab: str,
    winner_ba: str,
) -> tuple[str, float, bool]:
    """
    Given pairwise scores from both orderings (A vs B and B vs A),
    return bias-corrected winner, confidence, and whether position bias was detected.
    """
    # Flip winner_ba so both are in A/B frame
    flipped = {"a": "b", "b": "a", "tie": "tie"}
    winner_ba_corrected = flipped[winner_ba]

    bias_detected = winner_ab != winner_ba_corrected

    if winner_ab == winner_ba_corrected:
        # Both agree — high confidence
        avg_margin = abs(score_ab - 0.5) + abs(score_ba - 0.5)
        confidence = min(1.0, 0.6 + avg_margin * 0.4)
        return winner_ab, round(confidence, 3), False

    # Disagreement — position bias likely, call tie with low confidence
    return "tie", 0.5, True


def compute_agreement(scores: list[float]) -> float:
    """
    Inter-judge agreement score (0-1). Uses normalized mean absolute deviation.
    1.0 = all judges agree exactly. 0.0 = maximum disagreement.
    """
    if len(scores) < 2:
        return 1.0
    mean = sum(scores) / len(scores)
    mad = sum(abs(s - mean) for s in scores) / len(scores)
    # Max possible MAD for [0,1] scores is 0.5
    return round(max(0.0, 1.0 - (mad / 0.5)), 4)
