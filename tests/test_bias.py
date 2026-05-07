import pytest
from touchstone.bias import (
    detect_self_preference,
    verbosity_adjust,
    position_bias_adjust,
    compute_agreement,
)


def test_self_preference_same_family():
    assert detect_self_preference("claude-sonnet-4-6", "claude-opus-4-7") is True


def test_self_preference_different_family():
    assert detect_self_preference("claude-sonnet-4-6", "gpt-4o") is False


def test_self_preference_no_evaluated_model():
    assert detect_self_preference("claude-sonnet-4-6", None) is False


def test_verbosity_short_output():
    score, flagged = verbosity_adjust(0.8, "Short answer.")
    assert score == 0.8
    assert flagged is False


def test_verbosity_long_output():
    long_output = " ".join(["word"] * 500)
    score, flagged = verbosity_adjust(0.8, long_output)
    assert score < 0.8
    assert flagged is True


def test_position_bias_agreement():
    winner, confidence, bias = position_bias_adjust(0.7, 0.3, "a", "b")
    assert winner == "a"
    assert bias is False


def test_position_bias_detected():
    winner, confidence, bias = position_bias_adjust(0.7, 0.7, "a", "a")
    assert bias is True
    assert winner == "tie"
    assert confidence == 0.5


def test_compute_agreement_identical():
    assert compute_agreement([0.8, 0.8, 0.8]) == 1.0


def test_compute_agreement_spread():
    agreement = compute_agreement([0.2, 0.8])
    assert agreement < 0.5


def test_compute_agreement_single():
    assert compute_agreement([0.7]) == 1.0
