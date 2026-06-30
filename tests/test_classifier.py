"""
Tests for pipeline/classifier.py — classify(heuristic_score, llm_score).

All formulas tested with exact arithmetic to catch regressions.
"""
import pytest

import config
from pipeline.classifier import classify


# ---------------------------------------------------------------------------
# Dual-signal mode (llm_score is not None)
# ---------------------------------------------------------------------------

def test_weighted_score_formula_dual_signal():
    result = classify(heuristic_score=0.80, llm_score=0.85)
    expected = round(0.65 * 0.85 + 0.35 * 0.80, 4)
    assert result["weighted_score"] == expected


def test_signal_agreement_formula():
    result = classify(heuristic_score=0.80, llm_score=0.85)
    expected = round(1.0 - abs(0.85 - 0.80), 4)
    assert result["signal_agreement"] == expected


def test_raw_confidence_formula_dual_signal():
    result = classify(heuristic_score=0.80, llm_score=0.85)
    ws = round(0.65 * 0.85 + 0.35 * 0.80, 4)
    expected = round(2.0 * abs(ws - 0.5), 4)
    assert result["raw_confidence"] == expected


def test_final_confidence_score_formula_dual_signal():
    result = classify(heuristic_score=0.80, llm_score=0.85)
    ws = round(0.65 * 0.85 + 0.35 * 0.80, 4)
    agreement = round(1.0 - abs(0.85 - 0.80), 4)
    raw = round(2.0 * abs(ws - 0.5), 4)
    expected = round(raw * agreement, 4)
    assert result["final_confidence_score"] == expected


def test_llm_signal_available_true_when_llm_score_provided():
    result = classify(heuristic_score=0.5, llm_score=0.6)
    assert result["llm_signal_available"] is True


def test_signal_agreement_present_in_dual_signal_mode():
    result = classify(heuristic_score=0.5, llm_score=0.6)
    assert result["signal_agreement"] is not None


# ---------------------------------------------------------------------------
# Dual-signal: agreement effects on final confidence
# ---------------------------------------------------------------------------

def test_perfect_agreement_maximises_final_confidence():
    """Both signals at 0.9 → agreement=1.0 → no penalty."""
    result = classify(heuristic_score=0.90, llm_score=0.90)
    assert result["signal_agreement"] == 1.0
    assert result["final_confidence_score"] == result["raw_confidence"]


def test_zero_agreement_zeroes_final_confidence():
    """Signals at extremes 0.0 and 1.0 → agreement=0.0 → final=0."""
    result = classify(heuristic_score=0.0, llm_score=1.0)
    assert result["signal_agreement"] == 0.0
    assert result["final_confidence_score"] == 0.0


def test_low_agreement_reduces_final_confidence_below_raw():
    result = classify(heuristic_score=0.80, llm_score=0.20)
    assert result["final_confidence_score"] < result["raw_confidence"]


def test_high_agreement_keeps_final_confidence_close_to_raw():
    result = classify(heuristic_score=0.88, llm_score=0.90)
    assert result["final_confidence_score"] >= 0.9 * result["raw_confidence"]


# ---------------------------------------------------------------------------
# Single-signal mode (llm_score is None)
# ---------------------------------------------------------------------------

def test_weighted_score_equals_heuristic_score_in_single_signal():
    result = classify(heuristic_score=0.70, llm_score=None)
    assert result["weighted_score"] == 0.70


def test_single_signal_multiplier_applied_to_raw_confidence():
    result = classify(heuristic_score=0.70, llm_score=None)
    raw = round(2.0 * abs(0.70 - 0.5), 4)
    expected = round(raw * config.SINGLE_SIGNAL_MULTIPLIER, 4)
    assert result["final_confidence_score"] == expected


def test_llm_signal_available_false_when_llm_score_none():
    result = classify(heuristic_score=0.5, llm_score=None)
    assert result["llm_signal_available"] is False


def test_signal_agreement_is_none_in_single_signal_mode():
    result = classify(heuristic_score=0.5, llm_score=None)
    assert result["signal_agreement"] is None


def test_single_signal_final_confidence_lower_than_dual_for_same_scores():
    """Dual signal (perfect agreement) vs single signal with same heuristic — dual wins."""
    single = classify(heuristic_score=0.80, llm_score=None)
    dual = classify(heuristic_score=0.80, llm_score=0.80)
    assert single["final_confidence_score"] < dual["final_confidence_score"]


# ---------------------------------------------------------------------------
# Return dict shape
# ---------------------------------------------------------------------------

def test_return_dict_has_all_required_keys_dual():
    result = classify(heuristic_score=0.7, llm_score=0.8)
    for key in ("weighted_score", "signal_agreement", "raw_confidence",
                "final_confidence_score", "llm_signal_available"):
        assert key in result, f"Missing key: {key}"


def test_return_dict_has_all_required_keys_single():
    result = classify(heuristic_score=0.7, llm_score=None)
    for key in ("weighted_score", "signal_agreement", "raw_confidence",
                "final_confidence_score", "llm_signal_available"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Boundary and range checks
# ---------------------------------------------------------------------------

def test_all_scores_in_zero_to_one_range_dual_signal():
    result = classify(heuristic_score=0.65, llm_score=0.75)
    for key in ("weighted_score", "signal_agreement", "raw_confidence", "final_confidence_score"):
        val = result[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


def test_all_scores_in_zero_to_one_range_single_signal():
    result = classify(heuristic_score=0.65, llm_score=None)
    for key in ("weighted_score", "raw_confidence", "final_confidence_score"):
        val = result[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


def test_midpoint_heuristic_score_gives_zero_raw_confidence():
    result = classify(heuristic_score=0.5, llm_score=None)
    assert result["raw_confidence"] == 0.0
    assert result["final_confidence_score"] == 0.0


def test_extreme_scores_give_high_confidence_in_dual_signal():
    result = classify(heuristic_score=0.95, llm_score=0.95)
    assert result["final_confidence_score"] > 0.7


def test_extreme_human_scores_give_low_weighted_score():
    result = classify(heuristic_score=0.05, llm_score=0.05)
    assert result["weighted_score"] < 0.35


def test_specific_dual_signal_values():
    """Spot-check: heuristic=0.40, llm=0.60 — moderate, low-confidence uncertain."""
    result = classify(heuristic_score=0.40, llm_score=0.60)
    ws = round(0.65 * 0.60 + 0.35 * 0.40, 4)
    agreement = round(1.0 - abs(0.60 - 0.40), 4)
    raw = round(2.0 * abs(ws - 0.5), 4)
    final = round(raw * agreement, 4)
    assert result["weighted_score"] == ws
    assert result["signal_agreement"] == agreement
    assert result["final_confidence_score"] == final
