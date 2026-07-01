"""
Tests for pipeline/classifier.py — classify(heuristic_score, llm_score, word_count).

Dual-signal: adaptive weighted average; three weight tiers.
Single-signal: weighted_score = heuristic_score.
No final_confidence_score or raw_confidence — weighted_score is the single output score.
"""
import pytest

import config
from pipeline.classifier import classify


# ---------------------------------------------------------------------------
# Dual-signal: standard case (word_count ≤ 150, gap ≤ 0.40) → 70/30 weights
# ---------------------------------------------------------------------------

def test_standard_weights_70_30_when_short_text_and_small_gap():
    result = classify(heuristic_score=0.80, llm_score=0.85, word_count=50)
    expected = round(0.70 * 0.85 + 0.30 * 0.80, 4)
    assert result["weighted_score"] == expected


def test_signal_agreement_formula():
    result = classify(heuristic_score=0.80, llm_score=0.85, word_count=50)
    assert result["signal_agreement"] == round(1.0 - abs(0.85 - 0.80), 4)


def test_llm_signal_available_true_when_llm_score_provided():
    result = classify(heuristic_score=0.5, llm_score=0.6, word_count=50)
    assert result["llm_signal_available"] is True


def test_signal_agreement_present_in_dual_signal_mode():
    result = classify(heuristic_score=0.5, llm_score=0.6, word_count=50)
    assert result["signal_agreement"] is not None


# ---------------------------------------------------------------------------
# Dual-signal: high disagreement (gap > 0.40) → 85/15 weights
# ---------------------------------------------------------------------------

def test_high_disagreement_weights_85_15_when_gap_above_threshold():
    result = classify(heuristic_score=0.30, llm_score=0.80, word_count=50)
    expected = round(0.85 * 0.80 + 0.15 * 0.30, 4)
    assert result["weighted_score"] == expected


def test_high_disagreement_gap_exactly_at_boundary_uses_standard_weights():
    # gap = |0.80 - 0.40| = 0.40, NOT > 0.40 → standard 70/30
    result = classify(heuristic_score=0.40, llm_score=0.80, word_count=50)
    expected = round(0.70 * 0.80 + 0.30 * 0.40, 4)
    assert result["weighted_score"] == expected


def test_high_disagreement_reversed_direction_still_uses_abs_gap():
    # gap = |0.20 - 0.75| = 0.55 > 0.40
    result = classify(heuristic_score=0.75, llm_score=0.20, word_count=50)
    expected = round(0.85 * 0.20 + 0.15 * 0.75, 4)
    assert result["weighted_score"] == expected


# ---------------------------------------------------------------------------
# Dual-signal: long text (word_count > 150) → 65/35 weights (highest priority)
# ---------------------------------------------------------------------------

def test_long_text_weights_65_35_when_word_count_above_150():
    result = classify(heuristic_score=0.30, llm_score=0.80, word_count=200)
    expected = round(0.65 * 0.80 + 0.35 * 0.30, 4)
    assert result["weighted_score"] == expected


def test_long_text_priority_over_high_gap():
    # word_count > 150 overrides gap > 0.40
    result = classify(heuristic_score=0.10, llm_score=0.90, word_count=300)
    expected = round(0.65 * 0.90 + 0.35 * 0.10, 4)
    assert result["weighted_score"] == expected


def test_long_text_boundary_at_exactly_150_uses_gap_based_weights():
    # word_count == 150 is NOT > 150 → gap check applies; gap=0.50 > 0.40 → 85/15
    result = classify(heuristic_score=0.30, llm_score=0.80, word_count=150)
    expected = round(0.85 * 0.80 + 0.15 * 0.30, 4)
    assert result["weighted_score"] == expected


def test_default_word_count_zero_uses_gap_based_weights():
    # word_count defaults to 0; gap=0.05 ≤ 0.40 → standard 70/30
    result = classify(heuristic_score=0.80, llm_score=0.85)
    expected = round(0.70 * 0.85 + 0.30 * 0.80, 4)
    assert result["weighted_score"] == expected


# ---------------------------------------------------------------------------
# Dual-signal: score range and discrimination
# ---------------------------------------------------------------------------

def test_weighted_score_in_zero_to_one_range_dual_signal():
    result = classify(heuristic_score=0.65, llm_score=0.75, word_count=50)
    assert 0.0 <= result["weighted_score"] <= 1.0
    assert 0.0 <= result["signal_agreement"] <= 1.0


def test_extreme_ai_scores_give_high_weighted_score():
    result = classify(heuristic_score=0.95, llm_score=0.95, word_count=50)
    assert result["weighted_score"] > 0.7


def test_extreme_human_scores_give_low_weighted_score():
    result = classify(heuristic_score=0.05, llm_score=0.05, word_count=50)
    assert result["weighted_score"] < 0.35


def test_midpoint_scores_give_midpoint_weighted_score():
    result = classify(heuristic_score=0.5, llm_score=0.5, word_count=50)
    assert result["weighted_score"] == 0.5


# ---------------------------------------------------------------------------
# Dual-signal: return dict shape
# ---------------------------------------------------------------------------

def test_return_dict_has_required_keys_dual():
    result = classify(heuristic_score=0.7, llm_score=0.8, word_count=50)
    for key in ("weighted_score", "signal_agreement", "llm_signal_available"):
        assert key in result, f"Missing key: {key}"


def test_dual_signal_return_has_no_raw_confidence_or_final_confidence():
    result = classify(heuristic_score=0.7, llm_score=0.8, word_count=50)
    assert "raw_confidence" not in result
    assert "final_confidence_score" not in result


# ---------------------------------------------------------------------------
# Single-signal mode (llm_score is None)
# ---------------------------------------------------------------------------

def test_weighted_score_equals_heuristic_score_in_single_signal():
    result = classify(heuristic_score=0.70, llm_score=None)
    assert result["weighted_score"] == 0.70


def test_llm_signal_available_false_when_llm_score_none():
    result = classify(heuristic_score=0.5, llm_score=None)
    assert result["llm_signal_available"] is False


def test_signal_agreement_is_none_in_single_signal_mode():
    result = classify(heuristic_score=0.5, llm_score=None)
    assert result["signal_agreement"] is None


def test_midpoint_heuristic_gives_half_weighted_score_single_signal():
    result = classify(heuristic_score=0.5, llm_score=None)
    assert result["weighted_score"] == 0.5


def test_single_signal_return_has_no_raw_confidence_or_final_confidence():
    result = classify(heuristic_score=0.70, llm_score=None)
    assert "raw_confidence" not in result
    assert "final_confidence_score" not in result


def test_return_dict_has_required_keys_single():
    result = classify(heuristic_score=0.7, llm_score=None)
    for key in ("weighted_score", "signal_agreement", "llm_signal_available"):
        assert key in result, f"Missing key: {key}"


def test_weighted_score_in_zero_to_one_range_single_signal():
    result = classify(heuristic_score=0.65, llm_score=None)
    assert 0.0 <= result["weighted_score"] <= 1.0


# ---------------------------------------------------------------------------
# Spot-check exact arithmetic for each weight scenario
# ---------------------------------------------------------------------------

def test_spot_check_standard_weights():
    # gap = |0.60 - 0.40| = 0.20 ≤ 0.40, word_count=50 → 70/30
    result = classify(heuristic_score=0.40, llm_score=0.60, word_count=50)
    assert result["weighted_score"] == round(0.70 * 0.60 + 0.30 * 0.40, 4)
    assert result["signal_agreement"] == round(1.0 - abs(0.60 - 0.40), 4)


def test_spot_check_high_disagreement_weights():
    # gap = |0.90 - 0.40| = 0.50 > 0.40, word_count=80 → 85/15
    result = classify(heuristic_score=0.40, llm_score=0.90, word_count=80)
    assert result["weighted_score"] == round(0.85 * 0.90 + 0.15 * 0.40, 4)


def test_spot_check_long_text_weights():
    # word_count=200 > 150 → 65/35 regardless of gap
    result = classify(heuristic_score=0.40, llm_score=0.90, word_count=200)
    assert result["weighted_score"] == round(0.65 * 0.90 + 0.35 * 0.40, 4)
