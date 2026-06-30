"""
Tests for pipeline/heuristic_signal.py — Signal 1 (Statistical Heuristics).

Each sub-feature is tested in isolation with hand-crafted inputs designed to
produce a known high or low score, then compute_heuristic_score is tested end-
to-end including the short-text cap and the parallel execution path.
"""
import config
from pipeline.heuristic_signal import (
    _score_punctuation_range,
    _score_sentence_length_uniformity,
    _score_structural_openers,
    _score_vocab_marker_density,
    compute_heuristic_score,
)

# ---------------------------------------------------------------------------
# Shared test inputs
# ---------------------------------------------------------------------------

# Well over MIN_TEXT_LENGTH (80 words), saturated with AI markers
AI_TEXT = (
    "Delving into the comprehensive realm of robust and seamless solutions, it is worth noting "
    "that leveraging these crucial and invaluable frameworks is pivotal. "
    "Moreover, the nuanced approach provides notably straightforward pathways to success. "
    "Furthermore, it is important to recognize that modern enterprises require sophisticated solutions. "
    "Additionally, these comprehensive methodologies ensure seamless integration throughout. "
    "In conclusion, the pivotal role of robust systems cannot be overstated. "
    "It is certainly worth noting that nuanced and comprehensive strategies are crucial. "
    "Leveraging these invaluable insights is pivotal for robust and seamless modern enterprises."
)

# Well over MIN_TEXT_LENGTH, no AI markers, varied punctuation, casual register
HUMAN_TEXT = (
    "I was out last tuesday with some friends and we got into this long argument about "
    "whether hot dogs are sandwiches — nobody could agree! One guy kept insisting they are their own "
    "category, which is honestly kind of fair. Anyway we ended up just dropping it. "
    "Super silly debate but it was a good time overall. I have had this conversation before and "
    "it never goes anywhere productive. Some questions just do not have clear answers I guess."
)


# ---------------------------------------------------------------------------
# _score_vocab_marker_density
# ---------------------------------------------------------------------------

def test_vocab_density_zero_when_no_markers_present():
    text = "I went to the store today and bought some apples and bananas for lunch."
    assert _score_vocab_marker_density(text) == 0.0


def test_vocab_density_high_with_many_ai_words():
    text = "delve comprehensive robust seamless nuanced leverage crucial invaluable pivotal certainly"
    score = _score_vocab_marker_density(text)
    assert score > 0.8


def test_vocab_density_detects_ai_phrase():
    text = "It is worth noting that this approach works well in most situations."
    score = _score_vocab_marker_density(text)
    assert score > 0.0


def test_vocab_density_detects_sentence_starters():
    # "Moreover" and "Furthermore" are AI starter words — should add to count
    text = "Moreover this is important to consider. Furthermore it should be noted as well."
    score = _score_vocab_marker_density(text)
    assert score > 0.0


def test_vocab_density_capped_at_one():
    # Pathologically dense — must not exceed 1.0
    dense = " ".join(["delve", "robust", "crucial", "leverage", "pivotal"] * 20)
    assert _score_vocab_marker_density(dense) == 1.0


def test_vocab_density_returns_zero_for_empty_text():
    assert _score_vocab_marker_density("") == 0.0


def test_vocab_density_in_range_for_all_sample_texts():
    for text in [AI_TEXT, HUMAN_TEXT, "Hello world.", ""]:
        score = _score_vocab_marker_density(text)
        assert 0.0 <= score <= 1.0, f"Out of [0,1]: {score!r} for {text[:40]!r}"


# ---------------------------------------------------------------------------
# _score_sentence_length_uniformity
# ---------------------------------------------------------------------------

def test_uniformity_high_for_equal_length_sentences():
    # Three sentences of 6-7 words each → very low CV → score near 1.0
    text = "The cat sat on the mat. The dog ran in the park. The sun lit the sky."
    score = _score_sentence_length_uniformity(text)
    assert score > 0.7


def test_uniformity_low_for_extremely_varied_lengths():
    # One 1-word sentence and one very long sentence → huge CV → score near 0
    long_part = " ".join(["word"] * 30)
    text = f"Hi. {long_part}."
    score = _score_sentence_length_uniformity(text)
    assert score < 0.3


def test_uniformity_returns_half_for_single_sentence():
    # Can't compute std dev on one sentence — should return 0.5 (neutral)
    score = _score_sentence_length_uniformity("This is just one sentence with no terminator")
    assert score == 0.5


def test_uniformity_in_range_for_all_sample_texts():
    for text in [AI_TEXT, HUMAN_TEXT]:
        score = _score_sentence_length_uniformity(text)
        assert 0.0 <= score <= 1.0, f"Out of [0,1]: {score!r}"


# ---------------------------------------------------------------------------
# _score_punctuation_range
# ---------------------------------------------------------------------------

def test_punctuation_range_is_one_with_no_special_characters():
    text = "This is a plain sentence. And another one here. No special chars at all."
    assert _score_punctuation_range(text) == 1.0


def test_punctuation_range_is_zero_with_four_or_more_special_types():
    # ?, !, (, ), :, —, ;, … → 8 distinct types → score = 0.0
    text = "Really? Yes! But (maybe) not: here — and there; ever…"
    assert _score_punctuation_range(text) == 0.0


def test_punctuation_range_partial_with_two_special_types():
    # One "?" → 1 type → score = 1.0 - 1/4 = 0.75
    text = "Is this right? Yes indeed."
    score = _score_punctuation_range(text)
    assert 0.0 < score < 1.0


def test_punctuation_range_in_range_for_all_sample_texts():
    for text in [AI_TEXT, HUMAN_TEXT, "Hello world."]:
        score = _score_punctuation_range(text)
        assert 0.0 <= score <= 1.0, f"Out of [0,1]: {score!r}"


# ---------------------------------------------------------------------------
# _score_structural_openers
# ---------------------------------------------------------------------------

def test_structural_openers_is_one_when_all_sentences_use_openers():
    text = (
        "However, this is true. "
        "Therefore, we must act. "
        "For example, consider this. "
        "In contrast, others disagree."
    )
    assert _score_structural_openers(text) == 1.0


def test_structural_openers_is_zero_when_no_openers_used():
    text = "The cat sat here. Dogs are great. Python is fun. I like code."
    assert _score_structural_openers(text) == 0.0


def test_structural_openers_partial_half():
    # 2 opener sentences out of 4 → 0.5
    text = (
        "However, this is true. "
        "The cat sat here. "
        "As a result, we act. "
        "Dogs are great."
    )
    assert _score_structural_openers(text) == 0.5


def test_structural_openers_returns_zero_for_empty_text():
    assert _score_structural_openers("") == 0.0


def test_structural_openers_in_range_for_all_sample_texts():
    for text in [AI_TEXT, HUMAN_TEXT, "Hello."]:
        score = _score_structural_openers(text)
        assert 0.0 <= score <= 1.0, f"Out of [0,1]: {score!r}"


# ---------------------------------------------------------------------------
# compute_heuristic_score — return shape
# ---------------------------------------------------------------------------

def test_returns_dict_with_all_required_top_level_keys():
    result = compute_heuristic_score("Some sample text for testing.")
    for key in ("heuristic_score", "sub_scores", "word_count", "is_short_text"):
        assert key in result, f"Missing key: {key}"


def test_sub_scores_dict_contains_all_four_feature_keys():
    result = compute_heuristic_score("Some sample text.")
    expected = {
        "vocab_marker_density",
        "sentence_length_uniformity",
        "punctuation_range",
        "structural_opener_patterns",
    }
    assert set(result["sub_scores"].keys()) == expected


def test_heuristic_score_in_range():
    assert 0.0 <= compute_heuristic_score(AI_TEXT)["heuristic_score"] <= 1.0


def test_all_sub_scores_in_range():
    result = compute_heuristic_score(AI_TEXT)
    for name, val in result["sub_scores"].items():
        assert 0.0 <= val <= 1.0, f"Sub-score {name}={val} outside [0,1]"


def test_heuristic_score_is_mean_of_sub_scores():
    result = compute_heuristic_score(AI_TEXT)
    expected = round(sum(result["sub_scores"].values()) / 4, 4)
    assert result["heuristic_score"] == expected


def test_word_count_is_accurate():
    text = "one two three four five"
    assert compute_heuristic_score(text)["word_count"] == 5


# ---------------------------------------------------------------------------
# compute_heuristic_score — AI vs human discrimination
# ---------------------------------------------------------------------------

def test_ai_text_scores_higher_than_human_text():
    ai_score = compute_heuristic_score(AI_TEXT)["heuristic_score"]
    human_score = compute_heuristic_score(HUMAN_TEXT)["heuristic_score"]
    assert ai_score > human_score


def test_ai_text_has_higher_vocab_density_than_human():
    ai = compute_heuristic_score(AI_TEXT)["sub_scores"]["vocab_marker_density"]
    human = compute_heuristic_score(HUMAN_TEXT)["sub_scores"]["vocab_marker_density"]
    assert ai > human


def test_human_text_has_lower_punctuation_range_score_than_plain_text():
    # HUMAN_TEXT contains em-dash and ! so it should score lower than plain AI text
    human = compute_heuristic_score(HUMAN_TEXT)["sub_scores"]["punctuation_range"]
    plain_ai = compute_heuristic_score(AI_TEXT)["sub_scores"]["punctuation_range"]
    assert plain_ai >= human


# ---------------------------------------------------------------------------
# compute_heuristic_score — short-text handling
# ---------------------------------------------------------------------------

def test_short_text_flag_is_true_below_threshold():
    short = "This is a short text."
    result = compute_heuristic_score(short)
    assert result["is_short_text"] is True
    assert result["word_count"] < config.MIN_TEXT_LENGTH


def test_short_text_flag_is_false_above_threshold():
    result = compute_heuristic_score(AI_TEXT)
    assert result["is_short_text"] is False
    assert result["word_count"] >= config.MIN_TEXT_LENGTH


def test_short_text_all_sub_scores_capped_at_half():
    short = "This is a short text."
    result = compute_heuristic_score(short)
    for name, val in result["sub_scores"].items():
        assert val <= 0.5, f"Sub-score {name}={val} exceeds 0.5 cap for short text"


def test_short_text_heuristic_score_also_capped_at_half():
    short = "This is a short text."
    assert compute_heuristic_score(short)["heuristic_score"] <= 0.5
