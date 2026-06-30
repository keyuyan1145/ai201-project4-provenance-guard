"""
Tests for pipeline/llm_signal.py — compute_llm_score.

Groq is mocked at the class level in every test so no real API calls are made.
time.sleep is mocked to keep retry tests fast.
"""
from unittest.mock import MagicMock, call

import pytest

import config
from pipeline.llm_signal import compute_llm_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_groq_response(content: str) -> MagicMock:
    """Return a mock Groq ChatCompletion response with the given content string."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp


def _patch_groq(monkeypatch, side_effect=None, return_value=None) -> MagicMock:
    """Patch pipeline.llm_signal.Groq and return the mock client instance."""
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.chat.completions.create.side_effect = side_effect
    elif return_value is not None:
        mock_client.chat.completions.create.return_value = return_value
    monkeypatch.setattr("pipeline.llm_signal.Groq", lambda: mock_client)
    return mock_client


def _no_sleep(monkeypatch) -> MagicMock:
    """Replace time.sleep with a no-op and return the mock for call inspection."""
    mock_sleep = MagicMock()
    monkeypatch.setattr("pipeline.llm_signal.time.sleep", mock_sleep)
    return mock_sleep


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_returns_float_for_valid_response(monkeypatch):
    payload = '{"ai_probability": 0.85, "reasoning": "test", "key_signals": ["a", "b"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    result = compute_llm_score("some text here")
    assert result == 0.85


def test_returns_float_rounded_to_4_decimals(monkeypatch):
    payload = '{"ai_probability": 0.123456789, "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    result = compute_llm_score("some text here")
    assert result == round(0.123456789, 4)


def test_returns_zero_for_ai_probability_zero(monkeypatch):
    payload = '{"ai_probability": 0.0, "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    assert compute_llm_score("some text") == 0.0


def test_returns_one_for_ai_probability_one(monkeypatch):
    payload = '{"ai_probability": 1.0, "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    assert compute_llm_score("some text") == 1.0


def test_result_is_float_type(monkeypatch):
    payload = '{"ai_probability": 0.5, "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    result = compute_llm_score("some text")
    assert isinstance(result, float)


def test_extra_json_fields_are_ignored(monkeypatch):
    payload = '{"ai_probability": 0.72, "reasoning": "x", "key_signals": ["a"], "extra": "ignored"}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    assert compute_llm_score("some text") == 0.72


def test_whitespace_trimmed_before_json_parse(monkeypatch):
    payload = '  \n{"ai_probability": 0.33, "reasoning": "x", "key_signals": ["a"]}\n  '
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    assert compute_llm_score("some text") == 0.33


# ---------------------------------------------------------------------------
# Out-of-range clamping
# ---------------------------------------------------------------------------

def test_ai_probability_above_one_is_clamped_to_one(monkeypatch):
    payload = '{"ai_probability": 1.5, "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    _no_sleep(monkeypatch)
    result = compute_llm_score("some text")
    assert result == 1.0


def test_ai_probability_below_zero_is_clamped_to_zero(monkeypatch):
    payload = '{"ai_probability": -0.3, "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    _no_sleep(monkeypatch)
    result = compute_llm_score("some text")
    assert result == 0.0


def test_clamped_value_does_not_retry(monkeypatch):
    """Out-of-range value should be clamped and returned on first attempt, not retried."""
    payload = '{"ai_probability": 2.0, "reasoning": "x", "key_signals": ["a"]}'
    client = _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    _no_sleep(monkeypatch)
    compute_llm_score("some text")
    assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# JSON parsing failures
# ---------------------------------------------------------------------------

def test_invalid_json_returns_none_after_all_retries(monkeypatch):
    _no_sleep(monkeypatch)
    _patch_groq(monkeypatch, return_value=_make_groq_response("not json at all"))
    assert compute_llm_score("some text") is None


def test_missing_ai_probability_key_returns_none(monkeypatch):
    _no_sleep(monkeypatch)
    payload = '{"confidence": 0.8, "reasoning": "x"}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    assert compute_llm_score("some text") is None


def test_non_numeric_ai_probability_returns_none(monkeypatch):
    _no_sleep(monkeypatch)
    payload = '{"ai_probability": "high", "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    assert compute_llm_score("some text") is None


def test_null_ai_probability_returns_none(monkeypatch):
    _no_sleep(monkeypatch)
    payload = '{"ai_probability": null, "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    assert compute_llm_score("some text") is None


def test_empty_json_object_returns_none(monkeypatch):
    _no_sleep(monkeypatch)
    _patch_groq(monkeypatch, return_value=_make_groq_response("{}"))
    assert compute_llm_score("some text") is None


# ---------------------------------------------------------------------------
# API exception handling
# ---------------------------------------------------------------------------

def test_api_exception_returns_none_after_all_retries(monkeypatch):
    _no_sleep(monkeypatch)
    _patch_groq(monkeypatch, side_effect=Exception("network error"))
    assert compute_llm_score("some text") is None


def test_api_exception_retries_exactly_max_retries_times(monkeypatch):
    _no_sleep(monkeypatch)
    client = _patch_groq(monkeypatch, side_effect=Exception("network error"))
    compute_llm_score("some text")
    assert client.chat.completions.create.call_count == config.LLM_MAX_RETRIES


def test_json_failure_retries_exactly_max_retries_times(monkeypatch):
    _no_sleep(monkeypatch)
    client = _patch_groq(monkeypatch, return_value=_make_groq_response("bad json"))
    compute_llm_score("some text")
    assert client.chat.completions.create.call_count == config.LLM_MAX_RETRIES


# ---------------------------------------------------------------------------
# Retry logic — succeeds on second attempt
# ---------------------------------------------------------------------------

def test_succeeds_on_second_attempt_after_one_failure(monkeypatch):
    _no_sleep(monkeypatch)
    good_payload = '{"ai_probability": 0.70, "reasoning": "x", "key_signals": ["a"]}'
    good_response = _make_groq_response(good_payload)
    client = _patch_groq(monkeypatch, side_effect=[Exception("API error"), good_response])
    result = compute_llm_score("some text")
    assert result == 0.70
    assert client.chat.completions.create.call_count == 2


def test_succeeds_on_third_attempt_after_two_failures(monkeypatch):
    _no_sleep(monkeypatch)
    good_payload = '{"ai_probability": 0.45, "reasoning": "x", "key_signals": ["a"]}'
    good_response = _make_groq_response(good_payload)
    client = _patch_groq(
        monkeypatch,
        side_effect=[Exception("err"), Exception("err"), good_response],
    )
    result = compute_llm_score("some text")
    assert result == 0.45
    assert client.chat.completions.create.call_count == 3


def test_returns_none_when_all_three_attempts_fail(monkeypatch):
    _no_sleep(monkeypatch)
    client = _patch_groq(
        monkeypatch,
        side_effect=[Exception("err"), Exception("err"), Exception("err")],
    )
    result = compute_llm_score("some text")
    assert result is None
    assert client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# Exponential backoff delay progression
# ---------------------------------------------------------------------------

def test_sleep_called_between_retries_not_after_last(monkeypatch):
    """sleep should be called LLM_MAX_RETRIES - 1 times (not after final failure)."""
    mock_sleep = _no_sleep(monkeypatch)
    _patch_groq(monkeypatch, side_effect=Exception("err"))
    compute_llm_score("some text")
    assert mock_sleep.call_count == config.LLM_MAX_RETRIES - 1


def test_sleep_delays_increase_exponentially(monkeypatch):
    """Delays should follow base * 2^(attempt-1): 1.0, 2.0 for LLM_MAX_RETRIES=3."""
    mock_sleep = _no_sleep(monkeypatch)
    _patch_groq(monkeypatch, side_effect=Exception("err"))
    compute_llm_score("some text")
    delays = [c.args[0] for c in mock_sleep.call_args_list]
    expected = [
        config.LLM_RETRY_BASE_DELAY * (2 ** i)
        for i in range(config.LLM_MAX_RETRIES - 1)
    ]
    assert delays == expected


def test_no_sleep_on_immediate_success(monkeypatch):
    mock_sleep = _no_sleep(monkeypatch)
    payload = '{"ai_probability": 0.9, "reasoning": "x", "key_signals": ["a"]}'
    _patch_groq(monkeypatch, return_value=_make_groq_response(payload))
    compute_llm_score("some text")
    mock_sleep.assert_not_called()


def test_sleep_called_after_json_parse_failure_not_after_last(monkeypatch):
    mock_sleep = _no_sleep(monkeypatch)
    _patch_groq(monkeypatch, return_value=_make_groq_response("invalid json"))
    compute_llm_score("some text")
    assert mock_sleep.call_count == config.LLM_MAX_RETRIES - 1


def test_sleep_not_called_when_succeeds_on_retry(monkeypatch):
    """Only one sleep between attempt 1 failure and attempt 2 success."""
    mock_sleep = _no_sleep(monkeypatch)
    good = _make_groq_response('{"ai_probability": 0.6, "reasoning": "x", "key_signals": ["a"]}')
    _patch_groq(monkeypatch, side_effect=[Exception("err"), good])
    compute_llm_score("some text")
    assert mock_sleep.call_count == 1
    assert mock_sleep.call_args == call(config.LLM_RETRY_BASE_DELAY * 1.0)
