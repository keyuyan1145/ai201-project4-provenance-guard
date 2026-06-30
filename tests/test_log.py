"""
Tests for GET /log endpoint and the audit trail written by POST /submit.
conftest.py redirects AUDIT_LOG_FILE to a per-test temp file automatically.
"""
import json
import os

import pytest

import config
from app import app as flask_app, limiter
from audit import write_log_entry

SUBMIT_URL = "/submit"
LOG_URL = "/log"

VALID_BODY = {"text": "This is some sample text for testing the audit log.", "creator_id": "user-99"}


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    limiter.enabled = False
    with flask_app.test_client() as c:
        yield c
    limiter.enabled = True


# ---------------------------------------------------------------------------
# GET /log — basic shape
# ---------------------------------------------------------------------------

def test_get_log_returns_200(client):
    res = client.get(LOG_URL)
    assert res.status_code == 200


def test_get_log_returns_json(client):
    res = client.get(LOG_URL)
    assert res.content_type.startswith("application/json")


def test_get_log_response_has_entries_and_total_keys(client):
    res = client.get(LOG_URL)
    data = res.get_json()
    assert "entries" in data
    assert "total" in data


def test_get_log_entries_is_a_list(client):
    res = client.get(LOG_URL)
    assert isinstance(res.get_json()["entries"], list)


def test_get_log_empty_when_no_submissions(client):
    res = client.get(LOG_URL)
    data = res.get_json()
    assert data["entries"] == []
    assert data["total"] == 0


# ---------------------------------------------------------------------------
# Audit trail written by POST /submit
# ---------------------------------------------------------------------------

def test_submit_creates_audit_log_file(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    assert os.path.exists(config.AUDIT_LOG_FILE)


def test_submit_writes_one_entry_per_call(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    client.post(SUBMIT_URL, json=VALID_BODY)
    client.post(SUBMIT_URL, json=VALID_BODY)
    data = client.get(LOG_URL).get_json()
    assert data["total"] == 3


def test_audit_entry_has_all_required_fields(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    entry = client.get(LOG_URL).get_json()["entries"][0]
    for field in ("event_type", "content_id", "creator_id", "timestamp",
                  "attribution", "confidence", "heuristic_score", "llm_score", "status"):
        assert field in entry, f"Audit entry missing field: {field}"


def test_audit_entry_event_type_is_classification(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    entry = client.get(LOG_URL).get_json()["entries"][0]
    assert entry["event_type"] == "classification"


def test_audit_entry_content_id_matches_submit_response(client):
    res = client.post(SUBMIT_URL, json=VALID_BODY)
    content_id_from_response = res.get_json()["content_id"]
    entry = client.get(LOG_URL).get_json()["entries"][0]
    assert entry["content_id"] == content_id_from_response


def test_audit_entry_creator_id_matches_request(client):
    client.post(SUBMIT_URL, json={"text": "Some text here for testing.", "creator_id": "alice"})
    entry = client.get(LOG_URL).get_json()["entries"][0]
    assert entry["creator_id"] == "alice"


def test_audit_entry_status_is_classified(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    entry = client.get(LOG_URL).get_json()["entries"][0]
    assert entry["status"] == "classified"


def test_audit_entry_attribution_is_valid_label_variant(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    entry = client.get(LOG_URL).get_json()["entries"][0]
    assert entry["attribution"] in {"high_confidence_ai", "high_confidence_human", "uncertain"}


def test_audit_entry_confidence_is_float_in_range(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    entry = client.get(LOG_URL).get_json()["entries"][0]
    assert isinstance(entry["confidence"], float)
    assert 0.0 <= entry["confidence"] <= 1.0


def test_audit_entry_heuristic_score_is_float_in_range(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    entry = client.get(LOG_URL).get_json()["entries"][0]
    assert isinstance(entry["heuristic_score"], float)
    assert 0.0 <= entry["heuristic_score"] <= 1.0


def test_audit_entry_llm_score_is_null_in_single_signal_mode(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    entry = client.get(LOG_URL).get_json()["entries"][0]
    assert entry["llm_score"] is None


def test_audit_entry_timestamp_format(client):
    import re
    client.post(SUBMIT_URL, json=VALID_BODY)
    entry = client.get(LOG_URL).get_json()["entries"][0]
    # Expect ISO8601 format: 2025-04-01T14:32:10.123Z
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
    assert re.match(pattern, entry["timestamp"]), f"Bad timestamp format: {entry['timestamp']}"


def test_audit_entries_for_different_creators_are_independent(client):
    client.post(SUBMIT_URL, json={"text": "Some text for alice.", "creator_id": "alice"})
    client.post(SUBMIT_URL, json={"text": "Some text for bob.", "creator_id": "bob"})
    entries = client.get(LOG_URL).get_json()["entries"]
    creator_ids = {e["creator_id"] for e in entries}
    assert creator_ids == {"alice", "bob"}


# ---------------------------------------------------------------------------
# GET /log query parameters
# ---------------------------------------------------------------------------

def test_get_log_limit_param_restricts_results(client):
    for _ in range(5):
        client.post(SUBMIT_URL, json=VALID_BODY)
    data = client.get(f"{LOG_URL}?limit=2").get_json()
    assert len(data["entries"]) == 2
    assert data["total"] == 2


def test_get_log_limit_returns_most_recent_entries(client):
    bodies = [
        {"text": "First submission text here.", "creator_id": "u-1"},
        {"text": "Second submission text here.", "creator_id": "u-2"},
        {"text": "Third submission text here.", "creator_id": "u-3"},
    ]
    for b in bodies:
        client.post(SUBMIT_URL, json=b)
    entries = client.get(f"{LOG_URL}?limit=2").get_json()["entries"]
    creator_ids = [e["creator_id"] for e in entries]
    assert creator_ids == ["u-2", "u-3"]


def test_get_log_event_type_filter_returns_only_matching(client):
    # Write one real submit entry, then manually write an appeal entry
    client.post(SUBMIT_URL, json=VALID_BODY)
    write_log_entry({"event_type": "appeal", "appeal_id": "ap-1", "content_id": "x"})

    classification_entries = client.get(f"{LOG_URL}?event_type=classification").get_json()["entries"]
    assert len(classification_entries) == 1
    assert all(e["event_type"] == "classification" for e in classification_entries)

    appeal_entries = client.get(f"{LOG_URL}?event_type=appeal").get_json()["entries"]
    assert len(appeal_entries) == 1
    assert appeal_entries[0]["event_type"] == "appeal"


def test_get_log_without_filter_returns_all_event_types(client):
    client.post(SUBMIT_URL, json=VALID_BODY)
    write_log_entry({"event_type": "appeal", "appeal_id": "ap-1"})
    data = client.get(LOG_URL).get_json()
    assert data["total"] == 2


def test_get_log_total_matches_entries_length(client):
    for _ in range(3):
        client.post(SUBMIT_URL, json=VALID_BODY)
    data = client.get(LOG_URL).get_json()
    assert data["total"] == len(data["entries"])


# ---------------------------------------------------------------------------
# Wrong HTTP methods
# ---------------------------------------------------------------------------

def test_post_log_returns_405(client):
    res = client.post(LOG_URL, json={})
    assert res.status_code == 405
