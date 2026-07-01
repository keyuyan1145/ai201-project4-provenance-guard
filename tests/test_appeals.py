"""
Tests for POST /appeals endpoint.
conftest.py redirects AUDIT_LOG_FILE to a per-test temp file automatically.
conftest.py mocks compute_llm_score to None so no real Groq calls are made.
"""
import pytest

from app import app as flask_app, limiter

SUBMIT_URL = "/submit"
APPEALS_URL = "/appeal"
LOG_URL = "/log"

VALID_SUBMIT_BODY = {
    "text": "This is some sample text for testing the appeals flow.",
    "creator_id": "user-appeals-test",
}
REASONING = "I wrote this content myself for a class assignment."


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    limiter.enabled = False
    with flask_app.test_client() as c:
        yield c
    limiter.enabled = True


def submit_and_get_content_id(client) -> str:
    res = client.post(SUBMIT_URL, json=VALID_SUBMIT_BODY)
    return res.get_json()["content_id"]


def post_appeal(client, body):
    return client.post(APPEALS_URL, json=body)


# ---------------------------------------------------------------------------
# Happy path — response shape
# ---------------------------------------------------------------------------

def test_valid_appeal_returns_200(client):
    content_id = submit_and_get_content_id(client)
    res = post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})
    assert res.status_code == 200


def test_appeal_response_is_json(client):
    content_id = submit_and_get_content_id(client)
    res = post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})
    assert res.content_type.startswith("application/json")


def test_appeal_response_has_required_fields(client):
    content_id = submit_and_get_content_id(client)
    data = post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING}).get_json()
    for field in ("appeal_id", "content_id", "status", "message", "timestamp"):
        assert field in data, f"Missing field: {field}"


def test_appeal_response_status_is_under_review(client):
    content_id = submit_and_get_content_id(client)
    data = post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING}).get_json()
    assert data["status"] == "under_review"


def test_appeal_response_content_id_matches_request(client):
    content_id = submit_and_get_content_id(client)
    data = post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING}).get_json()
    assert data["content_id"] == content_id


def test_appeal_response_message_is_non_empty_string(client):
    content_id = submit_and_get_content_id(client)
    data = post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING}).get_json()
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 0


def test_appeal_id_is_unique_per_appeal(client):
    cid1 = submit_and_get_content_id(client)
    cid2 = submit_and_get_content_id(client)
    id1 = post_appeal(client, {"content_id": cid1, "creator_reasoning": REASONING}).get_json()["appeal_id"]
    id2 = post_appeal(client, {"content_id": cid2, "creator_reasoning": REASONING}).get_json()["appeal_id"]
    assert id1 != id2


# ---------------------------------------------------------------------------
# Audit log — classification entry updated
# ---------------------------------------------------------------------------

def test_classification_entry_status_updated_to_under_review(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    entries = client.get(f"{LOG_URL}?event_type=classification").get_json()["entries"]
    classification = next(e for e in entries if e["content_id"] == content_id)
    assert classification["status"] == "under_review"


def test_classification_entry_appeal_reasoning_populated(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    entries = client.get(f"{LOG_URL}?event_type=classification").get_json()["entries"]
    classification = next(e for e in entries if e["content_id"] == content_id)
    assert classification.get("appeal_reasoning") == REASONING


def test_classification_entry_status_was_classified_before_appeal(client):
    content_id = submit_and_get_content_id(client)

    entries = client.get(f"{LOG_URL}?event_type=classification").get_json()["entries"]
    classification = next(e for e in entries if e["content_id"] == content_id)
    assert classification["status"] == "classified"


# ---------------------------------------------------------------------------
# Audit log — appeal entry appended
# ---------------------------------------------------------------------------

def test_appeal_event_is_written_to_audit_log(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    appeal_entries = client.get(f"{LOG_URL}?event_type=appeal").get_json()["entries"]
    assert len(appeal_entries) == 1


def test_appeal_audit_entry_has_required_fields(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    entry = client.get(f"{LOG_URL}?event_type=appeal").get_json()["entries"][0]
    for field in ("event_type", "appeal_id", "content_id", "creator_reasoning", "timestamp", "status"):
        assert field in entry, f"Appeal audit entry missing field: {field}"


def test_appeal_audit_entry_event_type_is_appeal(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    entry = client.get(f"{LOG_URL}?event_type=appeal").get_json()["entries"][0]
    assert entry["event_type"] == "appeal"


def test_appeal_audit_entry_content_id_matches(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    entry = client.get(f"{LOG_URL}?event_type=appeal").get_json()["entries"][0]
    assert entry["content_id"] == content_id


def test_appeal_audit_entry_reasoning_matches(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    entry = client.get(f"{LOG_URL}?event_type=appeal").get_json()["entries"][0]
    assert entry["creator_reasoning"] == REASONING


def test_appeal_audit_entry_status_is_under_review(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    entry = client.get(f"{LOG_URL}?event_type=appeal").get_json()["entries"][0]
    assert entry["status"] == "under_review"


def test_total_audit_entries_after_appeal_is_two(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})

    data = client.get(LOG_URL).get_json()
    assert data["total"] == 2


# ---------------------------------------------------------------------------
# Validation — missing / invalid fields
# ---------------------------------------------------------------------------

def test_missing_content_id_returns_400(client):
    res = post_appeal(client, {"creator_reasoning": REASONING})
    assert res.status_code == 400


def test_missing_creator_reasoning_returns_400(client):
    res = post_appeal(client, {"content_id": "some-uuid"})
    assert res.status_code == 400


def test_empty_content_id_returns_400(client):
    res = post_appeal(client, {"content_id": "", "creator_reasoning": REASONING})
    assert res.status_code == 400


def test_empty_creator_reasoning_returns_400(client):
    content_id = submit_and_get_content_id(client)
    res = post_appeal(client, {"content_id": content_id, "creator_reasoning": ""})
    assert res.status_code == 400


def test_whitespace_only_reasoning_returns_400(client):
    content_id = submit_and_get_content_id(client)
    res = post_appeal(client, {"content_id": content_id, "creator_reasoning": "   "})
    assert res.status_code == 400


def test_null_content_id_returns_400(client):
    res = post_appeal(client, {"content_id": None, "creator_reasoning": REASONING})
    assert res.status_code == 400


def test_non_string_content_id_returns_400(client):
    res = post_appeal(client, {"content_id": 42, "creator_reasoning": REASONING})
    assert res.status_code == 400


def test_empty_json_body_returns_400(client):
    res = post_appeal(client, {})
    assert res.status_code == 400


def test_no_body_returns_400(client):
    res = client.post(APPEALS_URL)
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# 404 — unknown content_id
# ---------------------------------------------------------------------------

def test_unknown_content_id_returns_404(client):
    res = post_appeal(client, {"content_id": "00000000-0000-0000-0000-000000000000", "creator_reasoning": REASONING})
    assert res.status_code == 404


def test_404_response_has_error_field(client):
    res = post_appeal(client, {"content_id": "no-such-id", "creator_reasoning": REASONING})
    data = res.get_json()
    assert "error" in data


# ---------------------------------------------------------------------------
# 409 — duplicate appeal
# ---------------------------------------------------------------------------

def test_second_appeal_for_same_content_returns_409(client):
    content_id = submit_and_get_content_id(client)
    post_appeal(client, {"content_id": content_id, "creator_reasoning": REASONING})
    res = post_appeal(client, {"content_id": content_id, "creator_reasoning": "Another reason."})
    assert res.status_code == 409


# ---------------------------------------------------------------------------
# Wrong HTTP methods
# ---------------------------------------------------------------------------

def test_get_appeals_returns_405(client):
    res = client.get(APPEALS_URL)
    assert res.status_code == 405


def test_put_appeals_returns_405(client):
    res = client.put(APPEALS_URL, json={})
    assert res.status_code == 405
