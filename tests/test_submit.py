import json
import uuid

import pytest

from app import app as flask_app, limiter

VALID_BODY = {"content": "This is some sample text for testing.", "creator_id": "user-123"}
SUBMIT_URL = "/submit"


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    # Flask-Limiter 4.x stores enabled as an instance attribute; toggle it here
    # so the in-memory counter never blocks test requests.
    limiter.enabled = False
    with flask_app.test_client() as c:
        yield c
    limiter.enabled = True


def post_json(client, body):
    return client.post(SUBMIT_URL, json=body)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_submit_returns_200(client):
    res = post_json(client, VALID_BODY)
    assert res.status_code == 200


def test_response_contains_all_required_fields(client):
    res = post_json(client, VALID_BODY)
    data = res.get_json()
    expected_fields = {"label_id", "weighted_score", "final_confidence_score", "label", "llm_score", "heuristic_score"}
    assert expected_fields == set(data.keys())


def test_label_id_is_valid_uuid(client):
    res = post_json(client, VALID_BODY)
    label_id = res.get_json()["label_id"]
    # uuid.UUID raises ValueError on invalid input
    parsed = uuid.UUID(label_id)
    assert str(parsed) == label_id


def test_score_fields_are_floats(client):
    data = post_json(client, VALID_BODY).get_json()
    for field in ("weighted_score", "final_confidence_score", "llm_score", "heuristic_score"):
        assert isinstance(data[field], float), f"Expected float for {field}, got {type(data[field])}"


def test_score_fields_are_in_zero_to_one_range(client):
    data = post_json(client, VALID_BODY).get_json()
    for field in ("weighted_score", "final_confidence_score", "llm_score", "heuristic_score"):
        assert 0.0 <= data[field] <= 1.0, f"{field}={data[field]} is outside [0, 1]"


def test_label_is_non_empty_string(client):
    data = post_json(client, VALID_BODY).get_json()
    assert isinstance(data["label"], str)
    assert len(data["label"]) > 0


def test_each_submission_gets_unique_label_id(client):
    id1 = post_json(client, VALID_BODY).get_json()["label_id"]
    id2 = post_json(client, VALID_BODY).get_json()["label_id"]
    assert id1 != id2


def test_content_type_of_response_is_json(client):
    res = post_json(client, VALID_BODY)
    assert res.content_type.startswith("application/json")


# ---------------------------------------------------------------------------
# Missing / empty required fields
# ---------------------------------------------------------------------------

def test_missing_content_returns_400(client):
    res = post_json(client, {"creator_id": "user-123"})
    assert res.status_code == 400


def test_missing_creator_id_returns_400(client):
    res = post_json(client, {"content": "Some text."})
    assert res.status_code == 400


def test_empty_content_string_returns_400(client):
    res = post_json(client, {"content": "", "creator_id": "user-123"})
    assert res.status_code == 400


def test_whitespace_only_content_returns_400(client):
    res = post_json(client, {"content": "   \t\n", "creator_id": "user-123"})
    assert res.status_code == 400


def test_empty_creator_id_string_returns_400(client):
    res = post_json(client, {"content": "Some text.", "creator_id": ""})
    assert res.status_code == 400


def test_whitespace_only_creator_id_returns_400(client):
    res = post_json(client, {"content": "Some text.", "creator_id": "   "})
    assert res.status_code == 400


def test_null_content_returns_400(client):
    res = post_json(client, {"content": None, "creator_id": "user-123"})
    assert res.status_code == 400


def test_null_creator_id_returns_400(client):
    res = post_json(client, {"content": "Some text.", "creator_id": None})
    assert res.status_code == 400


def test_non_string_content_returns_400(client):
    res = post_json(client, {"content": 42, "creator_id": "user-123"})
    assert res.status_code == 400


def test_non_string_creator_id_returns_400(client):
    res = post_json(client, {"content": "Some text.", "creator_id": 99})
    assert res.status_code == 400


def test_empty_json_object_returns_400(client):
    res = post_json(client, {})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# Malformed / missing body
# ---------------------------------------------------------------------------

def test_no_body_returns_400(client):
    res = client.post(SUBMIT_URL)
    assert res.status_code == 400


def test_non_json_content_type_returns_400(client):
    res = client.post(
        SUBMIT_URL,
        data="content=hello&creator_id=user-1",
        content_type="application/x-www-form-urlencoded",
    )
    assert res.status_code == 400


def test_malformed_json_returns_400(client):
    res = client.post(
        SUBMIT_URL,
        data="{not valid json}",
        content_type="application/json",
    )
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# Error response structure
# ---------------------------------------------------------------------------

def test_400_response_has_error_field(client):
    res = post_json(client, {"creator_id": "user-123"})
    data = res.get_json()
    assert "error" in data
    assert isinstance(data["error"], str)
    assert len(data["error"]) > 0


def test_error_response_does_not_leak_extra_fields(client):
    res = post_json(client, {"creator_id": "user-123"})
    data = res.get_json()
    # On error, only "error" key — no score fields leaked
    assert "label_id" not in data
    assert "weighted_score" not in data


# ---------------------------------------------------------------------------
# Wrong HTTP methods
# ---------------------------------------------------------------------------

def test_get_submit_returns_405(client):
    res = client.get(SUBMIT_URL)
    assert res.status_code == 405


def test_put_submit_returns_405(client):
    res = client.put(SUBMIT_URL, json=VALID_BODY)
    assert res.status_code == 405


def test_delete_submit_returns_405(client):
    res = client.delete(SUBMIT_URL)
    assert res.status_code == 405
