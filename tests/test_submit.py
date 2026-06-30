import uuid

import pytest

from app import app as flask_app, limiter

VALID_BODY = {"text": "This is some sample text for testing.", "creator_id": "user-123"}
SUBMIT_URL = "/submit"

EXPECTED_FIELDS = {
    "label_id", "content_id", "weighted_score", "final_confidence_score",
    "attribution", "label", "llm_score", "heuristic_score",
}

# Texts long enough to avoid the short-text cap (> MIN_TEXT_LENGTH=80 words)
_AI_TEXT = (
    "Delving into the comprehensive realm of robust and seamless solutions, it is worth noting "
    "that leveraging these crucial and invaluable frameworks is pivotal. "
    "Moreover, the nuanced approach provides notably straightforward pathways to success. "
    "Furthermore, it is important to recognize that modern enterprises require sophisticated solutions. "
    "Additionally, these comprehensive methodologies ensure seamless integration throughout. "
    "In conclusion, the pivotal role of robust systems cannot be overstated. "
    "It is certainly worth noting that nuanced and comprehensive strategies are crucial. "
    "Leveraging these invaluable insights is pivotal for robust and seamless modern enterprises."
)
_HUMAN_TEXT = (
    "I was out last tuesday with some friends and we got into this long argument about "
    "whether hot dogs are sandwiches and nobody could agree. One guy kept insisting they are their own "
    "category which is honestly kind of fair. Anyway we ended up just dropping it. "
    "Super silly debate but it was a good time overall. I have had this conversation before and "
    "it never goes anywhere productive. Some questions just do not have clear answers I guess."
)


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    limiter.enabled = False
    with flask_app.test_client() as c:
        yield c
    limiter.enabled = True


def post_json(client, body):
    return client.post(SUBMIT_URL, json=body)


# ---------------------------------------------------------------------------
# Happy path — response shape
# ---------------------------------------------------------------------------

def test_valid_submit_returns_200(client):
    res = post_json(client, VALID_BODY)
    assert res.status_code == 200


def test_response_contains_exactly_the_required_fields(client):
    data = post_json(client, VALID_BODY).get_json()
    assert set(data.keys()) == EXPECTED_FIELDS


def test_content_type_of_response_is_json(client):
    res = post_json(client, VALID_BODY)
    assert res.content_type.startswith("application/json")


# ---------------------------------------------------------------------------
# Happy path — field values
# ---------------------------------------------------------------------------

def test_label_id_is_valid_uuid(client):
    label_id = post_json(client, VALID_BODY).get_json()["label_id"]
    parsed = uuid.UUID(label_id)  # raises ValueError on bad input
    assert str(parsed) == label_id


def test_content_id_equals_label_id(client):
    data = post_json(client, VALID_BODY).get_json()
    assert data["content_id"] == data["label_id"]


def test_attribution_equals_final_confidence_score(client):
    data = post_json(client, VALID_BODY).get_json()
    assert data["attribution"] == data["final_confidence_score"]


def test_numeric_score_fields_are_floats(client):
    data = post_json(client, VALID_BODY).get_json()
    for field in ("weighted_score", "final_confidence_score", "attribution", "heuristic_score"):
        assert isinstance(data[field], float), f"Expected float for {field}, got {type(data[field])}"


def test_llm_score_is_null_in_single_signal_mode(client):
    data = post_json(client, VALID_BODY).get_json()
    assert data["llm_score"] is None


def test_score_fields_are_in_zero_to_one_range(client):
    data = post_json(client, VALID_BODY).get_json()
    for field in ("weighted_score", "final_confidence_score", "attribution", "heuristic_score"):
        assert 0.0 <= data[field] <= 1.0, f"{field}={data[field]} is outside [0, 1]"


def test_label_is_one_of_the_three_valid_variants(client):
    data = post_json(client, VALID_BODY).get_json()
    assert data["label"] in {"high_confidence_ai", "high_confidence_human", "uncertain"}


def test_each_submission_gets_unique_label_id(client):
    id1 = post_json(client, VALID_BODY).get_json()["label_id"]
    id2 = post_json(client, VALID_BODY).get_json()["label_id"]
    assert id1 != id2


# ---------------------------------------------------------------------------
# Happy path — Signal 1 is actually running (not hardcoded)
# ---------------------------------------------------------------------------

def test_heuristic_score_is_higher_for_ai_text_than_human_text(client):
    ai_score = post_json(client, {"text": _AI_TEXT, "creator_id": "u1"}).get_json()["heuristic_score"]
    human_score = post_json(client, {"text": _HUMAN_TEXT, "creator_id": "u2"}).get_json()["heuristic_score"]
    assert ai_score > human_score


def test_heuristic_score_is_not_hardcoded(client):
    # Two different texts should produce different heuristic scores
    score1 = post_json(client, {"text": _AI_TEXT, "creator_id": "u1"}).get_json()["heuristic_score"]
    score2 = post_json(client, {"text": _HUMAN_TEXT, "creator_id": "u2"}).get_json()["heuristic_score"]
    assert score1 != score2


def test_weighted_score_equals_heuristic_score_without_llm(client):
    # In single-signal mode weighted_score must mirror heuristic_score exactly
    data = post_json(client, {"text": _AI_TEXT, "creator_id": "u1"}).get_json()
    assert data["weighted_score"] == data["heuristic_score"]


def test_final_confidence_score_is_derived_from_heuristic(client):
    # final_confidence_score = raw_confidence * SINGLE_SIGNAL_MULTIPLIER
    # raw_confidence = 2 * |weighted_score - 0.5|
    import config
    data = post_json(client, {"text": _AI_TEXT, "creator_id": "u1"}).get_json()
    ws = data["weighted_score"]
    expected_fc = round(2 * abs(ws - 0.5) * config.SINGLE_SIGNAL_MULTIPLIER, 4)
    assert data["final_confidence_score"] == expected_fc


# ---------------------------------------------------------------------------
# Missing / empty required fields
# ---------------------------------------------------------------------------

def test_missing_content_returns_400(client):
    res = post_json(client, {"creator_id": "user-123"})
    assert res.status_code == 400


def test_missing_creator_id_returns_400(client):
    res = post_json(client, {"text": "Some text."})
    assert res.status_code == 400


def test_empty_content_string_returns_400(client):
    res = post_json(client, {"text": "", "creator_id": "user-123"})
    assert res.status_code == 400


def test_whitespace_only_content_returns_400(client):
    res = post_json(client, {"text": "   \t\n", "creator_id": "user-123"})
    assert res.status_code == 400


def test_empty_creator_id_string_returns_400(client):
    res = post_json(client, {"text": "Some text.", "creator_id": ""})
    assert res.status_code == 400


def test_whitespace_only_creator_id_returns_400(client):
    res = post_json(client, {"text": "Some text.", "creator_id": "   "})
    assert res.status_code == 400


def test_null_content_returns_400(client):
    res = post_json(client, {"text": None, "creator_id": "user-123"})
    assert res.status_code == 400


def test_null_creator_id_returns_400(client):
    res = post_json(client, {"text": "Some text.", "creator_id": None})
    assert res.status_code == 400


def test_non_string_content_returns_400(client):
    res = post_json(client, {"text": 42, "creator_id": "user-123"})
    assert res.status_code == 400


def test_non_string_creator_id_returns_400(client):
    res = post_json(client, {"text": "Some text.", "creator_id": 99})
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


def test_error_response_does_not_leak_score_fields(client):
    res = post_json(client, {"creator_id": "user-123"})
    data = res.get_json()
    assert "label_id" not in data
    assert "weighted_score" not in data
    assert "heuristic_score" not in data


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
