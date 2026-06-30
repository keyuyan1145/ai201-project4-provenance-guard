"""
Tests for audit.py — write_log_entry and get_log in isolation.
conftest.py redirects AUDIT_LOG_FILE to a per-test temp file automatically.
"""
import json
import os

import config
from audit import get_log, write_log_entry

SAMPLE_ENTRY = {
    "event_type": "classification",
    "content_id": "abc-123",
    "creator_id": "user-1",
    "timestamp": "2025-04-01T14:32:10.123Z",
    "attribution": "uncertain",
    "confidence": 0.31,
    "heuristic_score": 0.42,
    "llm_score": None,
    "status": "classified",
}


# ---------------------------------------------------------------------------
# write_log_entry
# ---------------------------------------------------------------------------

def test_write_creates_file_when_it_does_not_exist():
    assert not os.path.exists(config.AUDIT_LOG_FILE)
    write_log_entry(SAMPLE_ENTRY)
    assert os.path.exists(config.AUDIT_LOG_FILE)


def test_write_produces_valid_json_line():
    write_log_entry(SAMPLE_ENTRY)
    with open(config.AUDIT_LOG_FILE, encoding="utf-8") as f:
        line = f.readline().strip()
    assert json.loads(line) == SAMPLE_ENTRY


def test_write_appends_multiple_entries_as_separate_lines():
    entry_a = {**SAMPLE_ENTRY, "content_id": "aaa"}
    entry_b = {**SAMPLE_ENTRY, "content_id": "bbb"}
    write_log_entry(entry_a)
    write_log_entry(entry_b)
    with open(config.AUDIT_LOG_FILE, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["content_id"] == "aaa"
    assert json.loads(lines[1])["content_id"] == "bbb"


def test_write_preserves_null_llm_score():
    write_log_entry(SAMPLE_ENTRY)
    with open(config.AUDIT_LOG_FILE, encoding="utf-8") as f:
        saved = json.loads(f.readline())
    assert saved["llm_score"] is None


def test_write_does_not_raise_on_io_error(monkeypatch):
    # Point to an unwritable path — should print [ERROR] but not raise
    monkeypatch.setattr(config, "AUDIT_LOG_FILE", "/no/such/dir/audit.jsonl")
    write_log_entry(SAMPLE_ENTRY)  # must not raise


# ---------------------------------------------------------------------------
# get_log
# ---------------------------------------------------------------------------

def test_get_log_returns_empty_list_when_file_missing():
    assert get_log() == []


def test_get_log_returns_all_entries_written():
    for i in range(3):
        write_log_entry({**SAMPLE_ENTRY, "content_id": f"id-{i}"})
    entries = get_log()
    assert len(entries) == 3


def test_get_log_respects_limit():
    for i in range(5):
        write_log_entry({**SAMPLE_ENTRY, "content_id": f"id-{i}"})
    entries = get_log(limit=3)
    assert len(entries) == 3


def test_get_log_limit_returns_most_recent_entries():
    for i in range(5):
        write_log_entry({**SAMPLE_ENTRY, "content_id": f"id-{i}"})
    entries = get_log(limit=2)
    # Last 2 written should be id-3 and id-4
    assert entries[0]["content_id"] == "id-3"
    assert entries[1]["content_id"] == "id-4"


def test_get_log_returns_entries_in_chronological_order():
    for i in range(3):
        write_log_entry({**SAMPLE_ENTRY, "content_id": f"id-{i}"})
    entries = get_log()
    ids = [e["content_id"] for e in entries]
    assert ids == ["id-0", "id-1", "id-2"]


def test_get_log_filters_by_event_type():
    write_log_entry({**SAMPLE_ENTRY, "event_type": "classification", "content_id": "c-1"})
    write_log_entry({**SAMPLE_ENTRY, "event_type": "appeal", "content_id": "a-1"})
    write_log_entry({**SAMPLE_ENTRY, "event_type": "classification", "content_id": "c-2"})

    classification_entries = get_log(event_type="classification")
    assert len(classification_entries) == 2
    assert all(e["event_type"] == "classification" for e in classification_entries)

    appeal_entries = get_log(event_type="appeal")
    assert len(appeal_entries) == 1
    assert appeal_entries[0]["content_id"] == "a-1"


def test_get_log_no_filter_returns_all_event_types():
    write_log_entry({**SAMPLE_ENTRY, "event_type": "classification"})
    write_log_entry({**SAMPLE_ENTRY, "event_type": "appeal"})
    assert len(get_log()) == 2


def test_get_log_skips_malformed_lines_without_crashing():
    with open(config.AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write('{"valid": true}\n')
        f.write("{not valid json}\n")
        f.write('{"also": "valid"}\n')
    entries = get_log()
    assert len(entries) == 2  # malformed line skipped
    assert entries[0] == {"valid": True}
    assert entries[1] == {"also": "valid"}


def test_get_log_skips_blank_lines():
    with open(config.AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write('{"content_id": "x"}\n')
        f.write("\n")
        f.write('{"content_id": "y"}\n')
    entries = get_log()
    assert len(entries) == 2


def test_get_log_returns_empty_list_for_unknown_event_type():
    write_log_entry(SAMPLE_ENTRY)
    assert get_log(event_type="nonexistent") == []
