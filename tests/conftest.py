import pytest
import config


@pytest.fixture(autouse=True)
def temp_audit_log(tmp_path, monkeypatch):
    """Redirect all audit log writes/reads to a per-test temp file.

    Applied to every test automatically so no test touches data/audit_log.jsonl.
    """
    monkeypatch.setattr(config, "AUDIT_LOG_FILE", str(tmp_path / "test_audit.jsonl"))
