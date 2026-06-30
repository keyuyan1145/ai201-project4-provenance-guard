import pytest
import config


@pytest.fixture(autouse=True)
def temp_audit_log(tmp_path, monkeypatch):
    """Redirect all audit log writes/reads to a per-test temp file.

    Applied to every test automatically so no test touches data/audit_log.jsonl.
    """
    monkeypatch.setattr(config, "AUDIT_LOG_FILE", str(tmp_path / "test_audit.jsonl"))


@pytest.fixture(autouse=True)
def mock_llm_score(monkeypatch):
    """Prevent real Groq API calls in all endpoint tests.

    Patches the reference in app.py so POST /submit runs in single-signal mode.
    Tests in test_llm_signal.py call pipeline.llm_signal.compute_llm_score directly
    and mock Groq at the class level, so they are unaffected by this fixture.
    """
    monkeypatch.setattr("app.compute_llm_score", lambda text: None)
