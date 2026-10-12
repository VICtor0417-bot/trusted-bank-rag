"""No model loads or network requests: initialization errors must be actionable."""
import pytest
from fastapi import HTTPException
from bank_rag import api


@pytest.mark.parametrize("endpoint", [api.ask, api.retrieve])
@pytest.mark.parametrize("failure", [OSError, ImportError, ValueError])
def test_initialization_failure_returns_503_without_sensitive_detail(monkeypatch, endpoint, failure):
    def fail():
        raise failure("private evidence or credential must not appear in response")
    monkeypatch.setattr(api, "get_rag", fail)
    with pytest.raises(HTTPException) as caught:
        endpoint(api.Question(question="synthetic archive query"))
    assert caught.value.status_code == 503
    assert caught.value.detail["error"] == "rag_initialization_failed"
    assert caught.value.detail["exception_type"] == failure.__name__
    assert "credential" not in str(caught.value.detail)
