"""Unit tests for Cortiqa client initialization and error mapping."""

import pytest
from cortiqa import Cortiqa, AsyncCortiqa
from cortiqa.exceptions import AuthenticationError


def test_client_missing_api_key(monkeypatch):
    monkeypatch.delenv("CORTIQA_API_KEY", raising=False)
    with pytest.raises(AuthenticationError):
        Cortiqa()


def test_client_custom_options():
    client = Cortiqa(
        api_key="sk-cortiqa-test-12345",
        base_url="https://custom.api.cortiqa.co",
        timeout=30.0,
    )
    assert client.api_key == "sk-cortiqa-test-12345"
    assert client.base_url == "https://custom.api.cortiqa.co"
    assert client.timeout == 30.0
    assert client._headers["Authorization"] == "Bearer sk-cortiqa-test-12345"
    assert "cortiqa-python" in client._headers["User-Agent"]
    client.close()


def test_async_client_initialization():
    client = AsyncCortiqa(
        api_key="sk-cortiqa-test-async",
        base_url="https://api.cortiqa.co",
    )
    assert client.api_key == "sk-cortiqa-test-async"
    assert client.base_url == "https://api.cortiqa.co"
