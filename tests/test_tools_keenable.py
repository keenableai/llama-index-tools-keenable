"""Unit tests for the Keenable LlamaIndex tool spec (offline).

The HTTP transport is faked at the ``requests`` boundary, so these exercise
endpoint selection, attribution headers, the SSRF guard, HTTPS enforcement,
error mapping, and Document construction without a network.
"""

import pytest

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.keenable import KeenableToolSpec
from llama_index.tools.keenable import _client
from llama_index.tools.keenable._client import (
    KeenableError,
    keenable_post,
    reject_private_fetch_target,
    resolve_api_key,
    resolve_base_url,
)


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text="", raise_on_json=False):
        self.status_code = status_code
        self._json = json_body
        self.text = text
        self._raise_on_json = raise_on_json

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._raise_on_json:
            raise ValueError("no json")
        return self._json


class _Recorder:
    """Records the last request issued through the faked requests module."""

    last = {}

    def post(self, url, json=None, headers=None, timeout=None):
        _Recorder.last = {"method": "POST", "url": url, "json": json, "headers": headers}
        return self._response

    def get(self, url, params=None, headers=None, timeout=None):
        _Recorder.last = {"method": "GET", "url": url, "params": params, "headers": headers}
        return self._response

    def __init__(self, response):
        self._response = response


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    monkeypatch.delenv("KEENABLE_API_URL", raising=False)


def _patch(monkeypatch, response):
    rec = _Recorder(response)
    monkeypatch.setattr(_client.requests, "post", rec.post)
    monkeypatch.setattr(_client.requests, "get", rec.get)


# --------------------------------------------------------------------------- #
# Tool spec wiring
# --------------------------------------------------------------------------- #


def test_is_base_tool_spec():
    assert BaseToolSpec.__name__ in [b.__name__ for b in KeenableToolSpec.__mro__]


def test_spec_functions():
    assert KeenableToolSpec.spec_functions == ["search", "fetch"]


def test_to_tool_list():
    tools = KeenableToolSpec().to_tool_list()
    names = {t.metadata.name for t in tools}
    assert {"search", "fetch"} <= names


# --------------------------------------------------------------------------- #
# resolve_base_url + SSRF + key
# --------------------------------------------------------------------------- #


def test_base_url_default_https():
    assert resolve_base_url() == "https://api.keenable.ai"


def test_base_url_http_public_rejected(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_URL", "http://api.keenable.ai")
    with pytest.raises(KeenableError):
        resolve_base_url()


def test_base_url_http_loopback_ok(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_URL", "http://localhost:8000")
    assert resolve_base_url() == "http://localhost:8000"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/x",
        "http://127.0.0.1/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/x",
        "http://metadata.google.internal/x",
        "https:///nohost",
    ],
)
def test_reject_private_fetch_target(url):
    with pytest.raises(KeenableError):
        reject_private_fetch_target(url)


def test_public_fetch_target_allowed():
    reject_private_fetch_target("https://example.com/article")


def test_resolve_api_key(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "env-key")
    assert resolve_api_key("  ") == "env-key"
    assert resolve_api_key("explicit") == "explicit"
    monkeypatch.delenv("KEENABLE_API_KEY")
    assert resolve_api_key("") is None


# --------------------------------------------------------------------------- #
# Transport: endpoint selection, attribution, errors
# --------------------------------------------------------------------------- #


def test_keyless_uses_public_path_and_attribution(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": []}))
    keenable_post("/v1/search/public", "/v1/search", {"query": "x"}, None, 30.0)
    sent = _Recorder.last
    assert sent["url"].endswith("/v1/search/public")
    assert sent["headers"]["X-Keenable-Title"] == "LlamaIndex"
    assert "X-API-Key" not in sent["headers"]
    assert sent["headers"]["User-Agent"].startswith("keenable-llamaindex/")


def test_keyed_uses_authenticated_path(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": []}))
    keenable_post("/v1/search/public", "/v1/search", {"query": "x"}, "secret", 30.0)
    sent = _Recorder.last
    assert sent["url"].endswith("/v1/search")
    assert sent["headers"]["X-API-Key"] == "secret"


@pytest.mark.parametrize(
    ("status", "needle"),
    [(401, "authentication"), (402, "credits"), (429, "rate limit"), (500, "500")],
)
def test_error_status_mapping(monkeypatch, status, needle):
    _patch(monkeypatch, _FakeResponse(status_code=status, json_body={"message": "boom"}))
    with pytest.raises(KeenableError) as exc:
        keenable_post("/v1/search/public", "/v1/search", {"query": "x"}, None, 30.0)
    assert needle in str(exc.value).lower()


def test_non_json_raises(monkeypatch):
    _patch(monkeypatch, _FakeResponse(text="<html>", raise_on_json=True))
    with pytest.raises(KeenableError):
        keenable_post("/v1/search/public", "/v1/search", {"query": "x"}, None, 30.0)


# --------------------------------------------------------------------------- #
# search / fetch -> Document
# --------------------------------------------------------------------------- #


def test_search_returns_documents(monkeypatch):
    body = {
        "results": [
            {"title": "T1", "url": "https://e1.com", "description": "d1"},
            {"title": "T2", "url": "https://e2.com", "description": "d2"},
        ]
    }
    _patch(monkeypatch, _FakeResponse(json_body=body))
    docs = KeenableToolSpec().search("typescript", site="github.com", mode="pro")
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].text == "d1"
    assert docs[0].metadata["url"] == "https://e1.com"
    sent = _Recorder.last["json"]
    assert sent["query"] == "typescript"
    assert sent["site"] == "github.com"
    assert sent["mode"] == "pro"


def test_search_text_falls_back_to_title(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"results": [{"title": "OnlyTitle", "url": "https://e.com"}]}))
    docs = KeenableToolSpec().search("q")
    assert docs[0].text == "OnlyTitle"


def test_search_bad_payload_raises(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"unexpected": True}))
    with pytest.raises(KeenableError):
        KeenableToolSpec().search("q")


def test_fetch_returns_document(monkeypatch):
    _patch(monkeypatch, _FakeResponse(json_body={"url": "https://e.com", "title": "T", "content": "body"}))
    docs = KeenableToolSpec().fetch("https://e.com")
    assert len(docs) == 1
    assert docs[0].text == "body"
    assert docs[0].metadata["title"] == "T"
    assert _Recorder.last["url"].endswith("/v1/fetch/public")


def test_fetch_keyed_path(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "secret")
    _patch(monkeypatch, _FakeResponse(json_body={"content": "x"}))
    KeenableToolSpec().fetch("https://e.com")
    assert _Recorder.last["url"].endswith("/v1/fetch")
    assert _Recorder.last["headers"]["X-API-Key"] == "secret"


@pytest.mark.parametrize("bad_url", ["ftp://e.com/x", "http://127.0.0.1/x", "not-a-url"])
def test_fetch_rejects_bad_urls(bad_url):
    with pytest.raises(KeenableError):
        KeenableToolSpec().fetch(bad_url)
