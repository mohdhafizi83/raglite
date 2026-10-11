"""Hardening tests: security headers, body caps, payload limits, generic errors."""

import pytest
from fastapi.testclient import TestClient

from raglite import app as app_module
from raglite.app import (
    MAX_BODY_BYTES,
    MAX_DOC_CHARS,
    MAX_META_KEYS,
    MAX_META_VALUE_CHARS,
    MAX_QUESTION_CHARS,
    app,
)

EXPECTED_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@pytest.fixture()
def client():
    with TestClient(app) as c:
        assert c.post("/seed").status_code == 200
        yield c


# -- security headers ----------------------------------------------------------


def test_security_headers_present_on_success(client):
    r = client.get("/health")
    assert r.status_code == 200
    for header, value in EXPECTED_HEADERS.items():
        assert r.headers[header] == value


def test_security_headers_present_on_error_responses(client):
    for r in (
        client.get("/nope"),  # 404
        client.post("/query", content="not json", headers={"Content-Type": "application/json"}),
    ):
        for header, value in EXPECTED_HEADERS.items():
            assert r.headers[header] == value, f"{header} missing on {r.status_code}"


# -- request body cap ----------------------------------------------------------


def test_oversize_body_rejected_413(client):
    r = client.post("/ingest", content=b"x" * (MAX_BODY_BYTES + 1))
    assert r.status_code == 413
    assert r.json() == {"error": "request body too large"}


def test_chunked_oversize_body_rejected_413(client, monkeypatch):
    # No Content-Length header (Transfer-Encoding: chunked): the middleware must
    # count bytes as they stream and still enforce the cap.
    monkeypatch.setattr(app_module, "MAX_BODY_BYTES", 100)
    chunks = iter([b"a" * 64, b"b" * 64, b"c" * 64])
    r = client.post("/ingest", content=chunks)
    assert r.status_code == 413
    assert r.json() == {"error": "request body too large"}


def test_body_under_cap_still_processed(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_BODY_BYTES", 10_000)
    r = client.post(
        "/ingest", json={"doc_id": "small.txt", "text": "Short and under the cap."}
    )
    assert r.status_code == 200
    assert r.json()["doc_id"] == "small.txt"


# -- generic error bodies ------------------------------------------------------


def test_malformed_json_generic_400(client):
    r = client.post("/query", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json() == {"error": "invalid request payload"}


def test_validation_error_does_not_leak_field_detail(client):
    r = client.post("/ingest", json={"text": "missing doc_id"})
    assert r.status_code == 400
    body = r.text
    assert r.json() == {"error": "invalid request payload"}
    assert "doc_id" not in body  # no field-level internals echoed


def test_unhandled_exception_generic_500(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app_module.engine, "query", boom)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/query", json={"question": "anything"})
    assert r.status_code == 500
    assert r.json() == {"error": "internal server error"}
    assert "secret internal detail" not in r.text


# -- payload limits ------------------------------------------------------------


def test_ingest_doc_over_char_cap_rejected(client):
    r = client.post("/ingest", json={"doc_id": "huge.txt", "text": "a" * (MAX_DOC_CHARS + 1)})
    assert r.status_code == 400
    assert r.json() == {"error": "invalid request payload"}


def test_ingest_meta_over_key_cap_rejected(client):
    meta = {f"k{i}": "v" for i in range(MAX_META_KEYS + 1)}
    r = client.post("/ingest", json={"doc_id": "m.txt", "text": "ok", "meta": meta})
    assert r.status_code == 400
    assert r.json() == {"error": "invalid request payload"}


def test_ingest_meta_value_over_cap_rejected(client):
    meta = {"k": "v" * (MAX_META_VALUE_CHARS + 1)}
    r = client.post("/ingest", json={"doc_id": "m.txt", "text": "ok", "meta": meta})
    assert r.status_code == 400
    assert r.json() == {"error": "invalid request payload"}


def test_query_over_char_cap_rejected(client):
    r = client.post("/query", json={"question": "a" * (MAX_QUESTION_CHARS + 1)})
    assert r.status_code == 400
    assert r.json() == {"error": "invalid request payload"}
