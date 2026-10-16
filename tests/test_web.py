"""Web UI tests: page serves with the query form, CSP relaxed only as designed,
document listing reflects the index."""

import pytest
from fastapi.testclient import TestClient

from raglite.app import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_ui_page_serves_query_form(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert 'id="qform"' in r.text  # query box
    assert 'id="iform"' in r.text  # ingest form
    assert "raglite" in r.text


def test_ui_page_csp_relaxed_only_for_inline_assets(client):
    r = client.get("/")
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "script-src 'unsafe-inline'" in csp
    assert "style-src 'unsafe-inline'" in csp
    assert "connect-src 'self'" in csp
    # other hardening headers still apply to the UI page
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_api_docs_empty_then_seeded(client):
    r = client.get("/api/docs")
    assert r.status_code == 200
    assert isinstance(r.json()["docs"], list)
    assert client.post("/seed").status_code == 200
    docs = {d["doc_id"]: d["chunks"] for d in client.get("/api/docs").json()["docs"]}
    assert "nimbus_warranty.txt" in docs
    assert docs["nimbus_warranty.txt"] >= 1


def test_api_docs_reflects_ingest(client):
    r = client.post("/ingest", json={"doc_id": "ui-test.txt", "text": "Hello from the UI test."})
    assert r.status_code == 200
    docs = {d["doc_id"] for d in client.get("/api/docs").json()["docs"]}
    assert "ui-test.txt" in docs
