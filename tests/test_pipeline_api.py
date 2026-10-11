"""End-to-end pipeline + FastAPI API tests (zero network)."""

import pytest
from fastapi.testclient import TestClient

from raglite.app import app
from raglite.eval import GOLDEN_SET, evaluate
from raglite.pipeline import RAGLite
from raglite.sample_corpus import SAMPLE_DOCS, seed


@pytest.fixture()
def engine():
    eng = RAGLite()
    seed(eng)
    yield eng
    eng.close()


def test_ingest_returns_chunk_count(engine):
    n = engine.ingest("Fresh text about the flux capacitor.", doc_id="fresh.txt")
    assert n >= 1
    assert engine.store.count() > len(SAMPLE_DOCS)


def test_query_returns_answer_with_citations(engine):
    result = engine.query("How long is the warranty on the Nimbus 900?", top_k=3)
    assert "two-year" in result.answer or "two years" in result.answer
    assert len(result.citations) == 3
    top = result.citations[0]
    assert top.doc_id == "nimbus_warranty.txt"
    # Citation span must be an exact slice of the source doc.
    src = SAMPLE_DOCS[top.doc_id]
    assert src[top.start : top.end] in src


def test_query_empty_engine_gives_no_match_message():
    eng = RAGLite()
    result = eng.query("anything")
    assert result.answer == "No matching documents found."
    assert result.citations == []
    eng.close()


def test_eval_hit_rate_above_threshold(engine):
    report = evaluate(engine)
    assert report.hit_rate_at_3 >= 0.99, report.to_dict()
    assert report.mrr >= 0.9


def test_eval_golden_set_covers_all_sample_docs():
    covered = {d for item in GOLDEN_SET for d in item.gold_doc_ids}
    assert covered == set(SAMPLE_DOCS.keys())


# -- FastAPI surface ---------------------------------------------------------


@pytest.fixture()
def client():
    # Fresh app module state per test module run is fine; seed explicitly.
    with TestClient(app) as c:
        r = c.post("/seed")
        assert r.status_code == 200
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["chunks"] > 0


def test_ingest_endpoint(client):
    r = client.post(
        "/ingest",
        json={"doc_id": "extra.txt", "text": "The widget spins at 3000 rpm.", "meta": {}},
    )
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1


def test_query_endpoint_end_to_end(client):
    r = client.post(
        "/query", json={"question": "How do I connect the Nimbus 900 to Wi-Fi?", "top_k": 3}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["citations"]) == 3
    assert body["citations"][0]["doc_id"] == "nimbus_setup.txt"
    assert all("score" in c for c in body["citations"])
    assert all("rrf_score" in s for s in body["scores"])


def test_query_on_empty_index_returns_409():
    from raglite import app as app_module

    fresh = RAGLite()
    old = app_module.engine
    app_module.engine = fresh
    try:
        c = TestClient(app)
        r = c.post("/query", json={"question": "anything"})
        assert r.status_code == 409
    finally:
        app_module.engine = old
        fresh.close()


def test_eval_endpoint(client):
    r = client.get("/eval")
    assert r.status_code == 200
    body = r.json()
    assert body["hit_rate_at_3"] >= 0.99
    assert body["n"] == len(GOLDEN_SET)
