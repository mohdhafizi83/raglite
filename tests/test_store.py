"""Vector store roundtrip + cosine search tests, and embedder determinism."""

import numpy as np
import pytest

from raglite.embedders import HashingEmbedder
from raglite.store import VectorStore


def test_vector_store_roundtrip():
    store = VectorStore(":memory:")
    vec = np.arange(8, dtype=np.float32)
    cid = store.add("doc1", "hello world", 0, 11, '{"k": "v"}', vec)
    stored = store.get(cid)
    assert stored.doc_id == "doc1"
    assert stored.text == "hello world"
    assert (stored.start, stored.end) == (0, 11)
    assert store.count() == 1


def test_vector_store_roundtrip_preserves_vector_bits():
    store = VectorStore(":memory:")
    rng = np.random.default_rng(7)
    vec = rng.standard_normal(32).astype(np.float32)
    cid = store.add("d", "t", 0, 1, "{}", vec)
    rows = store.search_vectors(vec, top_k=1)
    assert rows[0][0] == cid
    assert rows[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vector_search_orders_by_cosine():
    store = VectorStore(":memory:")
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.7, 0.7], dtype=np.float32)
    c = np.array([0.0, 1.0], dtype=np.float32)
    ca = store.add("a", "a", 0, 1, "{}", a)
    cb = store.add("b", "b", 0, 1, "{}", b)
    cc = store.add("c", "c", 0, 1, "{}", c)
    hits = store.search_vectors(a, top_k=3)
    assert [cid for cid, _ in hits] == [ca, cb, cc]


def test_delete_doc_removes_chunks():
    store = VectorStore(":memory:")
    store.add("keep", "x", 0, 1, "{}", np.ones(4, dtype=np.float32))
    store.add("drop", "y", 0, 1, "{}", np.ones(4, dtype=np.float32))
    store.add("drop", "z", 1, 2, "{}", np.ones(4, dtype=np.float32))
    assert store.delete_doc("drop") == 2
    assert store.count() == 1


def test_hashing_embedder_is_deterministic():
    emb = HashingEmbedder(dim=128)
    v1 = emb.embed(["the nimbus warranty is two years"])
    v2 = emb.embed(["the nimbus warranty is two years"])
    assert np.array_equal(v1, v2)


def test_hashing_embedder_rows_are_unit_norm():
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["alpha beta", "gamma delta epsilon"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0)


def test_hashing_embedder_similar_texts_score_higher():
    emb = HashingEmbedder(dim=256).fit(["battery charging", "warranty policy", "wifi setup"])
    q = emb.embed(["battery charging time"])[0]
    sims = emb.embed(["the battery charges in ninety minutes", "the warranty is two years"]) @ q
    assert sims[0] > sims[1]
