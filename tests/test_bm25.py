"""BM25 ranking sanity tests."""

from raglite.store import BM25Index


def test_bm25_prefers_doc_with_query_term():
    idx = BM25Index()
    idx.add(1, "the nimbus warranty covers manufacturing defects")
    idx.add(2, "the battery capacity is 4800 mah")
    hits = idx.search("warranty defects", top_k=2)
    assert hits[0][0] == 1


def test_bm25_idf_prefers_rare_term():
    idx = BM25Index()
    idx.add(1, "python python python snake")
    idx.add(2, "java python enterprise")
    idx.add(3, "go golang systems")
    hits = idx.search("python snake", top_k=3)
    assert hits[0][0] == 1  # rare term "snake" dominates


def test_bm25_no_match_returns_empty():
    idx = BM25Index()
    idx.add(1, "alpha beta")
    assert idx.search("gamma", top_k=5) == []


def test_bm25_scores_descend():
    idx = BM25Index()
    idx.add(1, "kafka kafka kafka")
    idx.add(2, "kafka rabbit")
    idx.add(3, "rabbit deer")
    hits = idx.search("kafka", top_k=2)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_remove_updates_index():
    idx = BM25Index()
    idx.add(1, "alpha beta")
    idx.add(2, "beta gamma")
    idx.remove(1)
    assert len(idx) == 1
    ids = [cid for cid, _ in idx.search("alpha", top_k=5)]
    assert 1 not in ids


def test_bm25_dump_load_roundtrip():
    idx = BM25Index()
    idx.add(1, "alpha beta gamma")
    idx.add(2, "gamma delta")
    clone = BM25Index()
    clone.load(idx.dump())
    assert clone.search("alpha", top_k=1) == idx.search("alpha", top_k=1)
