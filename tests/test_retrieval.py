"""RRF fusion order + hybrid retrieval tests."""

from raglite.retrieval import hybrid_search, rrf_fuse


def test_rrf_agrees_with_both_rankers():
    fused = rrf_fuse([[1, 2, 3], [1, 3, 2]])
    assert fused[0][0] == 1  # rank 1 in both lists wins


def test_rrf_rescues_item_ranked_mid_by_both():
    fused = rrf_fuse([[10, 11, 12], [10, 11, 12]])
    ids = [cid for cid, _ in fused]
    assert ids == [10, 11, 12]


def test_rrf_score_formula():
    fused = dict(rrf_fuse([[7], [7]], k=60))
    assert fused[7] == 2 / 61


def test_rrf_single_ranker_passthrough():
    fused = rrf_fuse([[5, 6, 7]])
    assert [cid for cid, _ in fused] == [5, 6, 7]


def test_rrf_empty_inputs():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_hybrid_search_fuses_both_lists():
    bm25 = [(1, 9.0), (2, 5.0)]
    vec = [(2, 0.9), (3, 0.8)]
    hits = hybrid_search(bm25, vec, top_k=3)
    ids = [h.chunk_id for h in hits]
    assert ids[0] == 2  # appears in both rankers -> best fused score
    assert set(ids) == {1, 2, 3}


def test_hybrid_search_attaches_ranks():
    bm25 = [(1, 9.0)]
    vec = [(2, 0.9)]
    hits = hybrid_search(bm25, vec, top_k=2)
    by_id = {h.chunk_id: h for h in hits}
    assert by_id[1].bm25_rank == 1 and by_id[1].vector_rank is None
    assert by_id[2].bm25_rank is None and by_id[2].vector_rank == 1
