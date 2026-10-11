"""Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion.

RRF score for a chunk is ``sum over rankers of 1 / (k + rank)`` with ``k=60``
(Ferragina & Ponton's constant). Rank-based fusion is deliberate: BM25 scores
and cosine similarities live on incomparable scales, and any score-weighted
blend needs per-corpus tuning. RRF needs only the orderings, is robust out of
the box, and has one interpretable constant.
"""

from __future__ import annotations

from dataclasses import dataclass

RRF_K = 60


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk id with its fused score and per-ranker ranks."""

    chunk_id: int
    score: float
    bm25_rank: int | None
    vector_rank: int | None


def rrf_fuse(
    rankings: list[list[int]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked id lists into one ranking via RRF.

    ``rankings`` is a list of lists of ids (best first). Returns
    ``(id, rrf_score)`` sorted by descending score, ties broken by id.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def hybrid_search(
    bm25_hits: list[tuple[int, float]],
    vector_hits: list[tuple[int, float]],
    top_k: int = 5,
    k: int = RRF_K,
) -> list[ScoredChunk]:
    """Run RRF over the two rankers and attach per-ranker rank metadata."""
    bm25_ids = [cid for cid, _ in bm25_hits]
    vector_ids = [cid for cid, _ in vector_hits]
    fused = rrf_fuse([bm25_ids, vector_ids], k=k)
    bm25_pos = {cid: i for i, cid in enumerate(bm25_ids, start=1)}
    vector_pos = {cid: i for i, cid in enumerate(vector_ids, start=1)}
    return [
        ScoredChunk(
            chunk_id=cid,
            score=score,
            bm25_rank=bm25_pos.get(cid),
            vector_rank=vector_pos.get(cid),
        )
        for cid, score in fused[:top_k]
    ]
