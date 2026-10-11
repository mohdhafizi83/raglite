"""The RAGLite pipeline: ingest -> chunk -> embed -> store; query -> fuse -> answer.

Wires the chunker, embedder, SQLite store, BM25 index, RRF fusion and the LLM
seam together. Everything is synchronous and dependency-light by design; the
FastAPI layer is a thin wrapper over this class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from raglite.chunker import split_text
from raglite.embedders import Embedder, HashingEmbedder
from raglite.llm import EchoLLM, LLMClient
from raglite.retrieval import ScoredChunk, hybrid_search
from raglite.store import BM25Index, VectorStore

_BM25_KEY = "bm25_state"


@dataclass(frozen=True)
class Citation:
    """A verifiable pointer: which doc, which exact character span, why ranked."""

    doc_id: str
    chunk_id: int
    start: int
    end: int
    score: float
    snippet: str

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 6),
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class QueryResult:
    """Answer plus its grounding citations and fused retrieval scores."""

    question: str
    answer: str
    citations: list[Citation]
    scores: list[dict[str, float | int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "scores": self.scores,
        }


class RAGLite:
    """Minimal hybrid RAG engine with an offline-friendly default stack."""

    def __init__(
        self,
        store_path: str = ":memory:",
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        corpus: list[str] | None = None,
    ) -> None:
        self.store = VectorStore(store_path)
        self.embedder: Embedder = embedder or HashingEmbedder(dim=256)
        self.llm: LLMClient = llm or EchoLLM()
        self.bm25 = BM25Index()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Optional corpus snapshot so the embedder can keep IDF stats current.
        self._corpus: list[str] = list(corpus) if corpus is not None else []
        self._restore_bm25()

    # -- ingestion ----------------------------------------------------------

    def ingest(self, text: str, doc_id: str, meta: dict[str, str] | None = None) -> int:
        """Chunk, embed and store one document. Returns the chunk count."""
        chunks = split_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            return 0
        self._corpus.append(text)
        if isinstance(self.embedder, HashingEmbedder):
            self.embedder.fit(self._corpus)
        meta_json = json.dumps(meta or {}, sort_keys=True)
        vectors = self.embedder.embed([c.text for c in chunks])
        for chunk, vec in zip(chunks, vectors, strict=True):
            cid = self.store.add(doc_id, chunk.text, chunk.start, chunk.end, meta_json, vec)
            self.bm25.add(cid, chunk.text)
        self._persist_bm25()
        return len(chunks)

    # -- query --------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 5, candidate_k: int = 20) -> list[ScoredChunk]:
        """Hybrid retrieval: BM25 + vector candidates fused via RRF.

        With the built-in HashingEmbedder, corpus IDF stats change on every
        ingest, so stored vectors can lag behind. Before a query we refit on
        the full corpus and re-embed stored chunks (self-healing refresh):
        cheap at this scale and it keeps query/doc vectors on the same footing.
        """
        if isinstance(self.embedder, HashingEmbedder) and self._corpus:
            self.embedder.fit(self._corpus)
            self._refresh_vectors()
        qvec = self.embedder.embed([question])[0]
        bm25_hits = self.bm25.search(question, top_k=candidate_k)
        vec_hits = self.store.search_vectors(qvec, top_k=candidate_k)
        return hybrid_search(bm25_hits, vec_hits, top_k=top_k)

    def _refresh_vectors(self) -> None:
        pairs = self.store.all_texts()
        if not pairs:
            return
        vectors = self.embedder.embed([t for _, t in pairs])
        for (cid, _), vec in zip(pairs, vectors, strict=True):
            self.store.update_vector(cid, vec)

    def query(self, question: str, top_k: int = 3) -> QueryResult:
        """Answer ``question`` from retrieved chunks, with citations."""
        hits = self.retrieve(question, top_k=top_k)
        citations: list[Citation] = []
        scores: list[dict[str, float | int]] = []
        for hit in hits:
            stored = self.store.get(hit.chunk_id)
            citations.append(
                Citation(
                    doc_id=stored.doc_id,
                    chunk_id=stored.chunk_id,
                    start=stored.start,
                    end=stored.end,
                    score=hit.score,
                    snippet=stored.text[:160],
                )
            )
            scores.append(
                {
                    "chunk_id": hit.chunk_id,
                    "rrf_score": round(hit.score, 6),
                    "bm25_rank": -1 if hit.bm25_rank is None else hit.bm25_rank,
                    "vector_rank": -1 if hit.vector_rank is None else hit.vector_rank,
                }
            )
        contexts = [self.store.get(h.chunk_id).text for h in hits]
        answer = self.llm.answer(question, contexts) if contexts else "No matching documents found."
        return QueryResult(question=question, answer=answer, citations=citations, scores=scores)

    # -- BM25 persistence -----------------------------------------------------

    def _persist_bm25(self) -> None:
        self.store.kv_set(_BM25_KEY, self.bm25.dump())

    def _restore_bm25(self) -> None:
        data = self.store.kv_get(_BM25_KEY)
        if data is not None:
            self.bm25.load(data)

    def close(self) -> None:
        self.store.close()
