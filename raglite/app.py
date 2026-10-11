"""FastAPI service exposing the RAGLite pipeline.

Endpoints:
    GET  /health   liveness + index size
    POST /ingest   add a document (text + doc_id + optional meta)
    POST /query    hybrid retrieval + extractive answer + citations
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from raglite.eval import evaluate
from raglite.pipeline import RAGLite
from raglite.sample_corpus import seed

app = FastAPI(title="raglite", version="0.1.0", description="Minimal offline hybrid RAG service.")
engine = RAGLite()


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    meta: dict[str, str] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    scores: list[dict]


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "chunks": engine.store.count()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    n = engine.ingest(req.text, doc_id=req.doc_id, meta=req.meta)
    return IngestResponse(doc_id=req.doc_id, chunks=n)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if engine.store.count() == 0:
        raise HTTPException(status_code=409, detail="index is empty; POST /ingest first")
    result = engine.query(req.question, top_k=req.top_k)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        citations=[c.to_dict() for c in result.citations],
        scores=result.scores,
    )


@app.post("/seed")
def seed_samples() -> dict[str, int]:
    """Load the built-in Nimbus 900 sample corpus (used by the demo)."""
    return {"chunks": seed(engine)}


@app.get("/eval")
def eval_endpoint() -> dict[str, object]:
    if engine.store.count() == 0:
        raise HTTPException(status_code=409, detail="index is empty; POST /seed first")
    return evaluate(engine).to_dict()
