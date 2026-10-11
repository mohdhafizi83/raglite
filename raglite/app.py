"""FastAPI service exposing the RAGLite pipeline.

Endpoints:
    GET  /health   liveness + index size
    POST /ingest   add a document (text + doc_id + optional meta)
    POST /query    hybrid retrieval + extractive answer + citations

Hardening (see SECURITY.md):
    * security headers on every response (CSP, nosniff, frame deny, ...)
    * request body cap: 10 MB (Content-Length checked, chunked streams counted)
    * payload limits: 200k chars per ingested doc, 32 meta keys, 4k-char queries
    * generic JSON error bodies: internals are logged, never echoed to clients
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from raglite.eval import evaluate
from raglite.pipeline import RAGLite
from raglite.sample_corpus import seed

logger = logging.getLogger("raglite.app")

app = FastAPI(title="raglite", version="0.1.0", description="Minimal offline hybrid RAG service.")
engine = RAGLite()

# -- hardening limits ---------------------------------------------------------

MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB per request body
MAX_DOC_CHARS = 200_000  # one document's text per /ingest call
MAX_META_KEYS = 32
MAX_META_VALUE_CHARS = 500
MAX_QUESTION_CHARS = 4_000

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _with_security_headers(response: JSONResponse) -> JSONResponse:
    for key, value in _SECURITY_HEADERS.items():
        response.headers[key] = value
    return response


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Reject oversized bodies and stamp security headers on every response."""
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length.isdigit():
        if int(content_length) > MAX_BODY_BYTES:
            return _with_security_headers(
                JSONResponse(status_code=413, content={"error": "request body too large"})
            )
    if request.method not in ("GET", "HEAD"):
        # Chunked uploads (no Content-Length) or lying Content-Length: buffer the
        # body to enforce the cap. Starlette caches it and replays it downstream.
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return _with_security_headers(
                JSONResponse(status_code=413, content={"error": "request body too large"})
            )
    response = await call_next(request)
    for key, value in _SECURITY_HEADERS.items():
        response.headers[key] = value
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Generic 400 for any invalid payload — no field-level detail leaked."""
    logger.info("validation error on %s %s", request.method, request.url.path)
    return _with_security_headers(
        JSONResponse(status_code=400, content={"error": "invalid request payload"})
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Generic 500: the exception is logged server-side, never sent to the client."""
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return _with_security_headers(
        JSONResponse(status_code=500, content={"error": "internal server error"})
    )


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=MAX_DOC_CHARS)
    meta: dict[str, str] = Field(default_factory=dict)

    @field_validator("meta")
    @classmethod
    def _bound_meta(cls, meta: dict[str, str]) -> dict[str, str]:
        if len(meta) > MAX_META_KEYS:
            raise ValueError(f"meta has more than {MAX_META_KEYS} keys")
        if any(len(str(v)) > MAX_META_VALUE_CHARS for v in meta.values()):
            raise ValueError(f"meta value longer than {MAX_META_VALUE_CHARS} chars")
        return meta


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
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
