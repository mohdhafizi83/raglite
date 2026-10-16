# Configuration — raglite

raglite is configured entirely through environment variables — no config
files. Every variable is optional; the defaults make the pure API hermetic
(in-memory index, built-in embedder).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `RAGLITE_INDEX_PATH` | `:memory:` | Path to the SQLite index file (chunks + vectors + BM25 state). `:memory:` keeps everything in RAM (tests, throwaway demos). Docker sets `/data/index.db` (named volume `raglite-data`). |
| `RAGLITE_UI` | `1` | `1` serves the built-in single-page web UI at `/`; `0` runs the pure JSON API only. |
| `RAGLITE_EMBEDDING_DIM` | `256` | Vector dimension of the built-in `HashingEmbedder`. Larger dims reduce hash collisions at linear-scan cost. |
| `RAGLITE_PORT` | `8000` | **Compose-only**: host port the container publishes on (always bound to `127.0.0.1`). |

`.env.example` documents the same set for `docker compose`.

## Index path & persistence

The whole index (chunk texts, float32 vector BLOBs, serialised BM25 state)
lives in one SQLite file. Point `RAGLITE_INDEX_PATH` at a path on persistent
storage to survive restarts; on startup raglite restores BM25 and rebuilds
embedder corpus stats from the stored chunks, so queries stay consistent with
stored vectors. Delete the file to start empty.

## Embedding providers

The default embedder is `HashingEmbedder` — signed blake2b hashing of token /
bigram features with IDF-style weighting. No model weights, no downloads, no
API keys; deterministic in CI. It captures lexical overlap, not semantics.

Any object satisfying the `Embedder` protocol plugs in with no other code
changes:

```python
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...  # (n, dim) float32, L2-normalised rows
```

Example with sentence-transformers (adds ~90 MB of model weights, needs
network on first run):

```python
# my_app.py
import numpy as np
from sentence_transformers import SentenceTransformer
from raglite.app import app, engine   # or build your own RAGLite()

class STEmbedder:
    dim = 384
    def __init__(self):
        self._m = SentenceTransformer("all-MiniLM-L6-v2")
    def embed(self, texts):
        return self._m.encode(texts, normalize_embeddings=True).astype(np.float32)

engine.embedder = STEmbedder()
```

The same seam logic applies to generation: `LLMClient.answer(question,
contexts)` swaps the offline extractive `EchoLLM` for any generator.

## Payload limits (hard-coded, see `raglite/app.py`)

| Constant | Value | Meaning |
|---|---|---|
| `MAX_BODY_BYTES` | 10 MB | Request body cap → 413 |
| `MAX_DOC_CHARS` | 200 000 | Text per `/ingest` call |
| `MAX_META_KEYS` | 32 | Meta keys per document |
| `MAX_META_VALUE_CHARS` | 500 | Meta value length |
| `MAX_QUESTION_CHARS` | 4 000 | Query length |

Chunking defaults (`RAGLite(chunk_size=400, chunk_overlap=80)`) are constructor
arguments, not env vars — set them if you build the engine in your own code.

## Advanced values

- RRF constant `k=60` and candidate pool `top-20` per ranker:
  `raglite/retrieval.py` / `RAGLite.retrieve`.
- BM25 parameters `k1=1.5`, `b=0.75`: `raglite/store.py::BM25Index`.
- uvicorn log level: `raglite serve` uses `info`; run
  `uvicorn raglite.app:app --log-level warning` for quieter logs.
