"""raglite: a small, fully-offline hybrid RAG service.

BM25 + hashing vector retrieval fused with Reciprocal Rank Fusion,
served by FastAPI, with a built-in eval harness. No API keys, ever.
"""

from raglite.chunker import Chunk, split_text
from raglite.embedders import Embedder, HashingEmbedder
from raglite.llm import EchoLLM, LLMClient
from raglite.pipeline import Citation, QueryResult, RAGLite
from raglite.retrieval import ScoredChunk, hybrid_search, rrf_fuse
from raglite.store import BM25Index, StoredChunk, VectorStore

__version__ = "0.1.0"

__all__ = [
    "BM25Index",
    "Chunk",
    "Citation",
    "EchoLLM",
    "Embedder",
    "HashingEmbedder",
    "LLMClient",
    "QueryResult",
    "RAGLite",
    "ScoredChunk",
    "StoredChunk",
    "VectorStore",
    "__version__",
    "hybrid_search",
    "rrf_fuse",
    "split_text",
]
