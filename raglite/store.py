"""Storage layer: a SQLite-backed vector store and a pure-python BM25 index.

Vectors are stored as raw float32 BLOBs; cosine similarity is computed in one
numpy pass over the loaded matrix. For a portfolio-scale corpus (thousands of
chunks) a linear scan is the right call: zero moving parts, exact results,
trivially testable. An ANN index would be the honest next step at ~100k chunks.

The BM25 index lives in the same SQLite file (term frequencies serialised as a
BLOB) so both retrieval paths always see the same corpus.
"""

from __future__ import annotations

import math
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raglite.embedders import tokenize


@dataclass(frozen=True)
class StoredChunk:
    """One indexed chunk with its document metadata and exact source span."""

    chunk_id: int
    doc_id: str
    text: str
    start: int
    end: int
    meta: str  # JSON string, opaque to the store


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id    TEXT NOT NULL,
    text      TEXT NOT NULL,
    start     INTEGER NOT NULL,
    end       INTEGER NOT NULL,
    meta      TEXT NOT NULL DEFAULT '{}',
    vector    BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value BLOB
);
"""


class VectorStore:
    """SQLite-backed store: chunk text + metadata + float32 vector BLOBs."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add(
        self,
        doc_id: str,
        text: str,
        start: int,
        end: int,
        meta: str,
        vector: np.ndarray,
    ) -> int:
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        cur = self._conn.execute(
            "INSERT INTO chunks (doc_id, text, start, end, meta, vector) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, text, start, end, meta, blob),
        )
        self._conn.commit()
        lastrowid = cur.lastrowid
        if lastrowid is None:  # pragma: no cover - sqlite always assigns rowid here
            raise RuntimeError("insert did not return a row id")
        return int(lastrowid)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def get(self, chunk_id: int) -> StoredChunk:
        row = self._conn.execute(
            "SELECT chunk_id, doc_id, text, start, end, meta FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no chunk {chunk_id}")
        return StoredChunk(*row)

    def all_ids(self) -> list[int]:
        rows = self._conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()
        return [int(r[0]) for r in rows]

    def all_texts(self) -> list[tuple[int, str]]:
        rows = self._conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    def update_vector(self, chunk_id: int, vector: np.ndarray) -> None:
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        self._conn.execute("UPDATE chunks SET vector = ? WHERE chunk_id = ?", (blob, chunk_id))
        self._conn.commit()

    def search_vectors(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Cosine top-k over all stored vectors (linear scan, exact)."""
        rows = self._conn.execute(
            "SELECT chunk_id, vector FROM chunks WHERE vector IS NOT NULL"
        ).fetchall()
        if not rows:
            return []
        ids = [int(r[0]) for r in rows]
        mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        q = np.asarray(query, dtype=np.float32)
        norm_q = float(np.linalg.norm(q))
        if norm_q > 0:
            q = q / norm_q
        sims = mat @ q / np.maximum(np.linalg.norm(mat, axis=1), 1e-9)
        k = min(top_k, len(ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(ids[i], float(sims[i])) for i in top]

    def delete_doc(self, doc_id: str) -> int:
        cur = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return cur.rowcount

    def kv_set(self, key: str, value: bytes) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def kv_get(self, key: str) -> bytes | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])


class BM25Index:
    """Okapi BM25 over the same chunks, kept in-process, serialised to SQLite.

    Pure python, no dependencies beyond the stdlib. ``search`` returns
    ``(chunk_id, score)`` pairs sorted by descending score.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        # chunk_id -> {term: tf}
        self._postings: dict[int, dict[str, int]] = {}
        self._df: dict[str, int] = {}
        self._doc_len: dict[int, int] = {}
        self._total_len = 0

    def add(self, chunk_id: int, text: str) -> None:
        toks = tokenize(text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self._postings[chunk_id] = tf
        self._doc_len[chunk_id] = len(toks)
        self._total_len += len(toks)
        for t in tf:
            self._df[t] = self._df.get(t, 0) + 1

    def remove(self, chunk_id: int) -> None:
        tf = self._postings.pop(chunk_id, None)
        if tf is None:
            return
        self._total_len -= self._doc_len.pop(chunk_id, 0)
        for t in tf:
            self._df[t] -= 1
            if self._df[t] <= 0:
                del self._df[t]

    def __len__(self) -> int:
        return len(self._postings)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        n = len(self._postings)
        if n == 0:
            return []
        avg_len = self._total_len / n
        scores: dict[int, float] = {}
        for term in tokenize(query):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for cid, tf in self._postings.items():
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = self._doc_len[cid]
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / avg_len)
                scores[cid] = scores.get(cid, 0.0) + idf * f * (self.k1 + 1.0) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    # -- persistence (pickle BLOB in a side table) ---------------------------

    def dump(self) -> bytes:
        state = (self.k1, self.b, self._postings, self._df, self._doc_len, self._total_len)
        return pickle.dumps(state)

    def load(self, data: bytes) -> None:
        state = pickle.loads(data)
        self.k1, self.b, self._postings, self._df, self._doc_len, self._total_len = state
