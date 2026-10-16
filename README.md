# raglite

[![ci](https://github.com/mohdhafizi83/raglite/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mohdhafizi83/raglite/actions/workflows/ci.yml)

A small, honest hybrid RAG service: BM25 + vector retrieval fused with
Reciprocal Rank Fusion, served by FastAPI, with a built-in eval harness — and
a demo that runs with **zero credentials, zero downloads, zero network**.

## The problem this solves

Naive top-k vector RAG fails in ways that only show up once you measure:

1. **Lexical misses.** A pure-embedding retriever can rank a paraphrase above
   the passage that contains the exact term the user searched ("Wi-Fi 6E",
   error codes, serial prefixes like `NB9-`). Dense similarity blurs rare,
   exact tokens — exactly the tokens people type into support search boxes.
2. **Unmeasurable quality.** Most toy RAG demos can't tell you whether they got
   better or worse after a change, because quality is judged by vibes. Without
   a golden set and hit-rate/MRR numbers, "we added a vector DB" is a
   configuration change, not an improvement.
3. **Demos that can't be reviewed.** If running the demo needs an OpenAI key,
   no reviewer, CI runner, or future teammate can actually run it.

raglite's answer: **hybrid retrieval (BM25 + vectors, fused with RRF) plus a
golden-set eval that runs in CI** — so retrieval quality is a number you can
watch, and the whole thing is hermetically offline.

## Quick start

```bash
pip install -e .
raglite demo        # seeds a sample corpus, answers 3 questions, prints eval
raglite eval        # golden-set eval: hit-rate@3 and MRR
raglite serve       # FastAPI + web UI on http://127.0.0.1:8000 (POST /ingest, POST /query)
```

Or with Docker (index persists in a named volume):

```bash
cp .env.example .env    # optional — all defaults are sane
docker compose up -d --build
```

Open **http://127.0.0.1:8000** for the built-in web UI: query box with scores
and citations, ingest form, document list. Dark-first design tokens, no CDN,
no build step; disable with `RAGLITE_UI=0` for the pure JSON API. Full setup:
[docs/getting-started.md](docs/getting-started.md) · every knob:
[docs/configuration.md](docs/configuration.md) · measuring quality:
[docs/eval-harness.md](docs/eval-harness.md).

### Exact demo output (`raglite demo`)

```
raglite 0.1.0 — offline demo (no API keys)
seeded 6 sample docs (11 chunks) from the Nimbus 900 manual

Q: How long is the warranty on the Nimbus 900?
A: The Nimbus 900 carries a standard two-year limited warranty covering manufacturing defects in materials and workmanship.
   [nimbus_warranty.txt chars 0-225] rrf=0.0328
   [nimbus_privacy.txt chars 0-331] rrf=0.0323
   [nimbus_warranty.txt chars 226-538] rrf=0.0315

Q: How do I connect the Nimbus 900 to Wi-Fi?
A: To connect Wi-Fi, open Settings, choose Network, and select your access point.
   [nimbus_setup.txt chars 0-345] rrf=0.0328
   [nimbus_troubleshooting.txt chars 0-369] rrf=0.0323
   [nimbus_warranty.txt chars 0-225] rrf=0.0315

Q: What should I do if the status LED blinks red?
A: If the status LED blinks red, the device has a firmware fault.
   [nimbus_troubleshooting.txt chars 0-369] rrf=0.0328
   [nimbus_battery.txt chars 364-518] rrf=0.0320
   [nimbus_setup.txt chars 0-345] rrf=0.0311

eval on 6 golden questions: hit-rate@3=1.00  MRR=1.00
```

Every answer is a verbatim sentence from a cited document, and every citation
carries the exact `[start, end)` character span in the source — you can check
the grounding by eye.

## Design notes

**Why RRF, not score blending.** BM25 scores are unbounded and corpus-dependent;
cosine similarities live in [-1, 1]. Any weighted blend `α·bm25 + (1-α)·cos`
needs per-corpus tuning of α and score normalisation. Reciprocal Rank Fusion
(`score = Σ 1/(k + rank)`, k=60) uses only the *orderings*, is robust out of
the box, and has one interpretable constant. A chunk that is mediocre in both
rankers beats a chunk that is first in one and absent from the other — which
is exactly the behaviour you want from hybrid retrieval.

**Why a hashing embedder is a feature, not a shortcut.** `HashingEmbedder`
embeds text via signed blake2b hashing of token/bigram features with an
IDF-style weighting — no model weights, no downloads, no API keys. That means:
the demo, the tests, and CI all run hermetically offline and deterministically;
a reviewer gets identical numbers to mine. The `Embedder` protocol is the
seam: swap in `sentence-transformers` or a vendor API in ~20 lines and nothing
else in the codebase changes. **The interface is the point.**

**Why SQLite + linear scan.** Vectors are float32 BLOBs in SQLite; cosine is
one numpy matmul over the loaded matrix. At portfolio scale (thousands of
chunks) this is exact, dependency-free, and trivially testable. An ANN index
(HNSW) is the honest next step around ~100k chunks — the `VectorStore`
interface is what would change, not the pipeline.

**Citations are spans, not vibes.** The chunker records exact `[start, end)`
offsets into the source document through the whole separator cascade, so a
citation like `nimbus_warranty.txt chars 0-225` is machine-checkable. A test
asserts `source[start:end]` round-trips.

**Eval is a first-class citizen.** `raglite/eval.py` holds a 6-question golden
set over the built-in corpus and reports hit-rate@3 and MRR. A pytest test
asserts hit-rate ≥ 0.99, so retrieval regressions fail CI, not someone's demo.

## Architecture

```
ingest(text) ──> chunker (recursive splitter, overlap, exact spans)
                    ├─> HashingEmbedder ──> SQLite vector store (BLOBs)
                    └─> BM25Index (pure python, persisted in same SQLite)

query(q) ──> BM25 top-20  ┐
       └──> cosine top-20  ┴─> RRF fusion ─> top-k chunks
                                   ├─> EchoLLM (extractive, offline) ─> answer
                                   └─> citations (doc_id + char span + scores)
```

Modules: `chunker.py`, `embedders.py`, `store.py`, `retrieval.py`, `llm.py`,
`pipeline.py`, `eval.py`, `app.py` (FastAPI), `web.py` (optional UI), `cli.py`.

## API

- `GET /` — built-in web UI (disable with `RAGLITE_UI=0`)
- `GET /health` — liveness + index size
- `POST /ingest` — `{doc_id, text, meta?}` → chunk count
- `POST /query` — `{question, top_k?}` → `{answer, citations[], scores[]}`
- `POST /seed` — load the built-in Nimbus 900 sample corpus
- `GET /eval` — golden-set hit-rate@3 / MRR
- `GET /api/docs` — documents in the index (UI helper)

## Security

raglite is a **local dev/demo service with no auth by design** — bind it to
localhost and keep it there. If you ever front it beyond localhost, put a
reverse proxy with auth + TLS in front. Every response carries strict security
headers (CSP `default-src 'none'`, `X-Frame-Options: DENY`, nosniff,
no-store), request bodies are capped at 10 MB (413), ingest/query payloads
have documented size limits, and errors are generic JSON — internals are
logged, never echoed. CI additionally runs bandit, pip-audit (pinned
`requirements.txt`), and gitleaks. Full threat model: [SECURITY.md](SECURITY.md).

## Honest limitations

- **The hashing embedder is a stand-in for real embeddings.** It captures
  lexical overlap, not semantics — it will not match "can't connect to the
  wireless" to "Wi-Fi setup". That is deliberate: the `Embedder` protocol is
  the integration point, and the offline default buys deterministic CI and a
  demo anyone can run.
- **EchoLLM is extractive, not generative.** It picks the best-matching
  sentence; it cannot synthesise across passages. Same reasoning: the
  `LLMClient` protocol is the seam.
- **Linear-scan vector search** is O(n) per query. Fine to ~10⁴–10⁵ chunks,
  wrong beyond; the store interface isolates that choice.
- **The golden set is small (6 questions)** and matches the built-in corpus.
  It proves the harness works and catches regressions; it is not a benchmark.

## Development

```bash
pip install -e .[dev]
ruff check .
pytest          # 50 tests, zero network
```

CI (GitHub Actions, Python 3.11 + 3.12): install, `ruff check`, `pytest`,
plus a security job (bandit + pip-audit) and gitleaks. All actions are pinned
to full commit SHAs.

## License

MIT
