# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-10-16

First product-grade release of raglite: a small, honest, fully-offline hybrid
RAG service.

### Added

- **Hybrid retrieval engine** — Okapi BM25 + hashing-trick vectors (signed
  blake2b, IDF-style weighting) fused with Reciprocal Rank Fusion (k=60).
  Zero credentials, zero downloads, deterministic results.
- **SQLite storage layer** — float32 vector BLOBs + persisted BM25 state in one
  file; exact linear-scan cosine search; index survives restarts.
- **FastAPI service** — `GET /health`, `POST /ingest`, `POST /query`,
  `POST /seed`, `GET /eval`; answers are verbatim sentences with machine-
  checkable `[start, end)` character-span citations.
- **Built-in single-page web UI** (`raglite/web.py`, optional, `RAGLITE_UI=0`
  disables) — query box with scores + citations, document ingest form,
  document list with empty/loading/error states. Dark-first design tokens,
  no CDN, no build step, XSS-safe rendering.
- **Golden-set eval harness** — 6-question golden set over the bundled Nimbus
  900 corpus reporting hit-rate@3 and MRR (`raglite eval`, `GET /eval`);
  a pytest gate fails CI on retrieval regressions.
- **CLI** — `raglite demo | eval | serve`.
- **Hardening** — strict security headers on every response, 10 MB body cap
  (413), documented payload limits, generic JSON errors (internals logged,
  never echoed). Threat model in SECURITY.md.
- **Docker deployment** — `Dockerfile` (non-root, healthcheck) +
  `docker-compose.yml` with a named volume for the index and `.env.example`.
- **Configuration via environment** — `RAGLITE_INDEX_PATH`, `RAGLITE_UI`,
  `RAGLITE_EMBEDDING_DIM` (see `docs/configuration.md`).
- **Docs** — `docs/getting-started.md` (Docker + pip), `docs/configuration.md`,
  `docs/eval-harness.md`.
- **CI** — pytest + ruff on Python 3.11/3.12, bandit, pip-audit, gitleaks;
  release workflow (tag → GitHub Release + GHCR image), all actions SHA-pinned.

[Unreleased]: https://github.com/mohdhafizi83/raglite/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mohdhafizi83/raglite/releases/tag/v0.1.0
