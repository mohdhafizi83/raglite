# Security Policy

## Scope and honest threat model

raglite is a **local dev/demo service**. It is designed to run on
`127.0.0.1` on a developer machine, with **no authentication by design** —
there are no users, no accounts, and no multi-tenant data to protect. The
index is throwaway: whatever you ingest is whatever you query.

**Do not expose this service to untrusted networks directly.** If you ever
front it beyond localhost (LAN demo, review environment), put a reverse proxy
(nginx, Caddy, Traefik) in front that adds authentication and TLS, and bind
uvicorn to localhost only. The application itself will never grow auth; that
is deliberate scope, not an oversight.

### Assets

| Asset | Threat | Mitigation |
| --- | --- | --- |
| Host process availability | Memory/CPU exhaustion via huge payloads | 10 MB request body cap (413), 200k-char doc cap, 4k-char query cap |
| Response consumers (browsers) | XSS / clickjacking if responses are ever rendered | Strict CSP (`default-src 'none'`), `X-Frame-Options: DENY`, `nosniff`, `base-uri 'none'`, `form-action 'none'`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` |
| Information disclosure | Stack traces / validation internals echoed to clients | Generic JSON errors: `400 {"error":"invalid request payload"}`, `500 {"error":"internal server error"}`; details logged server-side only |
| Supply chain | Vulnerable pinned dependencies | CI runs `pip-audit --strict` against `requirements.txt` (fully pinned freeze) and `bandit` on the package; all actions pinned to full commit SHAs |
| Secrets in repo | Accidentally committed credentials | gitleaks job on every push/PR |

### Input validation table

| Input | Limit | Rejected with |
| --- | --- | --- |
| Request body (any endpoint) | 10 MB (`MAX_BODY_BYTES`) | `413 {"error":"request body too large"}` |
| `doc_id` | 1–200 chars | `400` generic |
| `text` (ingest doc) | 1–200,000 chars (`MAX_DOC_CHARS`) | `400` generic |
| `meta` | ≤ 32 keys, values ≤ 500 chars | `400` generic |
| `question` | 1–4,000 chars (`MAX_QUESTION_CHARS`) | `400` generic |
| `top_k` | 1–10 | `400` generic |
| JSON body | must parse | `400` generic (no parser internals echoed) |

The BM25 index persistence uses `pickle` on BLOBs written by the same process
into its own SQLite file (never on network input); the DB path is
operator-controlled. This is flagged `# nosec` with justification in
`raglite/store.py`.

## Out of scope

- Authentication / authorization / multi-user isolation (single-user local tool)
- TLS termination (reverse proxy's job if you go beyond localhost)
- Rate limiting (local tool; a proxy can add it if ever exposed)
- Hardening against a malicious *SQLite file* or tampered local disk (if an
  attacker controls your filesystem, the pickle note above is the least of it)
- Prompt injection / LLM safety: raglite is extractive and fully offline —
  there is no generative LLM in the loop to prompt-inject
- DoS resistance against a determined local attacker (they already run code as you)

## Reporting a vulnerability

Open a [private security advisory](https://github.com/mohdhafizi83/raglite/security/advisories/new).
Expect a response within a week; this is a portfolio project, not a company.
