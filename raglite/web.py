"""Optional single-page web UI, mounted by ``raglite.app`` (RAGLITE_UI=0 disables).

One self-contained HTML page — no SPA framework, no CDN, no build step. All CSS
is inline and follows the Fixzy design tokens (dark-first palette, Inter font
stack, 8px radii, 4px spacing scale). The page is a thin client over the same
JSON endpoints the API already exposes (``/health``, ``/query``, ``/ingest``,
``/seed``) plus one read-only listing endpoint added here (``GET /api/docs``).

Security notes:
    * Every dynamic string is rendered via ``textContent`` — document text can
      never be interpreted as HTML (no innerHTML anywhere).
    * The page sets its own CSP (relaxed only for its own inline style/script,
      ``connect-src 'self'`` so it can only talk to this origin). The app
      middleware leaves a route-set CSP untouched.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from raglite.pipeline import RAGLite

UI_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
)

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>raglite — offline hybrid RAG</title>
<style>
:root{
  --bg:#0f1117; --surface:#161923; --border:#2a2f3d; --text:#e2e8f0;
  --muted:#8b93a7; --primary:#6366f1; --success:#22c55e; --danger:#ef4444;
  --warning:#f59e0b;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:400 14px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{height:56px;display:flex;align-items:center;gap:12px;padding:0 24px;
  border-bottom:1px solid var(--border);background:var(--surface)}
header strong{font-size:20px;font-weight:600;letter-spacing:-0.01em}
main{max-width:960px;margin:0 auto;padding:24px;display:grid;gap:24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:24px;display:grid;gap:16px}
h1{font-size:16px;font-weight:600;margin:0}
h2{font-size:14px;font-weight:600;margin:0}
.muted{color:var(--muted)}
.small{font-size:12px}
.row{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}
.gap{display:flex;gap:8px;align-items:center}
input,textarea{width:100%;background:var(--bg);color:var(--text);
  border:1px solid var(--border);border-radius:8px;padding:8px 12px;font:inherit}
input:focus,textarea:focus{outline:2px solid var(--primary);outline-offset:-1px}
textarea{resize:vertical}
.btn{background:var(--primary);border:1px solid transparent;color:#fff;border-radius:8px;
  padding:8px 16px;font:inherit;font-weight:600;cursor:pointer}
.btn.secondary{background:transparent;border-color:var(--border);color:var(--text)}
.btn:hover{filter:brightness(1.1)}
.btn:disabled{opacity:.5;cursor:default}
.pill{border:1px solid var(--border);border-radius:8px;padding:4px 8px;
  font-size:12px;color:var(--muted)}
.pill.ok{color:var(--success);border-color:var(--success)}
.pill.err{color:var(--danger);border-color:var(--danger)}
.answer{background:var(--bg);border:1px solid var(--border);border-left:4px solid var(--primary);
  border-radius:8px;padding:16px;font-size:15px}
.cite{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px 16px}
.cite .head{display:flex;gap:8px;flex-wrap:wrap;align-items:baseline}
.cite .doc{color:var(--primary);font-weight:600}
.cite .snippet{margin:8px 0 4px;color:var(--text)}
.cite .rank{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.empty{display:grid;justify-items:center;gap:8px;text-align:center;padding:24px;
  border:1px dashed var(--border);border-radius:8px}
.empty svg{stroke:var(--muted)}
.error{background:var(--surface);border:1px solid var(--danger);border-radius:8px;
  padding:12px 16px;display:flex;gap:12px;align-items:center;color:var(--danger)}
.skeleton{background:linear-gradient(90deg,var(--surface),var(--border),var(--surface));
  background-size:200% 100%;animation:sh 1.2s infinite;border-radius:8px;height:16px;margin:8px 0}
.skeleton.tall{height:64px}
@keyframes sh{0%{background-position:200% 0}100%{background-position:-200% 0}}
.ok-msg{color:var(--success)} .err-msg{color:var(--danger)}
form{display:grid;gap:12px}
footer{color:var(--muted);text-align:center;padding:16px 0 32px;font-size:12px}
</style>
</head>
<body>
<header>
  <strong>raglite</strong>
  <span class="muted">offline hybrid RAG · BM25 + vectors + RRF</span>
  <span id="status" class="pill">connecting…</span>
</header>
<main>
  <section class="card">
    <h1>Ask the index</h1>
    <form id="qform">
      <input id="q" placeholder="e.g. How long is the warranty on the Nimbus 900?"
             maxlength="4000" autocomplete="off" required>
      <div><button class="btn" type="submit">Search</button></div>
    </form>
    <div id="qout"></div>
  </section>
  <section class="card">
    <h1>Ingest a document</h1>
    <form id="iform">
      <input id="docid" placeholder="doc_id, e.g. runbook.txt" maxlength="200" required>
      <textarea id="dtext" rows="6" placeholder="Paste the document text…"
                maxlength="200000" required></textarea>
      <div class="row"><button class="btn" type="submit">Ingest</button>
      <span id="istatus" class="muted"></span></div>
    </form>
  </section>
  <section class="card">
    <div class="row"><h1>Documents</h1>
      <div class="gap">
        <button id="seedbtn" class="btn secondary" type="button">Seed sample corpus</button>
        <button id="refreshbtn" class="btn secondary" type="button">Refresh</button>
      </div>
    </div>
    <div id="docs"></div>
  </section>
</main>
<footer>raglite · every answer is a verbatim sentence with a machine-checkable
character-span citation · <span class="muted">API docs at /docs</span></footer>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (e) { /* non-JSON response */ }
  if (!r.ok) {
    const msg = body && (body.detail || body.error) ? (body.detail || body.error)
                                                    : r.status + " " + r.statusText;
    throw new Error(msg);
  }
  return body;
}
function skeleton(n) {
  const box = el("div");
  for (let i = 0; i < n; i++) box.appendChild(el("div", i === 0 ? "skeleton tall" : "skeleton"));
  return box;
}
function errorBox(msg, retry) {
  const box = el("div", "error");
  box.appendChild(el("span", null, msg));
  const b = el("button", "btn secondary", "Try again");
  b.type = "button";
  b.addEventListener("click", retry);
  box.appendChild(b);
  return box;
}
function emptyDocs() {
  const box = el("div", "empty");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", "32"); svg.setAttribute("height", "32");
  svg.setAttribute("viewBox", "0 0 24 24"); svg.setAttribute("fill", "none");
  svg.setAttribute("stroke-width", "1.5"); svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  const p1 = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p1.setAttribute("d", "M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z");
  const p2 = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p2.setAttribute("d", "M14 2v4a2 2 0 0 0 2 2h4");
  svg.appendChild(p1); svg.appendChild(p2);
  box.appendChild(svg);
  box.appendChild(el("p", "muted",
    "No documents indexed yet. Seed the sample corpus or ingest one above."));
  const b = el("button", "btn", "Seed sample corpus");
  b.type = "button";
  b.addEventListener("click", seed);
  box.appendChild(b);
  return box;
}
async function refreshStatus() {
  const pill = $("status");
  try {
    const h = await api("/health");
    pill.textContent = h.chunks + " chunks indexed";
    pill.classList.add("ok"); pill.classList.remove("err");
  } catch (e) {
    pill.textContent = "service unreachable";
    pill.classList.add("err"); pill.classList.remove("ok");
  }
}
async function refreshDocs() {
  const box = $("docs");
  box.textContent = "";
  box.appendChild(skeleton(2));
  try {
    const data = await api("/api/docs");
    box.textContent = "";
    if (!data.docs.length) { box.appendChild(emptyDocs()); return; }
    const table = el("table");
    const thead = el("thead");
    const hr = el("tr");
    hr.appendChild(el("th", null, "doc_id"));
    hr.appendChild(el("th", null, "chunks"));
    thead.appendChild(hr);
    table.appendChild(thead);
    const tb = el("tbody");
    for (const d of data.docs) {
      const tr = el("tr");
      tr.appendChild(el("td", null, d.doc_id));
      tr.appendChild(el("td", "num", String(d.chunks)));
      tb.appendChild(tr);
    }
    table.appendChild(tb);
    box.appendChild(table);
  } catch (e) {
    box.textContent = "";
    box.appendChild(errorBox("Could not load documents: " + e.message, refreshDocs));
  }
}
async function seed() {
  const s = $("istatus");
  s.textContent = "seeding sample corpus…";
  try {
    const r = await api("/seed", { method: "POST" });
    s.textContent = "seeded " + r.chunks + " chunks";
    s.className = "ok-msg";
  } catch (e) {
    s.textContent = "seed failed: " + e.message;
    s.className = "err-msg";
  }
  refreshStatus(); refreshDocs();
}
function renderCitations(r) {
  const box = $("qout");
  const ans = el("div", "answer", r.answer || "No matching documents found.");
  box.appendChild(el("h2", "muted", "Answer"));
  box.appendChild(ans);
  if (!r.citations.length) return;
  box.appendChild(el("h2", "muted", "Citations"));
  for (let i = 0; i < r.citations.length; i++) {
    const c = r.citations[i];
    const card = el("div", "cite");
    const head = el("div", "head");
    head.appendChild(el("span", "doc", c.doc_id));
    head.appendChild(el("span", "muted small",
      "chars " + c.start + "–" + c.end + " · rrf " + Number(c.score).toFixed(4)));
    card.appendChild(head);
    card.appendChild(el("p", "snippet", c.snippet));
    const s = r.scores[i];
    if (s) {
      card.appendChild(el("div", "rank",
        "bm25 rank " + (s.bm25_rank >= 0 ? s.bm25_rank : "—") +
        " · vector rank " + (s.vector_rank >= 0 ? s.vector_rank : "—")));
    }
    box.appendChild(card);
  }
}
$("qform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  const out = $("qout");
  out.textContent = "";
  out.appendChild(skeleton(3));
  try {
    const r = await api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });
    out.textContent = "";
    renderCitations(r);
  } catch (e) {
    out.textContent = "";
    out.appendChild(errorBox("Search failed: " + e.message,
      () => $("qform").requestSubmit()));
  }
});
$("iform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const s = $("istatus");
  const docId = $("docid").value.trim();
  const text = $("dtext").value;
  if (!docId || !text.trim()) return;
  s.textContent = "ingesting…"; s.className = "muted";
  try {
    const r = await api("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId, text: text })
    });
    s.textContent = "ingested " + r.chunks + " chunks from " + r.doc_id;
    s.className = "ok-msg";
    $("dtext").value = "";
  } catch (e) {
    s.textContent = "ingest failed: " + e.message;
    s.className = "err-msg";
  }
  refreshStatus(); refreshDocs();
});
$("seedbtn").addEventListener("click", seed);
$("refreshbtn").addEventListener("click", refreshDocs);
refreshStatus(); refreshDocs();
</script>
</body>
</html>
"""


def mount_ui(app: FastAPI, engine: RAGLite) -> None:
    """Attach the single-page UI (``GET /``) and document listing (``GET /api/docs``)."""

    @app.get("/", include_in_schema=False)
    def ui_page() -> HTMLResponse:
        response = HTMLResponse(_PAGE)
        response.headers["Content-Security-Policy"] = UI_CSP
        return response

    @app.get("/api/docs", tags=["ui"])
    def list_docs() -> dict[str, list[dict[str, object]]]:
        """Documents currently in the index: ``doc_id`` + chunk count."""
        return {
            "docs": [
                {"doc_id": doc_id, "chunks": n} for doc_id, n in engine.store.doc_summary()
            ]
        }
