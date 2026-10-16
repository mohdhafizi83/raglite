# Getting Started — raglite

Target: a working raglite in under 10 minutes, with zero API keys and zero
model downloads.

## Requirements

- Docker + Docker Compose (recommended), **or**
- Python ≥ 3.11 (native install)

## Quickstart (Docker — recommended)

```bash
git clone https://github.com/mohdhafizi83/raglite.git
cd raglite
cp .env.example .env      # optional — every value has a sane default
docker compose up -d --build
```

Wait until the container reports healthy:

```bash
docker compose ps         # STATUS: healthy
curl -s http://127.0.0.1:8000/health
# {"status":"ok","chunks":0}
```

Open **http://127.0.0.1:8000** — the web UI loads with an empty index.
Click **Seed sample corpus**, then ask:
*"How long is the warranty on the Nimbus 900?"* — you get a verbatim-sentence
answer with character-span citations. The index lives in the `raglite-data`
named volume, so it survives `docker compose down` and rebuilds.

## Install without Docker

```bash
git clone https://github.com/mohdhafizi83/raglite.git
cd raglite
python -m venv .venv && source .venv/bin/activate
pip install -e .

raglite demo              # seeds a sample corpus, answers 3 questions, prints eval
raglite eval              # golden-set eval: hit-rate@3 and MRR
raglite serve             # FastAPI + web UI on http://127.0.0.1:8000
```

To persist the index across restarts with a plain serve:

```bash
RAGLITE_INDEX_PATH=./data/index.db raglite serve
```

## First real steps (either install)

1. Ingest your own document:

   ```bash
   curl -s -X POST http://127.0.0.1:8000/ingest \
     -H 'Content-Type: application/json' \
     -d '{"doc_id":"runbook.txt","text":"The staging server reboots every Sunday at 03:00 UTC."}'
   ```

2. Query it:

   ```bash
   curl -s -X POST http://127.0.0.1:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"question":"When does the staging server reboot?"}'
   ```

   The `answer` is a verbatim sentence from `runbook.txt` and `citations[0]`
   carries the exact `[start, end)` character span. The same flow works in the
   web UI (ingest form + query box).

3. Check retrieval quality: `curl -s http://127.0.0.1:8000/eval` (after seeding)
   or `raglite eval` — see [the eval guide](eval-harness.md).

## Next

- [Configuration](configuration.md) — every environment variable, embedding
  providers, index path
- [Eval harness guide](eval-harness.md) — measuring retrieval quality
- [SECURITY.md](../SECURITY.md) — threat model; raglite is a localhost service
  with no auth by design
