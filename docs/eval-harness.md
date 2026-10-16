# Running the eval harness

raglite treats retrieval quality as a number, not a vibe. `raglite/eval.py`
ships a 6-question golden set over the bundled Nimbus 900 sample corpus and
reports two metrics:

- **hit-rate@3** — is a gold document in the top-3 retrieved chunks?
- **MRR** — mean reciprocal rank of the first gold document.

Both run fully offline and deterministically: same numbers in CI as on your
laptop.

## CLI

```bash
raglite eval
```

```
[PASS] How long is the warranty on the Nimbus 900?
       gold=['nimbus_warranty.txt'] top=['nimbus_warranty.txt', ...]
...
hit-rate@3=1.00  MRR=1.00
```

Exit code is `0` when hit-rate@3 ≥ 0.99, `1` otherwise — so it works as a CI
gate on its own. `raglite demo` also prints the mini-eval after answering
three sample questions.

## HTTP

With a seeded index (the web UI's *Seed sample corpus* button, or `POST /seed`):

```bash
curl -s -X POST http://127.0.0.1:8000/seed
curl -s http://127.0.0.1:8000/eval
```

Returns `hit_rate_at_3`, `mrr`, `n` and a `per_question` breakdown with the
retrieved top docs per question.

## CI gate

`tests/test_pipeline_api.py` asserts hit-rate ≥ 0.99, so any retrieval
regression (chunker change, fusion tweak, embedder swap) fails `pytest` — and
therefore CI — before it ships.

## Measuring your own corpus

The harness is corpus-agnostic: build a `RAGLite` engine, ingest your
documents, and pass your own golden list:

```python
from raglite.eval import GoldenQA, evaluate
from raglite.pipeline import RAGLite

engine = RAGLite(store_path="./data/index.db")
# ... ingest your documents ...
golden = [GoldenQA("What is our refund window?", ("policy.txt",))]
report = evaluate(engine, golden=golden, top_k=3)
print(report.hit_rate_at_3, report.mrr)
```

Practical notes:

- A golden question needs at least one `doc_id` that *contains* the answer;
  aim for 10–50 questions covering your real query mix.
- Re-run the eval after every chunking/fusion/embedder change — that is the
  whole point. `hit-rate@3` moving is a signal, not noise.
- The bundled 6-question set proves the harness works; it is not a benchmark.
