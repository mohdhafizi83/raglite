"""Golden evaluation set for the built-in Nimbus 900 sample corpus.

Each question lists the doc_ids that contain the answer. The harness measures
hit-rate@3 (is a gold doc in the top-3 retrieved chunks?) and MRR over the
retrieval ranking. This runs fully offline — same numbers in CI as on your
laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

from raglite.pipeline import RAGLite


@dataclass(frozen=True)
class GoldenQA:
    question: str
    gold_doc_ids: tuple[str, ...]


GOLDEN_SET: list[GoldenQA] = [
    GoldenQA("How long is the warranty on the Nimbus 900?", ("nimbus_warranty.txt",)),
    GoldenQA("How do I connect the Nimbus 900 to Wi-Fi?", ("nimbus_setup.txt",)),
    GoldenQA("How long does it take to fully charge the Nimbus 900?", ("nimbus_battery.txt",)),
    GoldenQA("What should I do if the status LED blinks red?", ("nimbus_troubleshooting.txt",)),
    GoldenQA("What is the water resistance rating of the Nimbus 900?", ("nimbus_specs.txt",)),
    GoldenQA(
        "Does the Nimbus 900 upload my voice recordings to the cloud by default?",
        ("nimbus_privacy.txt",),
    ),
]


@dataclass
class EvalReport:
    hit_rate_at_3: float
    mrr: float
    per_question: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "hit_rate_at_3": round(self.hit_rate_at_3, 4),
            "mrr": round(self.mrr, 4),
            "n": len(self.per_question),
            "per_question": self.per_question,
        }


def evaluate(engine: RAGLite, golden: list[GoldenQA] | None = None, top_k: int = 3) -> EvalReport:
    """Run the golden set against ``engine`` and report hit-rate@k and MRR."""
    golden = golden or GOLDEN_SET
    hits = 0
    rr_sum = 0.0
    per_question: list[dict[str, object]] = []
    for item in golden:
        retrieved = engine.retrieve(item.question, top_k=max(top_k, 5))
        doc_ids = [engine.store.get(h.chunk_id).doc_id for h in retrieved]
        top = doc_ids[:top_k]
        hit = any(d in item.gold_doc_ids for d in top)
        rank = next((i + 1 for i, d in enumerate(doc_ids) if d in item.gold_doc_ids), None)
        rr = 1.0 / rank if rank else 0.0
        hits += int(hit)
        rr_sum += rr
        per_question.append(
            {
                "question": item.question,
                "gold": list(item.gold_doc_ids),
                "top_docs": top,
                "hit": hit,
                "reciprocal_rank": round(rr, 4),
            }
        )
    n = len(golden) or 1
    return EvalReport(hit_rate_at_3=hits / n, mrr=rr_sum / n, per_question=per_question)
