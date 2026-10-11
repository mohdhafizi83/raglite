"""Console entry point: ``raglite demo``, ``raglite eval``, ``raglite serve``.

Everything runs offline — no API keys, no downloads.
"""

from __future__ import annotations

import argparse
import sys

from raglite import __version__
from raglite.eval import evaluate
from raglite.pipeline import RAGLite
from raglite.sample_corpus import SAMPLE_DOCS, seed

DEMO_QUESTIONS = [
    "How long is the warranty on the Nimbus 900?",
    "How do I connect the Nimbus 900 to Wi-Fi?",
    "What should I do if the status LED blinks red?",
]


def _cmd_demo(_: argparse.Namespace) -> int:
    engine = RAGLite()
    n = seed(engine)
    print(f"raglite {__version__} — offline demo (no API keys)")
    print(f"seeded {len(SAMPLE_DOCS)} sample docs ({n} chunks) from the Nimbus 900 manual\n")
    for q in DEMO_QUESTIONS:
        result = engine.query(q, top_k=3)
        print(f"Q: {q}")
        print(f"A: {result.answer}")
        for c in result.citations:
            print(f"   [{c.doc_id} chars {c.start}-{c.end}] rrf={c.score:.4f}")
        print()
    report = evaluate(engine)
    print(f"eval on {len(report.per_question)} golden questions: "
          f"hit-rate@3={report.hit_rate_at_3:.2f}  MRR={report.mrr:.2f}")
    engine.close()
    return 0


def _cmd_eval(_: argparse.Namespace) -> int:
    engine = RAGLite()
    seed(engine)
    report = evaluate(engine)
    for item in report.per_question:
        mark = "PASS" if item["hit"] else "FAIL"
        print(f"[{mark}] {item['question']}")
        print(f"       gold={item['gold']} top={item['top_docs']}")
    print(f"\nhit-rate@3={report.hit_rate_at_3:.2f}  MRR={report.mrr:.2f}")
    engine.close()
    return 0 if report.hit_rate_at_3 >= 0.99 else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from raglite.app import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raglite", description="Offline hybrid RAG demo service.")
    parser.add_argument("--version", action="version", version=f"raglite {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="seed sample corpus, answer 3 questions, print mini-eval")
    sub.add_parser("eval", help="run the golden QA set and report hit-rate@3 / MRR")
    serve = sub.add_parser("serve", help="run the FastAPI app with uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "eval":
        return _cmd_eval(args)
    return _cmd_serve(args)


if __name__ == "__main__":
    sys.exit(main())
