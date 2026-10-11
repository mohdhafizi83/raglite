"""LLM seam: a protocol plus an offline extractive stub.

``EchoLLM`` answers by selecting the sentence with the highest query-token
overlap from the retrieved passages. It is deliberately not clever — it is a
stand-in that makes the full pipeline (and its eval) runnable with zero
credentials. Swap in any client satisfying ``LLMClient`` for real generation;
nothing else changes.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from raglite.embedders import tokenize

# Minimal stopword list for the extractive answerer: these carry no topical
# signal in questions ("how long is the...", "what should I do if...") and
# would otherwise dominate IDF in a tiny candidate set.
_STOPWORDS = frozenset(
    "a an and are as at be but by do does did for from has have how i if in "
    "is it its me my of on or our should the their them they this to was we "
    "what when where which who will with you your not no can".split()
)


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a question + context passages into an answer."""

    def answer(self, question: str, contexts: list[str]) -> str:
        ...


class EchoLLM:
    """Offline extractive answerer: best-matching sentence from the context.

    Each candidate sentence is scored by the IDF-weighted overlap with the
    question (``df`` counts candidate sentences containing the token), so
    generic tokens ("the", "is") wash out and distinctive ones ("warranty")
    dominate. The sum is divided by ``sqrt(len(sentence))`` — a cosine-style
    length penalty so one bloated sentence cannot win on token count alone.
    Heading-like fragments (fewer than 8 tokens, e.g. the manual's title
    lines like "Nimbus 900 Privacy and Data Handling.") are skipped as
    answer candidates: they echo the topic without stating a fact. A small
    decay favours earlier (higher-ranked) passages. The winning sentence is
    returned verbatim, so answers are always grounded in retrieved text and
    trivially checkable in the demo.
    """

    def answer(self, question: str, contexts: list[str]) -> str:
        q_tokens = set(tokenize(question)) - _STOPWORDS
        if not q_tokens:
            return contexts[0].strip() if contexts else ""
        sentences_per_ctx = [_sentences(ctx) for ctx in contexts]
        all_sentences = [s for sents in sentences_per_ctx for s in sents]
        if not all_sentences:
            return contexts[0].strip() if contexts else ""
        n = len(all_sentences)
        df: dict[str, int] = {}
        for s in all_sentences:
            for tok in q_tokens & set(tokenize(s)):
                df[tok] = df.get(tok, 0) + 1
        idf = {t: math.log(1.0 + n / (1.0 + d)) for t, d in df.items()}
        best_score = 0.0
        best: str | None = None
        for weight, sents in enumerate(sentences_per_ctx):
            decay = 1.0 / (1.0 + 0.1 * weight)
            for sentence in sents:
                toks = tokenize(sentence)
                if len(toks) < 8:  # heading-like fragment, not an answer
                    continue
                matched = q_tokens & set(toks)
                if not matched:
                    continue
                score = decay * sum(idf[t] for t in matched) / math.sqrt(max(len(toks), 1))
                if score > best_score:
                    best_score, best = score, sentence
        if best is None:
            return contexts[0].strip() if contexts else ""
        return best.strip()


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts: list[str] = []
        buf = ""
        for ch in para:
            buf += ch
            if ch == ".":
                parts.append(buf.strip())
                buf = ""
        if buf.strip():
            parts.append(buf.strip())
        out.extend(parts)
    return out
