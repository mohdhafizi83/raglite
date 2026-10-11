"""Recursive text splitter with configurable overlap.

The splitter breaks text into units by a cascade of separators (paragraphs,
lines, sentences, words), greedily packs units into chunks up to
``chunk_size`` characters, and carries trailing units from the previous chunk
into the next one (``overlap`` characters) so facts are never cut at a single
hard boundary.

Every chunk records its exact ``[start, end)`` character span in the source
document, which is what makes citations verifiable downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    """A piece of text plus its exact [start, end) span in the source."""

    text: str
    start: int
    end: int


def _split_keep_offsets(text: str, sep: str) -> list[Chunk]:
    """Split ``text`` by ``sep`` while preserving each piece's offsets."""
    out: list[Chunk] = []
    pos = 0
    for piece in text.split(sep):
        if piece:
            out.append(Chunk(piece, pos, pos + len(piece)))
        pos += len(piece) + len(sep)
    return out


def _to_units(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[Chunk]:
    """Split ``text`` into units of at most ``chunk_size`` chars, exact offsets.

    Separator cascade: at each level, only pieces that are still oversized get
    split by the next (finer) separator; pieces that fit are emitted whole.
    Anything still oversized after the cascade gets a hard character split.
    """
    units: list[Chunk] = []

    def expand(unit: Chunk, seps: tuple[str, ...]) -> None:
        if len(unit.text) <= chunk_size:
            units.append(unit)
            return
        if not seps:
            for i in range(0, len(unit.text), chunk_size):
                end = min(i + chunk_size, len(unit.text))
                units.append(Chunk(unit.text[i:end], unit.start + i, unit.start + end))
            return
        pieces = _split_keep_offsets(unit.text, seps[0])
        if len(pieces) <= 1:
            expand(unit, seps[1:])
            return
        for p in pieces:
            expand(Chunk(p.text, unit.start + p.start, unit.start + p.end), seps[1:])

    expand(Chunk(text, 0, len(text)), separators)
    units.sort(key=lambda c: c.start)
    return units


def split_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` chars.

    Overlap is expressed in characters: when a new chunk starts, trailing
    units from the previous chunk (up to ``overlap`` chars) are carried over.
    Chunk spans are exact offsets into ``text``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))

    if not text.strip():
        return []

    units = _to_units(text, chunk_size, separators)

    # Greedy packing with unit-level overlap carry-over. ``current_len``
    # tracks the emitted span length, i.e. unit lengths PLUS the separator
    # gaps between them, so chunks never exceed chunk_size.
    chunks: list[Chunk] = []
    current: list[Chunk] = []

    def span_len(units: list[Chunk]) -> int:
        return units[-1].end - units[0].start

    for unit in units:
        projected = unit.end - current[0].start if current else len(unit.text)
        if current and projected > chunk_size:
            chunks.append(_emit(text, current))
            tail: list[Chunk] = []
            for u in reversed(current):
                candidate = [u, *tail]
                # Keep the carry-over within the overlap budget AND make sure
                # the next unit still fits after the carry-over.
                if span_len(candidate) > overlap or unit.end - candidate[0].start > chunk_size:
                    break
                tail = candidate
            current = tail
        current.append(unit)
    if current:
        chunks.append(_emit(text, current))
    return chunks


def _emit(source: str, units: list[Chunk]) -> Chunk:
    start, end = units[0].start, units[-1].end
    return Chunk(source[start:end], start, end)
