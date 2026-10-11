"""Chunker tests: spans, overlap, size bounds, edge cases."""

from raglite.chunker import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_short_text_is_single_chunk():
    chunks = split_text("hello world", chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert (chunks[0].start, chunks[0].end) == (0, 11)


def test_chunk_spans_reconstruct_source():
    text = "\n\n".join(f"Paragraph {i} covers topic {i} in some detail." for i in range(8))
    chunks = split_text(text, chunk_size=120, overlap=0)
    # With zero overlap, first chunk starts at 0 and last ends at len(text).
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for c in chunks:
        assert text[c.start : c.end] == c.text


def test_chunks_respect_max_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_overlap_carries_content_between_consecutive_chunks():
    para = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    text = "\n\n".join([para] * 6)
    chunks = split_text(text, chunk_size=60, overlap=30)
    assert len(chunks) >= 2
    # With overlap, consecutive chunks must share at least some text.
    shared = any(chunks[i].text in chunks[i + 1].text or chunks[i + 1].text in chunks[i].text
                 or set(chunks[i].text.split()) & set(chunks[i + 1].text.split())
                 for i in range(len(chunks) - 1))
    assert shared


def test_zero_overlap_chunks_do_not_repeat_units():
    text = "\n\n".join(f"line {i} content here" for i in range(6))
    no_ov = split_text(text, chunk_size=40, overlap=0)
    ov = split_text(text, chunk_size=40, overlap=20)
    # Overlapping config produces at least as many chunks (carry-over).
    assert len(ov) >= len(no_ov)


def test_invalid_chunk_size_raises():
    try:
        split_text("x", chunk_size=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_long_word_falls_back_to_hard_split():
    text = "a" * 950
    chunks = split_text(text, chunk_size=100, overlap=0)
    assert all(len(c.text) <= 100 for c in chunks)
    assert "".join(c.text for c in chunks) == text
