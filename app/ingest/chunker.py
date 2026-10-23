"""Sentence-aware chunking with overlap.

Chunks are packed from whole sentences up to `chunk_words`, and the last few sentences
(up to `overlap_words`) are repeated at the start of the next chunk so a fact that sits
on a boundary is still fully inside at least one chunk. Each chunk remembers the page
its first sentence came from, for citations.
"""

import re
from dataclasses import dataclass

from app.ingest.parsers import Block

# split after . ! ? (optionally followed by a closing quote/bracket) when the next thing starts a sentence,
# but not after common abbreviations ("Mr. Darcy" is one sentence)
SENTENCE_END_RE = re.compile(
    r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bNo\.)(?<!\bvs\.)(?<!\bJr\.)(?<!\bSr\.)"
    r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])"
)


@dataclass
class Chunk:
    text: str
    page: int | None


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END_RE.split(text) if s.strip()]


def _page_pieces(blocks: list[Block]) -> list[tuple[int | None, str]]:
    """Join consecutive blocks on the same page into one string (PDF lines -> page text)."""
    pieces: list[tuple[int | None, list[str]]] = []
    for block in blocks:
        if pieces and pieces[-1][0] == block.page:
            pieces[-1][1].append(block.text)
        else:
            pieces.append((block.page, [block.text]))
    return [(page, " ".join(texts)) for page, texts in pieces]


def _sentences_with_pages(blocks: list[Block], max_words: int) -> list[tuple[int | None, list[str]]]:
    out = []
    for page, text in _page_pieces(blocks):
        for sentence in split_sentences(text):
            words = sentence.split()
            # a "sentence" longer than a chunk (tables, OCR soup) gets cut into word windows
            for start in range(0, len(words), max_words):
                out.append((page, words[start : start + max_words]))
    return out


def chunk_blocks(blocks: list[Block], chunk_words: int, overlap_words: int) -> list[Chunk]:
    sentences = _sentences_with_pages(blocks, chunk_words)
    chunks: list[Chunk] = []
    current: list[tuple[int | None, list[str]]] = []
    n_words = 0
    has_new = False  # does `current` contain anything not already emitted?

    def emit():
        chunks.append(Chunk(" ".join(" ".join(w) for _, w in current), current[0][0]))

    for page, words in sentences:
        if current and n_words + len(words) > chunk_words:
            emit()
            # keep trailing sentences as overlap, as long as they fit in overlap_words
            keep: list[tuple[int | None, list[str]]] = []
            kept_words = 0
            for item in reversed(current):
                if kept_words + len(item[1]) > overlap_words:
                    break
                keep.insert(0, item)
                kept_words += len(item[1])
            current, n_words, has_new = keep, kept_words, False
        current.append((page, words))
        n_words += len(words)
        has_new = True
    if current and has_new:
        emit()
    return chunks
