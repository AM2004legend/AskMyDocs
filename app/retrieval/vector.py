"""Semantic search: nearest chunks by cosine distance (pgvector `<=>`), scoped to one document."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    chunk_id: int
    text: str
    page: int | None
    section_ordinal: int | None
    section_title: str | None
    section_label: str | None = None  # "ch.16" / "sec.2"
    similarity: float | None = None  # cosine similarity, only from vector search
    score: float = 0.0  # fused RRF score

    @property
    def loc(self) -> str:
        if self.page is not None:
            return f"p.{self.page}"
        if self.section_label:
            return self.section_label
        if self.section_ordinal is not None:
            return f"sec.{self.section_ordinal}"
        return "doc"


SELECT = (
    "SELECT c.id AS chunk_id, c.text, c.page, s.ordinal AS section_ordinal, s.title AS section_title,"
    " s.label AS section_label"
)


def _section_filter(section_ordinals: list[int] | None) -> tuple[str, list]:
    if not section_ordinals:
        return "", []
    return " AND s.ordinal = ANY(%s)", [section_ordinals]


def vector_search(
    conn, doc_id: int, query_vector: np.ndarray, k: int, section_ordinals: list[int] | None = None
) -> list[Hit]:
    where, params = _section_filter(section_ordinals)
    rows = conn.execute(
        f"{SELECT}, 1 - (c.embedding <=> %s) AS similarity"
        " FROM chunks c LEFT JOIN sections s ON s.id = c.section_id"
        f" WHERE c.document_id = %s{where}"
        " ORDER BY c.embedding <=> %s LIMIT %s",
        [query_vector, doc_id, *params, query_vector, k],
    ).fetchall()
    return [Hit(**row) for row in rows]
