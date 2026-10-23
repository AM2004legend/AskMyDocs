"""Keyword search with PostgreSQL full-text search (tsvector + GIN index), scoped to one document.

Catches what embeddings miss: exact names, numbers, rare terms ("clause 14.2", "IFSC").
Words are OR-ed, not AND-ed: a natural question has many words that never all appear in one chunk.
English stop words and stemming ("running" -> "run") are handled by Postgres.
"""

import re

from app.retrieval.vector import SELECT, Hit, _section_filter


def to_or_query(text: str) -> str | None:
    # only letters/digits reach to_tsquery, so the user can't inject tsquery syntax (& | ! :)
    words = list(dict.fromkeys(w.lower() for w in re.findall(r"[^\W_]+", text)))
    return " | ".join(words) if words else None


def keyword_search(conn, doc_id: int, text: str, k: int, section_ordinals: list[int] | None = None) -> list[Hit]:
    query = to_or_query(text)
    if not query:
        return []
    where, params = _section_filter(section_ordinals)
    rows = conn.execute(
        f"{SELECT} FROM chunks c LEFT JOIN sections s ON s.id = c.section_id,"
        " to_tsquery('english', %s) q"
        f" WHERE c.document_id = %s AND c.tsv @@ q{where}"
        " ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT %s",
        [query, doc_id, *params, k],
    ).fetchall()
    return [Hit(**row) for row in rows]
