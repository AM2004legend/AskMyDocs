"""Hybrid search = vector search + keyword search merged with Reciprocal Rank Fusion (RRF).

RRF score of a chunk = sum over result lists of 1 / (k + rank), rank starting at 1.
It only uses positions, so the two lists' very different score scales don't matter, and a
chunk that ranks well in BOTH lists beats one that ranks first in only one.
"""

from dataclasses import dataclass

from app.config import get_settings
from app.db import get_pool
from app.embeddings import embed_one
from app.retrieval.keyword import keyword_search
from app.retrieval.vector import Hit, vector_search


def rrf_merge(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class SearchResult:
    hits: list[Hit]
    best_similarity: float  # best cosine similarity from the vector side; used for refusal


def hybrid_search(
    doc_id: int,
    question: str,
    top_k: int | None = None,
    section_ordinals: list[int] | None = None,
    extra_keywords: str = "",
) -> SearchResult:
    s = get_settings()
    top_k = top_k or s.final_top_k
    query_vector = embed_one(question)
    with get_pool().connection() as conn:
        vector_hits = vector_search(conn, doc_id, query_vector, s.vector_top_k, section_ordinals)
        keyword_hits = keyword_search(
            conn, doc_id, f"{question} {extra_keywords}".strip(), s.keyword_top_k, section_ordinals
        )

    by_id: dict[int, Hit] = {h.chunk_id: h for h in keyword_hits}
    by_id.update({h.chunk_id: h for h in vector_hits})  # vector hits carry the similarity
    fused = rrf_merge([[h.chunk_id for h in vector_hits], [h.chunk_id for h in keyword_hits]], k=s.rrf_k)
    hits = []
    for chunk_id, score in fused[:top_k]:
        hit = by_id[chunk_id]
        hit.score = score
        hits.append(hit)
    best = max((h.similarity for h in vector_hits if h.similarity is not None), default=0.0)
    return SearchResult(hits=hits, best_similarity=float(best))
