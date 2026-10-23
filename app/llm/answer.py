"""Answer a question about one document: route -> retrieve -> refuse or ask the LLM -> citations.

`prepare()` does everything except the LLM call, so the eval script can measure retrieval
and refusal without spending API quota.
"""

import re
import time
from dataclasses import asdict, dataclass, field

from app.config import get_settings
from app.db import get_pool
from app.docmap.router import ENTITY, SPECIFIC, WHOLE_DOCUMENT, route_question
from app.llm.client import get_llm
from app.llm.prompts import NOT_FOUND, Excerpt, answer_messages
from app.retrieval.hybrid import hybrid_search

WHOLE_DOC_BUDGET_WORDS = 9000  # cap on summary text sent for "summarise the book" questions
MAX_LIST_ENTITIES = 25


class DocumentNotFound(Exception):
    pass


class DocumentNotReady(Exception):
    pass


@dataclass
class Prepared:
    route: str
    excerpts: list[Excerpt]
    refuse: bool
    best_similarity: float | None = None
    note: str = ""
    retrieved_chunk_ids: list[int] = field(default_factory=list)


@dataclass
class Answer:
    answer: str
    route: str
    refused: bool
    citations: list[dict]  # excerpts the answer actually cites
    sources: list[dict]  # every excerpt that was shown to the model
    latency_ms: int
    best_similarity: float | None = None


def _load_doc(conn, doc_id: int) -> dict:
    doc = conn.execute(
        "SELECT id, filename, title, status, map_status FROM documents WHERE id = %s", [doc_id]
    ).fetchone()
    if doc is None:
        raise DocumentNotFound(doc_id)
    if doc["status"] != "ready":
        raise DocumentNotReady(doc["status"])
    return doc


def _chunk_excerpts(hits, source: str, start_id: int) -> list[Excerpt]:
    return [
        Excerpt(id=start_id + i, source=source, loc=h.loc, text=h.text, kind="chunk", chunk_id=h.chunk_id)
        for i, h in enumerate(hits)
    ]


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def prepare(doc_id: int, question: str) -> Prepared:
    settings = get_settings()
    with get_pool().connection() as conn:
        doc = _load_doc(conn, doc_id)
        map_ready = doc["map_status"] == "ready"
        entities = (
            conn.execute(
                "SELECT id, name, aliases, kind, description, section_ids FROM entities WHERE document_id = %s"
                " ORDER BY cardinality(section_ids) DESC, name",
                [doc_id],
            ).fetchall()
            if map_ready
            else []
        )
        route = route_question(question, {e["id"]: {e["name"], *e["aliases"]} for e in entities})
        section_ordinals = None
        if route.section_number is not None:
            rows = conn.execute(
                "SELECT ordinal FROM sections WHERE document_id = %s AND number = %s", [doc_id, route.section_number]
            ).fetchall()
            section_ordinals = [r["ordinal"] for r in rows] or None
        summaries = []
        overview = None
        if map_ready and route.kind == WHOLE_DOCUMENT:
            where = " AND ordinal = ANY(%s)" if section_ordinals else ""
            summaries = conn.execute(
                "SELECT ordinal, title, summary FROM sections WHERE document_id = %s AND summary IS NOT NULL"
                f"{where} ORDER BY ordinal",
                [doc_id, section_ordinals] if section_ordinals else [doc_id],
            ).fetchall()
            if not section_ordinals:
                row = conn.execute("SELECT summary FROM doc_summary WHERE document_id = %s", [doc_id]).fetchone()
                overview = row["summary"] if row else None

    source = doc["filename"]
    note = ""

    # ----- whole-document: answer from the document map -----
    if route.kind == WHOLE_DOCUMENT and summaries:
        excerpts: list[Excerpt] = []
        if overview:
            excerpts.append(Excerpt(id=1, source=source, loc="overview", text=overview, kind="doc_summary"))
        per_section = max(40, WHOLE_DOC_BUDGET_WORDS // max(len(summaries), 1))
        for s in summaries:
            excerpts.append(
                Excerpt(
                    id=len(excerpts) + 1,
                    source=source,
                    loc=f"ch.{s['ordinal']}",
                    text=f"[{s['title']}] {_truncate_words(s['summary'], per_section)}",
                    kind="section_summary",
                )
            )
        chunk_ids: list[int] = []
        if section_ordinals:  # "summarise chapter 4": add real text from that chapter too
            result = hybrid_search(doc_id, question, top_k=4, section_ordinals=section_ordinals)
            excerpts += _chunk_excerpts(result.hits, source, len(excerpts) + 1)
            chunk_ids = [h.chunk_id for h in result.hits]
        return Prepared(WHOLE_DOCUMENT, excerpts, refuse=False, retrieved_chunk_ids=chunk_ids)
    if route.kind == WHOLE_DOCUMENT:
        note = "(Note: the document map is not ready yet, so only individual excerpts are available.)"

    # ----- entity: entity records + chunks that mention them -----
    if route.kind == ENTITY and entities:
        chosen = [e for e in entities if e["id"] in route.entity_ids] or entities[:MAX_LIST_ENTITIES]
        lines = []
        for e in chosen:
            aliases = f" Also called: {', '.join(e['aliases'])}." if e["aliases"] else ""
            where = ", ".join(f"ch.{n}" for n in e["section_ids"][:30])
            lines.append(f"{e['name']} ({e['kind']}).{aliases} {e['description']} Appears in: {where}.")
        excerpts = [Excerpt(id=1, source=source, loc="entity index", text="\n".join(lines), kind="entity")]
        keywords = " ".join(n for e in chosen[:5] for n in [e["name"], *e["aliases"]]) if route.entity_ids else ""
        result = hybrid_search(doc_id, question, section_ordinals=section_ordinals, extra_keywords=keywords)
        excerpts += _chunk_excerpts(result.hits, source, 2)
        return Prepared(
            ENTITY,
            excerpts,
            refuse=False,
            best_similarity=result.best_similarity,
            retrieved_chunk_ids=[h.chunk_id for h in result.hits],
        )

    # ----- specific: hybrid search over chunks -----
    result = hybrid_search(doc_id, question, section_ordinals=section_ordinals)
    refuse = not result.hits or result.best_similarity < settings.min_vector_similarity
    return Prepared(
        SPECIFIC,
        _chunk_excerpts(result.hits, source, 1),
        refuse=refuse,
        best_similarity=result.best_similarity,
        note=note,
        retrieved_chunk_ids=[h.chunk_id for h in result.hits],
    )


CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def _norm_loc(loc: str) -> str:
    return re.sub(r"\s+", "", loc.lower())


def extract_citations(answer: str, excerpts: list[Excerpt]) -> list[Excerpt]:
    """Map [source, loc] markers in the answer back to the excerpts they refer to."""
    by_loc: dict[str, list[Excerpt]] = {}
    for e in excerpts:
        by_loc.setdefault(_norm_loc(e.loc), []).append(e)
    cited: dict[int, Excerpt] = {}
    for bracket in CITATION_RE.findall(answer):
        for part in bracket.split(";"):
            if "," not in part:
                continue
            loc = _norm_loc(part.rsplit(",", 1)[1])
            for e in by_loc.get(loc, []):
                cited.setdefault(e.id, e)
    return list(cited.values())


def _log(doc_id: int, question: str, prepared: Prepared, refused: bool, latency_ms: int) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            "INSERT INTO query_log (document_id, question, route, retrieved_chunk_ids, refused, latency_ms)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            [doc_id, question, prepared.route, prepared.retrieved_chunk_ids, refused, latency_ms],
        )


def answer_question(doc_id: int, question: str) -> Answer:
    started = time.perf_counter()
    prepared = prepare(doc_id, question)
    if prepared.refuse:
        text, cited = NOT_FOUND, []  # no LLM call at all
    else:
        text = get_llm().chat(answer_messages(question, prepared.excerpts, prepared.note))
        cited = extract_citations(text, prepared.excerpts)
    refused = text.strip() == NOT_FOUND
    latency_ms = int((time.perf_counter() - started) * 1000)
    _log(doc_id, question, prepared, refused, latency_ms)
    return Answer(
        answer=text,
        route=prepared.route,
        refused=refused,
        citations=[asdict(e) for e in cited],
        sources=[asdict(e) for e in prepared.excerpts],
        latency_ms=latency_ms,
        best_similarity=prepared.best_similarity,
    )
