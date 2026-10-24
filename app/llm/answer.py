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
from app.privacy.redactor import Redactor, restore
from app.privacy.tripwire import sensitive_scope
from app.privacy.vault import load_entries
from app.retrieval.hybrid import hybrid_search

WHOLE_DOC_BUDGET_WORDS = 9000  # cap on summary text sent for "summarise the book" questions
PLACEHOLDER_NOTE = (
    "(Note: personal details in this document were replaced by placeholders such as [PERSON_1] or "
    "[PHONE_1]. A placeholder IS the value: answer with the placeholder itself, e.g. 'The phone number "
    "is [PHONE_1].' It is replaced with the real value before the user sees it.)"
)
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
    question: str = ""  # the question as the LLM will see it (redacted for sensitive documents)
    sensitive: bool = False
    restore_map: dict[str, str] = field(default_factory=dict)  # placeholder -> real value, stays local
    filename: str = ""


@dataclass
class Answer:
    answer: str  # what the user sees (placeholders restored to real values)
    route: str
    refused: bool
    citations: list[dict]  # excerpts the answer actually cites
    sources: list[dict]  # every excerpt that was shown to the model
    latency_ms: int
    best_similarity: float | None = None
    sensitive: bool = False
    answer_redacted: str | None = None  # the LLM's raw reply, with placeholders
    llm_payload: list[dict] | None = None  # exactly what was sent to the LLM (transparency panel)


def _load_doc(conn, doc_id: int) -> dict:
    doc = conn.execute(
        "SELECT id, filename, file_type, title, status, map_status, sensitive FROM documents WHERE id = %s", [doc_id]
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
    with get_pool().connection() as conn:
        doc = _load_doc(conn, doc_id)
    if not doc["sensitive"]:
        prepared = _prepare(doc, question)
        prepared.question, prepared.filename = question, doc["filename"]
        return prepared
    # Sensitive: hide known values (from the vault) and any new PII typed in the question, locally.
    redactor = Redactor(load_entries(doc_id), new_prefix="Q")
    redacted_question = redactor.redact(question, use_ner=True)
    prepared = _prepare(doc, redacted_question, source=f"document.{doc['file_type']}")
    # without this the model treated "[PHONE_1]" as "the phone number is not given" (seen in testing)
    prepared.note = (prepared.note + " " if prepared.note else "") + PLACEHOLDER_NOTE
    prepared.question, prepared.filename, prepared.sensitive = redacted_question, doc["filename"], True
    prepared.restore_map = {placeholder: value for placeholder, (_, value) in redactor.values.items()}
    return prepared


def _prepare(doc: dict, question: str, source: str | None = None) -> Prepared:
    settings = get_settings()
    doc_id = doc["id"]
    with get_pool().connection() as conn:
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
        labels = {
            r["ordinal"]: r["label"]
            for r in conn.execute(
                "SELECT ordinal, COALESCE(label, 'sec.' || ordinal) AS label FROM sections WHERE document_id = %s",
                [doc_id],
            ).fetchall()
        }
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

    source = source or doc["filename"]  # sensitive documents send a neutral name: filenames can hold PII
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
                    loc=labels[s["ordinal"]],
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
            where = ", ".join(labels.get(n, f"sec.{n}") for n in e["section_ids"][:30])
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


def _for_display(e: Excerpt, prepared: Prepared) -> dict:
    """Excerpt as the user sees it: real filename and (for sensitive docs) real values restored locally."""
    shown = asdict(e)
    if prepared.sensitive:
        shown.update(text=restore(e.text, prepared.restore_map), text_redacted=e.text, source=prepared.filename)
    return shown


def answer_question(doc_id: int, question: str) -> Answer:
    started = time.perf_counter()
    prepared = prepare(doc_id, question)
    payload = None
    if prepared.refuse:
        text, cited = NOT_FOUND, []  # no LLM call at all
    else:
        payload = answer_messages(prepared.question, prepared.excerpts, prepared.note)
        # tripwire: blocks the call if raw PII slipped through (patterns + this document's own vault values)
        known = {value: placeholder.rsplit("_", 1)[0] for placeholder, value in prepared.restore_map.items()}
        with sensitive_scope(prepared.sensitive, known_values=known):
            text = get_llm().chat(payload)
        cited = extract_citations(text, prepared.excerpts)
    refused = text.strip() == NOT_FOUND
    latency_ms = int((time.perf_counter() - started) * 1000)
    _log(doc_id, prepared.question, prepared, refused, latency_ms)  # the redacted question: no PII in logs
    return Answer(
        answer=restore(text, prepared.restore_map) if prepared.sensitive else text,
        route=prepared.route,
        refused=refused,
        citations=[_for_display(e, prepared) for e in cited],
        sources=[_for_display(e, prepared) for e in prepared.excerpts],
        latency_ms=latency_ms,
        best_similarity=prepared.best_similarity,
        sensitive=prepared.sensitive,
        answer_redacted=text if prepared.sensitive else None,
        llm_payload=payload,
    )
