"""Build the document map: section summaries + entities (map step), whole-document summary (reduce step).

Cost/limit tricks:
- Several short chapters are packed into ONE LLM call (up to map_batch_words), and the same
  call returns both the summary and the entities, so a 60-chapter novel needs ~10-15 calls,
  not 120.
- Results are saved after every call, so a crash or a daily-quota stop resumes where it left off.
- If a request is too large for the model, the batch is split in half and retried.
- Calls are paced (llm_min_interval_seconds) to stay under free-tier requests/minute.
"""

import json
import logging
import re
import threading

from psycopg.types.json import Jsonb

from app.config import get_settings
from app.db import get_pool
from app.docmap.entities import Entity, apply_groups, clean_raw, merge_entities
from app.llm import prompts
from app.llm.client import LLMClient, LLMError, LLMRequestTooLarge, get_llm

log = logging.getLogger(__name__)
_map_lock = threading.Lock()  # one document map at a time, to respect rate limits
MAX_CONSOLIDATE = 300


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data


def call_json(llm: LLMClient, messages: list[dict]) -> dict:
    reply = llm.chat(messages, paced=True)
    try:
        return parse_json(reply)
    except (ValueError, json.JSONDecodeError):
        retry = messages + [
            {"role": "assistant", "content": reply[:2000]},
            {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
        ]
        try:
            return parse_json(llm.chat(retry, paced=True))
        except (ValueError, json.JSONDecodeError) as exc:
            raise LLMError("The AI returned malformed data while building the document map.") from exc


def pack_batches(sections: list[dict], max_words: int) -> list[list[dict]]:
    """Group consecutive sections so each batch has at most max_words (a huge section goes alone)."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    words = 0
    for s in sections:
        if current and words + s["n_words"] > max_words:
            batches.append(current)
            current, words = [], 0
        current.append(s)
        words += s["n_words"]
    if current:
        batches.append(current)
    return batches


def _entities_from(item: dict) -> list[dict]:
    raw = item.get("entities") if isinstance(item, dict) else None
    return [e for e in (clean_raw(x) for x in (raw or [])) if e]


def summarise_long_section(llm: LLMClient, section: dict, part_words: int) -> tuple[str, list[dict]]:
    settings = get_settings()
    words = section["text"].split()
    parts = [" ".join(words[i : i + part_words]) for i in range(0, len(words), part_words)]
    summaries, entities = [], []
    for n, part in enumerate(parts, start=1):
        title = f"{section['title']} (part {n} of {len(parts)})"
        try:
            data = call_json(llm, prompts.map_messages([(1, title, part)], settings.summary_words))
        except LLMRequestTooLarge:
            if part_words <= 500:
                raise
            return summarise_long_section(llm, section, part_words // 2)
        items = data.get("sections") or []
        item = items[0] if items and isinstance(items[0], dict) else {}
        summaries.append(str(item.get("summary") or "").strip())
        entities.extend(_entities_from(item))
    summaries = [s for s in summaries if s]
    if len(summaries) == 1:
        return summaries[0], entities
    combined = llm.chat(prompts.combine_messages(section["title"], summaries, settings.summary_words), paced=True)
    return combined, entities


def summarise_batch(llm: LLMClient, batch: list[dict]) -> dict[int, tuple[str, list[dict]]]:
    """Return {section ordinal: (summary, entities)} for every section in the batch."""
    settings = get_settings()
    if len(batch) == 1 and batch[0]["n_words"] > settings.map_batch_words:
        return {batch[0]["ordinal"]: summarise_long_section(llm, batch[0], settings.map_batch_words)}
    try:
        data = call_json(
            llm,
            prompts.map_messages([(s["ordinal"], s["title"], s["text"]) for s in batch], settings.summary_words),
        )
    except LLMRequestTooLarge:
        if len(batch) == 1:
            s = batch[0]
            return {s["ordinal"]: summarise_long_section(llm, s, max(s["n_words"] // 2, 500))}
        half = len(batch) // 2
        return summarise_batch(llm, batch[:half]) | summarise_batch(llm, batch[half:])

    results: dict[int, tuple[str, list[dict]]] = {}
    for item in data.get("sections") or []:
        if not isinstance(item, dict):
            continue
        try:
            ordinal = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        summary = str(item.get("summary") or "").strip()
        if summary and any(s["ordinal"] == ordinal for s in batch):
            results[ordinal] = (summary, _entities_from(item))
    missing = [s for s in batch if s["ordinal"] not in results]
    if missing and len(batch) > 1:  # the model skipped some: ask again for just those
        for s in missing:
            results |= summarise_batch(llm, [s])
    elif missing:
        raise LLMError(f"The AI did not return a summary for section '{batch[0]['title']}'.")
    return results


def reduce_summaries(llm: LLMClient, title: str, items: list[tuple[str, str]], max_words: int) -> str:
    total = sum(len(s.split()) for _, s in items)
    if total <= max_words or len(items) == 1:
        return llm.chat(prompts.reduce_messages(title, items), paced=True)
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    words = 0
    for item in items:
        n = len(item[1].split())
        if current and words + n > max_words:
            groups.append(current)
            current, words = [], 0
        current.append(item)
        words += n
    groups.append(current)
    partial = [(f"{g[0][0]} to {g[-1][0]}", reduce_summaries(llm, title, g, max_words)) for g in groups]
    return reduce_summaries(llm, title, partial, max_words)


def consolidate(llm: LLMClient, entities: list[Entity]) -> list[Entity]:
    """One extra LLM call to catch duplicates the rules missed (e.g. 'Lizzy' vs 'Elizabeth Bennet')."""
    if len(entities) < 2:
        return entities
    head, tail = entities[:MAX_CONSOLIDATE], entities[MAX_CONSOLIDATE:]
    lines = [
        f"{i}. {e.name}"
        + (f" (also: {', '.join(sorted(e.aliases)[:5])})" if e.aliases else "")
        + f" - {e.kind} - {e.description[:100]}"
        for i, e in enumerate(head, start=1)
    ]
    try:
        data = call_json(llm, prompts.consolidate_messages(lines))
        return apply_groups(head, data.get("groups") or []) + tail
    except LLMError as exc:
        log.warning("entity consolidation skipped: %s", exc)
        return entities


def _set(doc_id: int, **fields) -> None:
    cols = ", ".join(f"{k} = %s" for k in fields)
    with get_pool().connection() as conn:
        conn.execute(f"UPDATE documents SET {cols} WHERE id = %s", [*fields.values(), doc_id])


def _build(doc_id: int, llm: LLMClient, force: bool) -> None:
    settings = get_settings()
    pool = get_pool()
    with pool.connection() as conn:
        doc = conn.execute("SELECT title, filename FROM documents WHERE id = %s", [doc_id]).fetchone()
        if doc is None:
            return
        if force:
            conn.execute("UPDATE sections SET summary = NULL, entities_raw = NULL WHERE document_id = %s", [doc_id])
            conn.execute("DELETE FROM doc_summary WHERE document_id = %s", [doc_id])
        sections = conn.execute(
            "SELECT id, ordinal, title, text, n_words, summary FROM sections WHERE document_id = %s ORDER BY ordinal",
            [doc_id],
        ).fetchall()
    _set(doc_id, map_status="building")

    todo = [s for s in sections if not s["summary"]]
    for batch in pack_batches(todo, settings.map_batch_words):
        _set(doc_id, progress=f"Summarising sections {batch[0]['ordinal']}-{batch[-1]['ordinal']} of {len(sections)}")
        results = summarise_batch(llm, batch)
        with pool.connection() as conn:  # save after every call so we can resume
            for s in batch:
                summary, entities = results[s["ordinal"]]
                conn.execute(
                    "UPDATE sections SET summary = %s, entities_raw = %s WHERE id = %s",
                    [summary, Jsonb(entities), s["id"]],
                )

    _set(doc_id, progress="Building the character / entity index")
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT ordinal, title, summary, entities_raw FROM sections WHERE document_id = %s ORDER BY ordinal",
            [doc_id],
        ).fetchall()
    raw = [(r["ordinal"], e) for r in rows for e in (r["entities_raw"] or [])]
    entities = consolidate(llm, merge_entities(raw))
    with pool.connection() as conn:
        conn.execute("DELETE FROM entities WHERE document_id = %s", [doc_id])
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO entities (document_id, name, aliases, kind, description, section_ids)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [(doc_id, e.name, sorted(e.aliases), e.kind, e.description, sorted(e.section_ids)) for e in entities],
            )

    _set(doc_id, progress="Writing the whole-document summary")
    overview = reduce_summaries(
        llm, doc["title"] or doc["filename"], [(r["title"], r["summary"]) for r in rows], settings.map_batch_words
    )
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO doc_summary (document_id, summary) VALUES (%s, %s)"
            " ON CONFLICT (document_id) DO UPDATE SET summary = EXCLUDED.summary",
            [doc_id, overview],
        )
    _set(doc_id, map_status="ready", progress=None)


def build_document_map(doc_id: int, force: bool = False) -> None:
    llm = get_llm()
    if not llm.configured:
        _set(doc_id, map_status="pending", progress="Document map skipped: no LLM key configured")
        return
    with _map_lock:
        try:
            _build(doc_id, llm, force)
        except LLMError as exc:
            _set(doc_id, map_status="failed", progress=f"Document map stopped: {exc} Click 'Build map' to resume.")
        except Exception:
            log.exception("document map failed for document %s", doc_id)
            _set(doc_id, map_status="failed", progress="Document map failed unexpectedly. Click 'Build map' to retry.")
