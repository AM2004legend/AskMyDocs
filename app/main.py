"""FastAPI backend. The Streamlit UI (and anything else) talks to the app only through these routes."""

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import db
from app.config import get_settings
from app.docmap.summaries import build_document_map
from app.ingest.parsers import SUPPORTED_TYPES
from app.ingest.pipeline import (
    DOC_COLUMNS,
    UPLOAD_DIR,
    create_document,
    delete_document,
    find_by_hash,
    process_document,
    sha256_of,
)
from app.llm.answer import DocumentNotFound, DocumentNotReady, answer_question
from app.llm.client import LLMError, get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_schema()
    with db.get_pool().connection() as conn:
        # work that was running when the server stopped can't be resumed mid-way
        conn.execute(
            "UPDATE documents SET status = 'failed', progress = NULL,"
            " error = 'Processing was interrupted (server restarted). Delete and upload again.'"
            " WHERE status = 'processing'"
        )
        conn.execute(
            "UPDATE documents SET map_status = 'failed',"
            " progress = 'Document map was interrupted. Click Build map to resume.' WHERE map_status = 'building'"
        )
    yield
    db.close_pool()


app = FastAPI(title="AskMyDocs", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _get_doc(doc_id: int) -> dict:
    with db.get_pool().connection() as conn:
        doc = conn.execute(
            f"SELECT {DOC_COLUMNS},"
            " (SELECT count(*) FROM sections s WHERE s.document_id = d.id) AS n_sections,"
            " (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS n_chunks"
            " FROM documents d WHERE id = %s",
            [doc_id],
        ).fetchone()
    if doc is None:
        raise HTTPException(404, "Document not found.")
    return doc


@app.get("/health")
def health():
    try:
        with db.get_pool().connection() as conn:
            conn.execute("SELECT 1")
        database = True
    except Exception:
        database = False
    return {"status": "ok" if database else "degraded", "database": database, "llm_configured": get_llm().configured}


@app.post("/documents", status_code=202)
async def upload(file: UploadFile, background: BackgroundTasks):
    settings = get_settings()
    filename = Path(file.filename or "upload").name
    file_type = Path(filename).suffix.lower().lstrip(".")
    if file_type not in SUPPORTED_TYPES:
        raise HTTPException(415, f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_TYPES))}.")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
    if not data:
        raise HTTPException(400, "The file is empty.")

    sha = sha256_of(data)
    existing = find_by_hash(sha)
    if existing:  # same bytes already uploaded: reuse it, never re-embed or re-summarise
        return JSONResponse({**jsonable(existing), "duplicate": True}, status_code=200)

    doc = create_document(filename, file_type, sha)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{doc['id']}.{file_type}"
    path.write_bytes(data)
    background.add_task(process_document, doc["id"], path, file_type)
    return {**jsonable(doc), "duplicate": False}


def jsonable(row: dict) -> dict:
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}


@app.get("/documents")
def list_documents():
    with db.get_pool().connection() as conn:
        rows = conn.execute(f"SELECT {DOC_COLUMNS} FROM documents ORDER BY created_at DESC").fetchall()
    return [jsonable(r) for r in rows]


@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    return jsonable(_get_doc(doc_id))


@app.delete("/documents/{doc_id}", status_code=204)
def remove_document(doc_id: int):
    if not delete_document(doc_id):
        raise HTTPException(404, "Document not found.")


@app.post("/documents/{doc_id}/ask")
def ask(doc_id: int, body: AskRequest):
    try:
        return asdict(answer_question(doc_id, body.question.strip()))
    except DocumentNotFound:
        raise HTTPException(404, "Document not found.") from None
    except DocumentNotReady as exc:
        raise HTTPException(409, f"Document is not ready yet (status: {exc}).") from None
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/documents/{doc_id}/map")
def document_map(doc_id: int):
    doc = _get_doc(doc_id)
    with db.get_pool().connection() as conn:
        sections = conn.execute(
            "SELECT ordinal, title, page_start, page_end, n_words, summary FROM sections"
            " WHERE document_id = %s ORDER BY ordinal",
            [doc_id],
        ).fetchall()
        entities = conn.execute(
            "SELECT name, aliases, kind, description, section_ids FROM entities WHERE document_id = %s"
            " ORDER BY cardinality(section_ids) DESC, name",
            [doc_id],
        ).fetchall()
        summary = conn.execute("SELECT summary FROM doc_summary WHERE document_id = %s", [doc_id]).fetchone()
    return {
        "map_status": doc["map_status"],
        "progress": doc["progress"],
        "summary": summary["summary"] if summary else None,
        "sections": sections,
        "entities": entities,
    }


@app.post("/documents/{doc_id}/map", status_code=202)
def rebuild_map(doc_id: int, background: BackgroundTasks, force: bool = False):
    doc = _get_doc(doc_id)
    if doc["status"] != "ready":
        raise HTTPException(409, "Document is not ready yet.")
    if doc["map_status"] == "building":
        raise HTTPException(409, "The document map is already being built.")
    if not get_llm().configured:
        raise HTTPException(503, "No LLM is configured. Set LLM_API_KEY and LLM_MODEL in .env.")
    background.add_task(build_document_map, doc_id, force)
    return {"map_status": "building"}
