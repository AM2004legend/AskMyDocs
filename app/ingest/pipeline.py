"""Upload -> parse (+OCR) -> sections -> chunks -> local embeddings -> Postgres, then the document map.

Runs in a background thread so the upload request returns immediately; the UI polls
documents.status / documents.progress to show what is happening.
"""

import hashlib
import logging
import threading
import time
from pathlib import Path

from app.config import get_settings
from app.db import get_pool
from app.embeddings import embed
from app.ingest.chunker import chunk_blocks
from app.ingest.parsers import ParseError, parse_file
from app.ingest.sections import section_labels, split_sections
from app.privacy.redactor import Redactor
from app.privacy.vault import save_entries

log = logging.getLogger(__name__)
UPLOAD_DIR = Path("data") / "uploads"
EMBED_BATCH = 128
_ingest_lock = threading.Lock()  # one document at a time keeps CPU/RAM use predictable on a laptop

DOC_COLUMNS = (
    "id, filename, file_type, title, n_pages, status, map_status, progress, error, ocr_pages, sensitive, created_at"
)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_by_hash(sha: str) -> dict | None:
    with get_pool().connection() as conn:
        return conn.execute(f"SELECT {DOC_COLUMNS} FROM documents WHERE content_sha256 = %s", [sha]).fetchone()


def create_document(filename: str, file_type: str, sha: str, sensitive: bool = False) -> dict:
    # a sensitive file's name may itself contain a name ("Rahul_offer_letter.pdf"), so don't use it as the title
    title = "Sensitive document" if sensitive else Path(filename).stem
    with get_pool().connection() as conn:
        return conn.execute(
            "INSERT INTO documents (filename, file_type, title, content_sha256, sensitive, progress)"
            f" VALUES (%s, %s, %s, %s, %s, 'Queued') RETURNING {DOC_COLUMNS}",
            [filename, file_type, title, sha, sensitive],
        ).fetchone()


def update_document(doc_id: int, **fields) -> None:
    cols = ", ".join(f"{k} = %s" for k in fields)
    with get_pool().connection() as conn:
        conn.execute(f"UPDATE documents SET {cols} WHERE id = %s", [*fields.values(), doc_id])


class _Progress:
    """Write progress messages to the DB at most twice a second."""

    def __init__(self, doc_id: int):
        self.doc_id, self.last = doc_id, 0.0

    def __call__(self, message: str, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last > 0.5:
            self.last = now
            update_document(self.doc_id, progress=message)


def _redact(sections, chunk_rows, title: str) -> tuple[list[str], list[str], str, list]:
    """Sensitive mode: replace PII everywhere before anything is embedded, stored or sent.

    Pass 1 runs all detectors (incl. NER) over every section to learn the full value -> placeholder
    map; pass 2 rewrites the texts, so a name first found in chapter 9 is also hidden in chapter 1.
    """
    redactor = Redactor()
    for s in sections:
        redactor.learn(s.text)
    texts = [redactor.redact(s.text, use_ner=False) for s in sections]
    titles = [redactor.redact(s.title, use_ner=False) for s in sections]
    for _, chunk in chunk_rows:  # chunks join PDF lines, so re-run the regex detectors on them too
        chunk.text = redactor.redact(chunk.text, use_ner=False)
    return texts, titles, redactor.redact(title, use_ner=False), redactor.entries()


def _ingest(doc_id: int, path: Path, file_type: str) -> None:
    settings = get_settings()
    progress = _Progress(doc_id)
    with get_pool().connection() as conn:
        sensitive = conn.execute("SELECT sensitive FROM documents WHERE id = %s", [doc_id]).fetchone()["sensitive"]
    parsed = parse_file(path, file_type, progress)
    sections = split_sections(parsed.blocks)

    chunk_rows = []  # (section index, chunk)
    for index, section in enumerate(sections):
        for chunk in chunk_blocks(section.blocks, settings.chunk_words, settings.chunk_overlap_words):
            chunk_rows.append((index, chunk))
    if not chunk_rows:
        raise ParseError("The file was read but contained no usable text.")

    texts, titles, title = [s.text for s in sections], [s.title for s in sections], parsed.title
    vault_entries = None
    if sensitive:
        progress("Finding and hiding personal data (on this computer)", force=True)
        texts, titles, title, vault_entries = _redact(sections, chunk_rows, parsed.title)

    vectors = []
    for start in range(0, len(chunk_rows), EMBED_BATCH):
        progress(f"Embedding chunks {start + 1}-{min(start + EMBED_BATCH, len(chunk_rows))} of {len(chunk_rows)}")
        vectors.extend(embed([c.text for _, c in chunk_rows[start : start + EMBED_BATCH]]))

    progress("Saving to database", force=True)
    with get_pool().connection() as conn, conn.transaction():
        conn.execute("DELETE FROM sections WHERE document_id = %s", [doc_id])  # cascades to chunks
        conn.execute("DELETE FROM chunks WHERE document_id = %s", [doc_id])
        section_ids = []
        labels = section_labels(sections)
        for ordinal, s in enumerate(sections, start=1):
            row = conn.execute(
                "INSERT INTO sections (document_id, ordinal, label, title, number, page_start, page_end, n_words, text)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                [doc_id, ordinal, labels[ordinal - 1], titles[ordinal - 1], s.number, s.page_start, s.page_end,
                 s.n_words, texts[ordinal - 1]],
            ).fetchone()  # fmt: skip
            section_ids.append(row["id"])
        if vault_entries is not None:
            save_entries(conn, doc_id, vault_entries)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, section_id, page, ordinal, text, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (doc_id, section_ids[index], chunk.page, ordinal, chunk.text, vector)
                    for ordinal, ((index, chunk), vector) in enumerate(zip(chunk_rows, vectors, strict=True), start=1)
                ],
            )
        conn.execute(
            # parsers return "" when the file has no real title; then keep the original filename
            "UPDATE documents SET title = COALESCE(NULLIF(%s, ''), title), n_pages = %s, ocr_pages = %s,"
            " status = 'ready', progress = NULL, error = NULL WHERE id = %s",
            [title[:300], parsed.n_pages, parsed.ocr_pages, doc_id],
        )


def process_document(doc_id: int, path: Path, file_type: str, build_map: bool = True) -> None:
    """Background task entry point. Never raises: failures are recorded on the document row."""
    with _ingest_lock:
        try:
            _ingest(doc_id, path, file_type)
        except ParseError as exc:
            update_document(doc_id, status="failed", error=str(exc), progress=None)
            return
        except Exception as exc:
            log.exception("ingest failed for document %s", doc_id)
            update_document(
                doc_id,
                status="failed",
                error=f"Unexpected error while processing ({type(exc).__name__}).",
                progress=None,
            )
            return
        finally:
            path.unlink(missing_ok=True)  # the original file is not kept once its text is in the DB
    if build_map:
        from app.docmap.summaries import build_document_map

        build_document_map(doc_id)


def delete_document(doc_id: int) -> bool:
    """Delete a document; ON DELETE CASCADE removes sections, chunks, entities, summary, PII vault and logs."""
    with get_pool().connection() as conn:
        deleted = conn.execute("DELETE FROM documents WHERE id = %s RETURNING id", [doc_id]).fetchone()
    for leftover in UPLOAD_DIR.glob(f"{doc_id}.*"):
        leftover.unlink(missing_ok=True)
    return deleted is not None
