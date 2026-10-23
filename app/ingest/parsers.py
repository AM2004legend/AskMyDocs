"""Turn an uploaded file into a flat list of text blocks, keeping page numbers and headings.

A Block is one line/paragraph. `level` > 0 marks a real heading (DOCX heading style,
EPUB <h1>-<h3>, Markdown #). PDFs and TXT files have no heading markup, so chapter
headings there are found later by regex in sections.py.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings

SUPPORTED_TYPES = {"pdf", "docx", "epub", "txt", "md", "png", "jpg", "jpeg"}


class ParseError(Exception):
    """A user-facing problem with the file (corrupt, encrypted, empty...)."""


@dataclass
class Block:
    text: str
    page: int | None = None
    level: int = 0  # 0 = normal text, 1..6 = heading level


@dataclass
class ParsedDoc:
    title: str
    blocks: list[Block]
    n_pages: int | None = None
    ocr_pages: int = 0
    warnings: list[str] = field(default_factory=list)


Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


def _clean(text: str) -> str:
    # Postgres text can't hold NUL bytes, which some PDFs produce
    return re.sub(r"[ \t ]+", " ", text.replace("\x00", "")).strip()


# ---------- PDF ----------
def parse_pdf(path: Path, progress: Progress = _noop) -> ParsedDoc:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    settings = get_settings()
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")  # many "encrypted" PDFs just have an empty owner password
        pages = reader.pages
        n_pages = len(pages)
    except (PdfReadError, ValueError, NotImplementedError) as exc:
        raise ParseError(f"Could not read this PDF ({exc}). Is it password-protected or damaged?") from exc

    title = (reader.metadata.title if reader.metadata else None) or ""
    blocks: list[Block] = []
    ocr_pages = 0
    pdfium_doc = None
    for page_no, page in enumerate(pages, start=1):
        progress(f"Reading page {page_no}/{n_pages}")
        try:
            text = page.extract_text() or ""
        except Exception:  # pypdf can choke on odd pages; OCR may still rescue them
            text = ""
        if len(text.strip()) < settings.ocr_min_chars and settings.ocr_enabled:
            progress(f"OCR on page {page_no}/{n_pages} (scanned page)")
            if pdfium_doc is None:
                import pypdfium2 as pdfium

                pdfium_doc = pdfium.PdfDocument(str(path))
            from app.ingest.ocr import ocr_image

            image = pdfium_doc[page_no - 1].render(scale=settings.ocr_render_scale).to_pil()
            ocr_text = ocr_image(image)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
                ocr_pages += 1
        for line in text.splitlines():
            line = _clean(line)
            if line:
                blocks.append(Block(line, page=page_no))
    if pdfium_doc is not None:
        pdfium_doc.close()
    return ParsedDoc(title=str(title), blocks=blocks, n_pages=n_pages, ocr_pages=ocr_pages)


# ---------- images ----------
def parse_image(path: Path, progress: Progress = _noop) -> ParsedDoc:
    from PIL import Image, UnidentifiedImageError

    if not get_settings().ocr_enabled:
        raise ParseError("Image uploads need OCR, which is turned off (OCR_ENABLED=false).")
    progress("Running OCR on image")
    try:
        with Image.open(path) as image:
            from app.ingest.ocr import ocr_image

            text = ocr_image(image)
    except UnidentifiedImageError as exc:
        raise ParseError("This file is not a readable image.") from exc
    blocks = [Block(_clean(line), page=1) for line in text.splitlines() if _clean(line)]
    return ParsedDoc(title="", blocks=blocks, n_pages=1, ocr_pages=1)


# ---------- DOCX ----------
def _docx_heading_level(style_name: str) -> int:
    match = re.match(r"Heading (\d)", style_name or "")
    return int(match.group(1)) if match else 0


def parse_docx(path: Path, progress: Progress = _noop) -> ParsedDoc:
    import docx
    from docx.table import Table

    progress("Reading DOCX")
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # python-docx raises several unrelated types for bad files
        raise ParseError(f"Could not read this DOCX file ({exc}).") from exc

    blocks: list[Block] = []
    for item in document.iter_inner_content():  # paragraphs and tables in document order
        if isinstance(item, Table):
            for row in item.rows:
                cells = [_clean(c.text) for c in row.cells]
                line = " | ".join(dict.fromkeys(c for c in cells if c))  # merged cells repeat
                if line:
                    blocks.append(Block(line))
        else:
            text = _clean(item.text)
            if text:
                level = _docx_heading_level(item.style.name if item.style else "")
                blocks.append(Block(text, level=level))
    title = document.core_properties.title or ""
    return ParsedDoc(title=title, blocks=blocks)


# ---------- EPUB ----------
def parse_epub(path: Path, progress: Progress = _noop) -> ParsedDoc:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    progress("Reading EPUB")
    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    except Exception as exc:
        raise ParseError(f"Could not read this EPUB file ({exc}).") from exc

    items_by_id = {item.get_id(): item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
    spine_items = [items_by_id[idref] for idref, _ in book.spine if idref in items_by_id]

    per_item: list[list[Block]] = []
    n_headings = 0
    for item in spine_items:
        soup = BeautifulSoup(item.get_content(), "html.parser")
        item_blocks = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "td"]):
            if tag.find(["p", "li", "blockquote"]):  # container; its children are visited separately
                continue
            text = _clean(tag.get_text(" "))
            if not text:
                continue
            level = int(tag.name[1]) if tag.name in {"h1", "h2", "h3", "h4"} else 0
            n_headings += level > 0
            item_blocks.append(Block(text, level=level))
        per_item.append(item_blocks)

    blocks: list[Block] = []
    for index, item_blocks in enumerate(per_item, start=1):
        if n_headings < 2 and item_blocks:  # no usable headings: each spine file becomes a section
            blocks.append(Block(f"Section {index}", level=1))
        blocks.extend(item_blocks)

    titles = book.get_metadata("DC", "title")
    title = titles[0][0] if titles else ""
    return ParsedDoc(title=title, blocks=blocks)


# ---------- TXT / MD ----------
def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _read_text_file(path: Path) -> str:
    # Windows files use \r\n line endings; normalise so paragraph splitting works everywhere
    return _decode(path.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")


GUTENBERG_START_RE = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE | re.IGNORECASE)
GUTENBERG_END_RE = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE | re.IGNORECASE)


def _strip_gutenberg(text: str) -> tuple[str, str]:
    """Project Gutenberg files wrap the book in licence text; keep only the book. Returns (text, title)."""
    start, end = GUTENBERG_START_RE.search(text), GUTENBERG_END_RE.search(text)
    if not (start and end and start.end() < end.start()):
        return text, ""
    title = re.search(r"^Title:\s*(.+)$", text[: start.start()], re.MULTILINE)
    return text[start.end() : end.start()], title.group(1).strip() if title else ""


def parse_txt(path: Path, progress: Progress = _noop) -> ParsedDoc:
    progress("Reading text")
    text, title = _strip_gutenberg(_read_text_file(path))
    # paragraphs are separated by blank lines; hard-wrapped lines inside a paragraph get joined
    blocks = [Block(_clean(p.replace("\n", " "))) for p in re.split(r"\n\s*\n", text)]
    return ParsedDoc(title=title, blocks=[b for b in blocks if b.text])


def parse_md(path: Path, progress: Progress = _noop) -> ParsedDoc:
    progress("Reading Markdown")
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush():
        if paragraph:
            blocks.append(Block(_clean(" ".join(paragraph))))
            paragraph.clear()

    for line in _read_text_file(path).splitlines():
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush()
            blocks.append(Block(_clean(heading.group(2).strip("# ")), level=len(heading.group(1))))
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.strip())
    flush()
    return ParsedDoc(title="", blocks=[b for b in blocks if b.text])


PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "epub": parse_epub,
    "txt": parse_txt,
    "md": parse_md,
    "png": parse_image,
    "jpg": parse_image,
    "jpeg": parse_image,
}


def parse_file(path: Path, file_type: str, progress: Progress = _noop) -> ParsedDoc:
    if file_type not in PARSERS:
        raise ParseError(f"Unsupported file type '.{file_type}'.")
    parsed = PARSERS[file_type](path, progress)
    if not any(b.text.strip() for b in parsed.blocks):
        hint = "" if get_settings().ocr_enabled else " (OCR is turned off)"
        raise ParseError(f"No readable text found in this file{hint}.")
    return parsed
