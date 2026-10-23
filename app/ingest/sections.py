"""Split a parsed document into sections (chapters / headings).

Order of preference:
1. Real heading markup (DOCX styles, EPUB <h*>, Markdown #): use the shallowest level that
   appears at least twice (a single <h1> is usually just the book title).
2. Heading-looking lines: "Chapter 4", "CHAPTER IV.", "Part Two", "Prologue", "4. TERMINATION".
3. Fallback: fixed page ranges (PDF) or fixed word counts (everything else).

Tiny sections (e.g. table-of-contents lines that look like chapter headings) are merged
into the next section, so they never become empty "chapters".
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from app.ingest.parsers import Block

MIN_SECTION_WORDS = 50
FALLBACK_PAGES = 10
FALLBACK_WORDS = 3000

NUMBER_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

CHAPTER_RE = re.compile(
    r"^(chapter|chap\.|book|part|letter|section|act|stave)\s+"
    r"([0-9]{1,3}|[ivxlcdm]{1,8}|" + "|".join(NUMBER_WORDS) + r")\b[\s.:\-—]*(.{0,80})$",
    re.IGNORECASE,
)
SPECIAL_RE = re.compile(
    r"^(prologue|epilogue|preface|foreword|introduction|conclusion|afterword|appendix(\s+\w+)?)\.?$",
    re.IGNORECASE,
)
NUMBERED_CAPS_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Z0-9 ,&'\-]{2,60})$")  # "4. TERMINATION"


@dataclass
class Section:
    title: str
    blocks: list[Block] = field(default_factory=list)
    number: int | None = None

    @property
    def n_words(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    @property
    def page_start(self) -> int | None:
        pages = [b.page for b in self.blocks if b.page is not None]
        return min(pages) if pages else None

    @property
    def page_end(self) -> int | None:
        pages = [b.page for b in self.blocks if b.page is not None]
        return max(pages) if pages else None


def parse_number(token: str) -> int | None:
    token = token.lower().strip(".")
    if token.isdigit():
        return int(token)
    if token in NUMBER_WORDS:
        return NUMBER_WORDS[token]
    if token and all(ch in ROMAN for ch in token):
        total = 0
        for i, ch in enumerate(token):
            value = ROMAN[ch]
            nxt = ROMAN[token[i + 1]] if i + 1 < len(token) else 0
            total += -value if value < nxt else value
        return total if total > 0 else None
    return None


def heading_number(title: str) -> int | None:
    title = title.strip()
    if match := CHAPTER_RE.match(title):
        return parse_number(match.group(2))
    if match := NUMBERED_CAPS_RE.match(title):
        return int(match.group(1))
    return None


def looks_like_heading(text: str) -> bool:
    text = text.strip()
    if len(text) > 90 or len(text.split()) > 12:
        return False
    return bool(CHAPTER_RE.match(text) or SPECIAL_RE.match(text) or NUMBERED_CAPS_RE.match(text))


def _heading_level(blocks: list[Block]) -> int | None:
    counts = Counter(b.level for b in blocks if b.level > 0)
    for level in sorted(counts):
        if counts[level] >= 2:
            return level
    return None


def _merge_tiny(sections: list[Section]) -> list[Section]:
    merged: list[Section] = []
    carry: list[Block] = []
    for section in sections:
        section.blocks = carry + section.blocks
        carry = []
        heading_words = len(section.title.split())
        if section.n_words - heading_words < MIN_SECTION_WORDS:
            carry = section.blocks  # too small: fold it into the next section
        else:
            merged.append(section)
    if carry:
        if merged:
            merged[-1].blocks.extend(carry)
        else:
            merged.append(Section(title="Document", blocks=carry))
    return merged


def _fallback(blocks: list[Block]) -> list[Section]:
    sections: list[Section] = []
    if any(b.page is not None for b in blocks):
        groups: dict[int, list[Block]] = {}
        for b in blocks:
            groups.setdefault(((b.page or 1) - 1) // FALLBACK_PAGES, []).append(b)
        for key in sorted(groups):
            pages = [b.page for b in groups[key] if b.page is not None]
            sections.append(Section(title=f"Pages {min(pages)}-{max(pages)}", blocks=groups[key]))
    else:
        current: list[Block] = []
        words = 0
        for b in blocks:
            current.append(b)
            words += len(b.text.split())
            if words >= FALLBACK_WORDS:
                sections.append(Section(title=f"Part {len(sections) + 1}", blocks=current))
                current, words = [], 0
        if current:
            sections.append(Section(title=f"Part {len(sections) + 1}", blocks=current))
    return _merge_tiny(sections)


def split_sections(blocks: list[Block]) -> list[Section]:
    level = _heading_level(blocks)
    if level is not None:
        is_boundary = [b.level == level for b in blocks]
    else:
        is_boundary = [looks_like_heading(b.text) for b in blocks]

    sections: list[Section] = []
    current = Section(title="Opening")
    for block, boundary in zip(blocks, is_boundary, strict=True):
        if boundary:
            if current.blocks:
                sections.append(current)
            current = Section(title=block.text.strip()[:120], number=heading_number(block.text))
        current.blocks.append(block)
    if current.blocks:
        sections.append(current)

    sections = _merge_tiny(sections)
    if len(sections) <= 1 and sum(len(b.text.split()) for b in blocks) > FALLBACK_WORDS:
        return _fallback(blocks)  # no structure found in a long document
    return sections
