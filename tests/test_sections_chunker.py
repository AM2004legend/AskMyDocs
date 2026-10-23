from app.ingest.chunker import chunk_blocks, split_sentences
from app.ingest.parsers import Block
from app.ingest.sections import heading_number, looks_like_heading, parse_number, section_labels, split_sections


def words(n: int, word: str = "word") -> str:
    return " ".join([word] * n) + "."


def test_parse_number_handles_digits_roman_and_words():
    assert parse_number("12") == 12
    assert parse_number("IV") == 4
    assert parse_number("xlii") == 42
    assert parse_number("seven") == 7
    assert parse_number("hello") is None


def test_heading_detection():
    assert looks_like_heading("CHAPTER IV.")
    assert looks_like_heading("Chapter 12: The Ball")
    assert looks_like_heading("Prologue")
    assert looks_like_heading("4. TERMINATION")
    assert not looks_like_heading("The chapter 4 results were surprising and the team was happy about it all.")
    assert not looks_like_heading("4. the cat sat on the mat")
    assert heading_number("CHAPTER IV.") == 4


def test_split_by_regex_headings_and_merge_toc_lines():
    blocks = [Block("Contents"), Block("Chapter 1"), Block("Chapter 2")]  # a table of contents
    blocks += [Block("Chapter 1"), Block(words(80, "alpha")), Block("Chapter 2"), Block(words(80, "beta"))]
    sections = split_sections(blocks)
    assert [s.title for s in sections] == ["Chapter 1", "Chapter 2"]
    assert [s.number for s in sections] == [1, 2]
    assert "alpha" in sections[0].text and "beta" in sections[1].text
    assert "Contents" in sections[0].text  # tiny TOC sections folded forward, nothing lost


def test_labels_use_real_chapter_numbers():
    blocks = [Block("Preface"), Block(words(60)), Block("CHAPTER I."), Block(words(60)), Block("CHAPTER II.")]
    blocks += [Block(words(60))]
    sections = split_sections(blocks)
    assert section_labels(sections) == ["sec.1", "ch.1", "ch.2"]  # not ch.2/ch.3 because of the preface


def test_labels_fall_back_when_numbers_repeat():
    blocks = []
    for book in ("Book 1", "Book 2"):
        blocks += [Block(f"Chapter 1 {book}"), Block(words(60))]
    assert section_labels(split_sections(blocks)) == ["sec.1", "sec.2"]


def test_split_uses_heading_level_that_repeats():
    blocks = [Block("My Book", level=1)]
    for n in (1, 2, 3):
        blocks += [Block(f"Part {n}", level=2), Block(words(60)), Block("detail", level=3), Block(words(10))]
    sections = split_sections(blocks)
    assert [s.title for s in sections] == ["Part 1", "Part 2", "Part 3"]


def test_fallback_to_page_ranges_without_headings():
    blocks = [Block(words(400), page=p) for p in range(1, 26)]
    sections = split_sections(blocks)
    assert [s.title for s in sections] == ["Pages 1-10", "Pages 11-20", "Pages 21-25"]
    assert (sections[1].page_start, sections[1].page_end) == (11, 20)


def test_split_sentences():
    assert split_sentences('He left. "Why?" she asked. Mr. Darcy was 3.5 m tall.') == [
        "He left.",
        '"Why?" she asked.',
        "Mr. Darcy was 3.5 m tall.",
    ]


def test_chunks_respect_size_overlap_and_pages():
    blocks = [Block(" ".join(f"Sentence p{p} n{i} has six words." for i in range(30)), page=p) for p in (1, 2)]
    chunks = chunk_blocks(blocks, chunk_words=60, overlap_words=12)
    assert all(len(c.text.split()) <= 60 for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):  # consecutive chunks share two 6-word sentences
        assert a.text.split()[-12:] == b.text.split()[:12]
    assert chunks[0].page == 1 and chunks[-1].page == 2
    all_text = " ".join(c.text for c in chunks)
    assert "p1 n0" in all_text and "p2 n29" in all_text  # nothing lost at the ends


def test_giant_sentence_is_split():
    chunks = chunk_blocks([Block(" ".join(["x"] * 500))], chunk_words=180, overlap_words=40)
    assert len(chunks) == 3 and all(len(c.text.split()) <= 180 for c in chunks)


def test_short_document_is_one_chunk():
    chunks = chunk_blocks([Block("Just one line.")], chunk_words=180, overlap_words=40)
    assert [c.text for c in chunks] == ["Just one line."]
