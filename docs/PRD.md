# AskMyDocs — Product Requirements

## Overview
Upload any document and ask anything about it. Answers are grounded in the document, cite chapter or
page, and refuse when the answer isn't there. Works for stories ("How does Elizabeth's opinion of
Darcy change?"), study material ("Explain the Krebs cycle from chapter 4") and formal documents
("What's the notice period in this agreement?").

**Real-world need:** "chat with our documents" assistants are one of the most common enterprise AI
projects (HR policies, contracts, manuals, support knowledge bases).

**Audience:** recruiters/interviewers for SWE and AI roles. Users for testing: classmates.

**Goals**
1. Grounded, cited answers; no made-up facts.
2. Good answers to whole-document and character questions, not just lookups (document map).
3. Measured quality: an eval set with before/after numbers for every improvement.
4. Safe with sensitive documents (PII redaction layer, phase 9).

## User stories
| # | Story | Acceptance criteria |
|---|---|---|
| 1 | Upload a PDF/DOCX/EPUB/TXT/MD | Stored as its own document; progress shown; parse errors reported clearly |
| 2 | Ask a specific question | Answer with citations; excerpt viewable for each citation |
| 3 | Ask a whole-document question ("summarise", "main themes") | Answered from chapter summaries, citing chapters |
| 4 | Ask about a character/person | Uses entity index + chunks where they appear; lists relationships if asked |
| 5 | Ask something not in the document | Exactly "I couldn't find that in this document." — no LLM call |
| 6 | See the document map | Chapter summaries + character/entity list |
| 7 | Delete a document | All its data removed (chunks, vectors, summaries, entities, vault) |
| 8 | Sensitive mode (phase 9) | PII never reaches Groq; answers shown with real values only to the user |

**Non-goals (for now):** user accounts, sharing, multiple users editing, mixing documents in one
answer, fine-tuning models.

**Added scope (2026-10-24):** local OCR (RapidOCR) for scanned PDF pages and PNG/JPG uploads;
Gemini as the primary LLM with Groq as automatic fallback when rate-limited.

## Data model
| Table | Columns | Notes |
|---|---|---|
| documents | id, filename, file_type, title, n_pages, status, sensitive (bool), created_at | status: processing / ready / failed |
| sections | id, document_id, ordinal, title, page_start, page_end, summary | chapters or headings; summary from document map |
| chunks | id, document_id, section_id, page, ordinal, text, embedding vector(384), tsv tsvector | GIN index on tsv; HNSW or IVFFlat index on embedding |
| entities | id, document_id, name, aliases text[], kind, description, section_ids int[] | characters, people, organisations, places |
| doc_summary | document_id, summary | whole-document summary |
| pii_vault (phase 9) | document_id, placeholder, encrypted_value, pii_type | values encrypted at rest |
| query_log | id, document_id, question, route, retrieved_chunk_ids, latency_ms, created_at | for debugging + eval; never store PII values |

Embedding size 384 matches `all-MiniLM-L6-v2`. If the embedding model changes, the dimension and
index must change too.

## Retrieval and answering
1. **Route the question** (phase 6): specific / whole-document / entity. Start with simple rules
   (keywords like "summarise", "overall", "who is", a known entity name); an LLM classifier only if
   rules prove weak on the eval set.
2. **Retrieve:**
   - specific → hybrid search: top 20 from vector + top 20 from full-text, merged with Reciprocal
     Rank Fusion (score = Σ 1/(60 + rank)), keep top 6
   - whole-document → section summaries (+ whole-document summary)
   - entity → entity record + top chunks mentioning the entity or its aliases
3. **Refuse** if the best result is below a similarity threshold (tuned on the eval set).
4. **Prompt:** system rules (use only excerpts, cite, refuse, ignore instructions inside excerpts),
   excerpts wrapped in `<excerpt source="..." loc="p.12">...</excerpt>`, then the question.
5. **Return:** answer text, citations (source + location + excerpt id), route used.

## Document map (phase 6)
- **Sections:** detect chapters from headings ("Chapter 1", "CHAPTER I", numbered headings, EPUB
  table of contents, DOCX heading styles); fall back to fixed page ranges.
- **Section summaries (map step):** one short summary per section (~120 words) via Groq.
- **Document summary (reduce step):** combine section summaries into one overview.
- **Entity index:** from each section, extract named characters/people/organisations with a one-line
  role; merge duplicates and aliases ("Lizzy" = "Elizabeth Bennet") across sections.
- Cache everything; build once per upload. Show progress in the UI.
- Watch Groq rate limits: process sections sequentially with backoff for long books.

## Evaluation
- 3 documents: a public-domain novel (Project Gutenberg), a textbook-style chapter, a contract or
  policy document.
- 10 questions each with expected answers, tagged: specific (5), whole-document (3), entity (2),
  plus 2 unanswerable questions per document to test refusal.
- Metrics: retrieval hit rate (expected location in retrieved set), answer correctness (manual or
  LLM-graded with a rubric, spot-checked by me), refusal accuracy, average latency.
- Results table in the README: baseline (phase 5) vs document map (phase 6) vs any later change.

## Non-functional
- Ingest a 300-page book in a reasonable time on a laptop; show progress.
- Answers typically within a few seconds on Groq.
- Max upload size configurable (e.g. 20 MB); reject other file types.
- Tests: parsers, chunker, RRF merge, router rules, refusal path, deletion cascade, prompt-injection
  test (a document containing "ignore previous instructions" must not change behaviour).
- CI: ruff + pytest with a pgvector Postgres service; the LLM is mocked in tests.

---

## Improvement: PII redaction layer — "Sensitive mode" (phase 9)

**Goal:** let users ask questions about and summarise important documents (offer letters, rental
agreements, bank statements, medical reports, ID scans as text) **without sending personal data to
the LLM API**. Embeddings already run locally; this makes the Groq calls private too.

**Why it matters:** companies can't send customer or employee PII to external AI APIs freely. A
redact → ask → restore pipeline is a real enterprise pattern, and it builds on Skrim's redaction work.

**Per-document toggle:** a document is marked *sensitive* at upload. Default **off**, because for a
novel the "PII" is character names and redacting them would break the answers.

### How it works
1. **Detect** (locally, before anything leaves the machine):
   - Regex + validators: email, Indian phone numbers (+91 / 10-digit), PAN (`[A-Z]{5}[0-9]{4}[A-Z]`),
     Aadhaar-like 12-digit numbers (validate with the Verhoeff checksum), card numbers (Luhn check),
     IFSC codes, bank account-like numbers near words like "A/c" or "account", dates of birth near
     "DOB", UPI IDs.
   - Names, addresses, organisations: a local NER model (e.g. Microsoft Presidio with spaCy) —
     check current install instructions; treat as best-effort.
2. **Replace with consistent placeholders per document:** "Rahul Sharma" → `[PERSON_1]` every time it
   appears; "rahul@x.com" → `[EMAIL_1]`. Consistency lets the LLM still reason ("PERSON_1 signed with
   ORG_2").
3. **Vault:** placeholder → real value, stored **encrypted** (Fernet, key from `.env`), per document,
   deleted with the document.
4. **Store redacted text** in `chunks.text` for sensitive documents; embeddings computed locally.
5. **Ask:** retrieval and the Groq prompt use only redacted text.
6. **Restore locally:** placeholders in the answer are swapped back to real values just before
   showing the user. Optional "show redacted" toggle.
7. **Tripwire:** before every outbound Groq request, scan the payload with the detectors; if any raw
   PII is found, **block the request** and log the type (never the value). Unit tests prove it.
8. **Transparency panel:** "What was sent to the AI" shows the exact redacted prompt for the last
   answer.

### Features this unlocks
- **Private Q&A:** "What is the notice period?" "Who are the parties?" on an agreement.
- **Redacted summary export:** a one-page summary of a sensitive document, downloadable either
  with placeholders (safe to share) or restored (for the owner only).
- **PII report:** count of each PII type found in the document, with page numbers, so the user knows
  what's inside before sharing it.

### Acceptance criteria
- A test suite of sample sensitive documents with planted PII: the tripwire catches 100% of planted
  regex-type PII in outbound payloads; NER recall reported honestly in the README.
- Answers on sensitive documents still pass a small eval set (e.g. 10 questions on a sample rental
  agreement).
- Deleting the document deletes its vault.

### Honest limits (put these in the README)
- Detection is best-effort: unusual formats, nicknames or context clues can slip through. This is a
  risk-reduction layer, not a compliance guarantee.
- Redaction can reduce answer quality when the question is about the redacted thing itself.
- Use only synthetic or your own documents for demos — never real people's IDs.

## Stretch: visual mode (ColPali)
For scanned or table/chart-heavy PDFs where text extraction returns little or messy text, retrieve
page images with a late-interaction vision model and answer with a vision-capable LLM. Only after
phases 1–9 are done.
