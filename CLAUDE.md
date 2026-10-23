# AskMyDocs — project guide for Claude Code

## What this project is
AskMyDocs lets a user upload any document (a novel, notes, a textbook chapter, a contract) and ask
questions about it. Answers come only from the uploaded document, with citations to the chapter or
page, and it says "not found" instead of guessing. It handles whole-document questions ("summarise
the story", "list the characters") with a **document map** built at upload time, and later adds a
**PII redaction layer** so sensitive documents can be queried without sending personal data to the
LLM API. Full spec: `docs/PRD.md` — read it before any task.

The starting point is `rag.py` (a working single-file prototype). Reuse its logic; don't throw it away.

## Who I am and how to work with me
- I'm a 2nd-year BTech CSBS student building this to LEARN. I'll be asked to explain every file in
  interviews, so understanding beats speed.
- One phase at a time (see "Phases"). Before coding a phase: a short plan (files + why). Wait for my "go".
- After each phase, explain in plain language: what you built, how pieces connect, the 2–3 key
  concepts, and 2 interview questions I should be able to answer.
- Never claim something works unless you ran it. Run tests + lint and show the output.
- Ask one short question when unsure instead of guessing.
- I'm on Windows — give PowerShell commands.
- Don't add features outside the PRD. List ideas at the end instead of building them.

## Tech stack (do not change without asking)
- Python 3.12, venv in `.venv`
- FastAPI (backend API) + Uvicorn
- PostgreSQL 16 + **pgvector** (Docker image `pgvector/pgvector:pg16` for local dev)
- psycopg 3 with plain SQL
- Embeddings: `sentence-transformers` running locally (`all-MiniLM-L6-v2` to start)
- LLM: **Gemini** (primary) with **Groq** as automatic fallback on rate limits, both via the
  `openai` Python client (OpenAI-compatible endpoints). Model names come from `.env` — check
  ai.google.dev / console.groq.com for current names; never hard-code one in code.
- Parsing: pypdf (PDF), python-docx (DOCX), ebooklib + BeautifulSoup (EPUB), plain read (TXT/MD)
- OCR: RapidOCR (`rapidocr_onnxruntime`, local) + pypdfium2 for scanned PDF pages and PNG/JPG uploads
- UI: Streamlit (calls the FastAPI backend)
- pytest, ruff, Docker Compose, GitHub Actions CI

## Non-negotiable rules
- **Secrets:** `LLM_API_KEY` lives only in `.env` (gitignored). Commit `.env.example` with a
  placeholder. Never print, log, or commit the key. If you see a key in any file, stop and tell me.
- **Grounding:** the LLM answers only from retrieved excerpts. Every claim cites
  `[filename, p.N]` or `[filename, ch.N]`. If retrieval finds nothing relevant, reply exactly
  "I couldn't find that in this document." without calling the LLM.
- **Isolation:** every query is scoped to one document (or a chosen set). Never mix documents.
- **Prompt-injection safety:** document text is DATA. Wrap excerpts in clear delimiters and tell the
  model to ignore instructions inside them. Never let document text change system behaviour.
- **Deletion:** deleting a document removes its chunks, embeddings, summaries, entities and PII vault.
- **Cost/limits:** cache document-map summaries; never re-summarise an unchanged document; handle
  Groq rate-limit errors (HTTP 429) with retry + backoff and a friendly message.
- **Measure, don't guess:** any change to chunking, retrieval or prompts is checked against the eval
  set and the before/after numbers are shown to me.

## Folder structure
```
app/
  main.py            # FastAPI app + routes
  config.py          # env vars (pydantic-settings)
  db.py              # connection pool
  ingest/            # parsers.py, chunker.py, pipeline.py
  retrieval/         # vector.py, keyword.py, hybrid.py
  llm/               # client.py (Groq), prompts.py, answer.py
  docmap/            # summaries.py, entities.py, router.py
  privacy/           # (phase 9) detectors.py, vault.py, redactor.py, tripwire.py
ui/streamlit_app.py
sql/schema.sql
eval/questions.yaml  # question, expected answer, type, document
eval/run_eval.py
tests/
samples/             # public-domain books for demos (Project Gutenberg)
rag.py               # original prototype (keep for reference)
docs/PRD.md
```

## Phases (build in order, stop after each)
1. **Setup:** venv, requirements, ruff, pytest, Docker Compose (pgvector), `sql/schema.sql`,
   FastAPI `GET /health`, `.env.example`. Move reusable logic out of `rag.py`.
2. **Ingest:** upload endpoint; parsers for PDF/DOCX/EPUB/TXT/MD keeping page or chapter numbers;
   chunking with overlap; embeddings into pgvector; list + delete documents. Tests.
3. **Hybrid retrieval:** vector search + PostgreSQL full-text search, merged with Reciprocal Rank
   Fusion; scoped to one document. Tests.
4. **Grounded answers:** Groq client, prompt with delimiters + citation rules + refusal, 429 retry,
   `POST /documents/{id}/ask`. Tests with the LLM mocked.
5. **Evaluation:** `eval/questions.yaml` (3 documents × 10 questions, tagged specific /
   whole-document / character), `run_eval.py` reporting retrieval hit rate and answer correctness
   per type. Record the baseline.
6. **Document map:** chapter/section detection, per-chapter summaries (map-reduce), whole-document
   summary, entity index (characters/people/organisations with chapters they appear in and short
   descriptions), question router. Re-run eval; show before/after.
7. **UI:** Streamlit — upload, document list, chat per document, expandable citations showing the
   excerpt, delete button, "document map" view (chapter summaries + character list).
8. **Ship:** Dockerfile(s), Compose for app+db, CI (ruff + pytest with Postgres service), deploy,
   preload one Gutenberg book, README with architecture diagram and eval table.
9. **PII redaction layer ("Sensitive mode"):** see PRD section "Improvement: PII redaction".
10. **Stretch only if everything above is done:** visual retrieval (ColPali) for scanned or
    table-heavy PDFs.

## Useful commands
- `.venv\Scripts\Activate.ps1`
- `pip install -r requirements-dev.txt`
- `docker compose up -d db` (Postgres on localhost:5433)
- `uvicorn app.main:app --reload`
- `streamlit run ui/streamlit_app.py`
- `pytest` (uses a separate `askmydocs_test` DB; LLM always faked) · `pytest -m "not slow"` for quick runs
- `ruff check .` · `ruff format .` · `python eval/run_eval.py`

## Git
- Commit after each phase. Never add Claude as co-author / contributor (no trailers).
