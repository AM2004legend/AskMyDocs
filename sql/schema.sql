-- AskMyDocs schema. Safe to run many times (everything is IF NOT EXISTS).
-- Every child table uses ON DELETE CASCADE, so deleting a document deletes
-- its sections, chunks, embeddings, entities, summary and query log.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id             bigserial PRIMARY KEY,
    filename       text        NOT NULL,
    file_type      text        NOT NULL,
    title          text,
    n_pages        int,
    content_sha256 text        NOT NULL UNIQUE,      -- same file uploaded twice = same document, no re-summarising
    status         text        NOT NULL DEFAULT 'processing'
                   CHECK (status IN ('processing', 'ready', 'failed')),
    map_status     text        NOT NULL DEFAULT 'pending'
                   CHECK (map_status IN ('pending', 'building', 'ready', 'failed')),
    progress       text,                              -- human-readable, shown in the UI
    error          text,
    ocr_pages      int         NOT NULL DEFAULT 0,
    sensitive      boolean     NOT NULL DEFAULT false, -- phase 9
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sections (
    id          bigserial PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal     int    NOT NULL,          -- 1, 2, 3 ... in reading order
    label       text,                     -- what citations show: "ch.16" (real chapter number) or "sec.2"
    title       text   NOT NULL,
    number      int,                      -- "Chapter IV" -> 4, so "chapter 4" questions can find it
    page_start  int,
    page_end    int,
    n_words     int    NOT NULL,
    text        text   NOT NULL,          -- kept so the document map can be (re)built without re-parsing
    summary     text,
    entities_raw jsonb,                   -- per-section entity extraction, merged into `entities`
    UNIQUE (document_id, ordinal)
);

ALTER TABLE sections ADD COLUMN IF NOT EXISTS label text;  -- for databases created before labels existed

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    section_id  bigint REFERENCES sections (id) ON DELETE CASCADE,
    page        int,
    ordinal     int    NOT NULL,
    text        text   NOT NULL,
    embedding   vector(384) NOT NULL,     -- 384 = all-MiniLM-L6-v2; change both if the model changes
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS entities (
    id          bigserial PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    name        text   NOT NULL,
    aliases     text[] NOT NULL DEFAULT '{}',
    kind        text   NOT NULL,          -- person / organisation / place / other
    description text   NOT NULL DEFAULT '',
    section_ids int[]  NOT NULL DEFAULT '{}'   -- section ordinals where the entity appears
);
CREATE INDEX IF NOT EXISTS entities_document_idx ON entities (document_id);

CREATE TABLE IF NOT EXISTS doc_summary (
    document_id bigint PRIMARY KEY REFERENCES documents (id) ON DELETE CASCADE,
    summary     text NOT NULL
);

-- phase 9: placeholder -> real value for sensitive documents, Fernet-encrypted (key only in .env)
CREATE TABLE IF NOT EXISTS pii_vault (
    id              bigserial PRIMARY KEY,
    document_id     bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    placeholder     text   NOT NULL,   -- "PERSON_1"
    pii_type        text   NOT NULL,   -- "PERSON", "EMAIL", "PAN", ...
    encrypted_value text   NOT NULL,
    UNIQUE (document_id, placeholder)
);

CREATE TABLE IF NOT EXISTS query_log (
    id                  bigserial PRIMARY KEY,
    document_id         bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    question            text   NOT NULL,
    route               text   NOT NULL,
    retrieved_chunk_ids bigint[] NOT NULL DEFAULT '{}',
    refused             boolean NOT NULL DEFAULT false,
    latency_ms          int    NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);
