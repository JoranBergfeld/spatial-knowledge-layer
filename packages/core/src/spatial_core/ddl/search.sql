-- One snapshot's search index in Postgres: a table of chunks with their vectors.
--
-- Ingestion applies this when it creates an index, to a table that is not there.
-- The vector extension is not created here. The owner may not create it, so it is
-- the database's superuser's to set up once, and load only checks it is there.
-- Placeholders in braces are filled in by ingestion.

CREATE SCHEMA IF NOT EXISTS search;
GRANT USAGE ON SCHEMA search TO {reader};

-- Field for field the Chunk record, plus the vector and the words to match.
CREATE TABLE search.{table} (
    id                   text PRIMARY KEY,
    document_id          text NOT NULL,
    role                 text NOT NULL CHECK (role IN ('passage', 'summary')),
    ordinal              integer NOT NULL,
    kind                 text NOT NULL,
    title                text NOT NULL,
    text                 text NOT NULL,
    language             text NOT NULL,
    url                  text NOT NULL,
    original_text_sha256 text NOT NULL,
    translated           boolean NOT NULL,
    space_ids            text[] NOT NULL,
    asset_type_ids       text[] NOT NULL,
    asset_ids            text[] NOT NULL,
    ancestor_ids         text[] NOT NULL,
    clearance_class      text NOT NULL CHECK (clearance_class IN ('visitor', 'staff', 'maintenance')),
    document_date        text,
    link_status          text NOT NULL,
    link_checked_on      text NOT NULL,
    embedding            vector({dimensions}) NOT NULL,
    -- English, like the Azure index's analyzer. Title and text both, as one.
    words                tsvector GENERATED ALWAYS AS
                             (to_tsvector('english', title || ' ' || text)) STORED
);
CREATE INDEX ON search.{table} USING gin (words);
-- No index on the vector, on purpose. The filter cuts the chunks first and the
-- distance is exact over what is left. An HNSW index would rank before the
-- filter and could come back short. A snapshot is a few thousand chunks.

-- Default privileges are not set on this database, so the reader gets each table by hand.
GRANT SELECT ON search.{table} TO {reader};
