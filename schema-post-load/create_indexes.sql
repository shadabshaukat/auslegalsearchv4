-- schema-post-load/create_indexes.sql
-- Post-load indexing and tuning for AUSLegalSearch v3 at large scale (~TBs)
-- This variant uses STORED GENERATED COLUMNS + simple indexes to maximize compatibility
-- with DB tools that reject expression indexes or "CONCURRENTLY".
-- Prefer to run during a low-traffic window.

-- IMPORTANT for very large tables (~4.5TB):
-- - No re-ingestion or external data load is required. PostgreSQL will derive generated
--   column values from existing JSONB per row.
-- - However, ALTER TABLE ... ADD COLUMN ... GENERATED ALWAYS AS ... STORED will cause a
--   table rewrite to materialize the new columns for existing rows. On very large tables,
--   schedule during a maintenance window. Alternatively, consider using expression indexes
--   (run via psql) to avoid a rewrite if your admin tool permits them.

-- Scope: ONLY the metadata fields listed by you are surfaced as columns + indexes:
--   Common: type, jurisdiction, subjurisdiction, database, date, year, title, titles[], author,
--           citations[], citation (single), countries[]
--   Not indexed: data_quality, url

-- Requires PostgreSQL 12+ for STORED generated columns.
-- Run CREATE EXTENSION (pg_trgm, vector, fuzzystrmatch) separately if needed.
-- After applying, ANALYZE the affected tables.

-- 0) Add STORED generated columns for requested JSONB metadata keys (embeddings)
--    Note: these columns mirror metadata; they are for indexing/filtering only.

ALTER TABLE public.embeddings
  ADD COLUMN IF NOT EXISTS md_type             text GENERATED ALWAYS AS ((chunk_metadata->>'type')) STORED,
  ADD COLUMN IF NOT EXISTS md_jurisdiction     text GENERATED ALWAYS AS ((chunk_metadata->>'jurisdiction')) STORED,
  ADD COLUMN IF NOT EXISTS md_subjurisdiction  text GENERATED ALWAYS AS ((chunk_metadata->>'subjurisdiction')) STORED,
  ADD COLUMN IF NOT EXISTS md_database         text GENERATED ALWAYS AS ((chunk_metadata->>'database')) STORED,
  ADD COLUMN IF NOT EXISTS md_date             date GENERATED ALWAYS AS (
    CASE
      WHEN
        length(split_part((chunk_metadata->>'date'), ' ', 1)) >= 10
        AND substr(split_part((chunk_metadata->>'date'), ' ', 1), 5, 1) = '-'
        AND substr(split_part((chunk_metadata->>'date'), ' ', 1), 8, 1) = '-'
        AND translate(substr(split_part((chunk_metadata->>'date'), ' ', 1), 1, 4), '0123456789', '') = ''
        AND translate(substr(split_part((chunk_metadata->>'date'), ' ', 1), 6, 2), '0123456789', '') = ''
        AND translate(substr(split_part((chunk_metadata->>'date'), ' ', 1), 9, 2), '0123456789', '') = ''
      THEN make_date(
        substr(split_part((chunk_metadata->>'date'), ' ', 1), 1, 4)::int,
        substr(split_part((chunk_metadata->>'date'), ' ', 1), 6, 2)::int,
        substr(split_part((chunk_metadata->>'date'), ' ', 1), 9, 2)::int
      )
      ELSE NULL
    END
  ) STORED,
  ADD COLUMN IF NOT EXISTS md_year             int  GENERATED ALWAYS AS (
    CASE
      WHEN (chunk_metadata ? 'year')
           AND translate(coalesce((chunk_metadata->>'year'), ''), '0123456789', '') = ''
           AND length(coalesce((chunk_metadata->>'year'), '')) BETWEEN 1 AND 6
      THEN (chunk_metadata->>'year')::int
      ELSE NULL
    END
  ) STORED,
  ADD COLUMN IF NOT EXISTS md_title            text GENERATED ALWAYS AS ((chunk_metadata->>'title')) STORED,
  ADD COLUMN IF NOT EXISTS md_title_lc         text GENERATED ALWAYS AS (lower(coalesce(chunk_metadata->>'title',''))) STORED,
  ADD COLUMN IF NOT EXISTS md_author           text GENERATED ALWAYS AS ((chunk_metadata->>'author')) STORED,
  ADD COLUMN IF NOT EXISTS md_author_lc        text GENERATED ALWAYS AS (lower(coalesce(chunk_metadata->>'author',''))) STORED,
  ADD COLUMN IF NOT EXISTS md_countries        jsonb GENERATED ALWAYS AS (chunk_metadata->'countries') STORED,
  ADD COLUMN IF NOT EXISTS md_titles           jsonb GENERATED ALWAYS AS (chunk_metadata->'titles') STORED,
  ADD COLUMN IF NOT EXISTS md_citations        jsonb GENERATED ALWAYS AS (chunk_metadata->'citations') STORED,
  ADD COLUMN IF NOT EXISTS md_citation         text  GENERATED ALWAYS AS ((chunk_metadata->>'citation')) STORED;

-- 1) Simple btree indexes for highly selective equality/range filters
CREATE INDEX IF NOT EXISTS idx_e_md_type_btree             ON public.embeddings (md_type);
CREATE INDEX IF NOT EXISTS idx_e_md_jurisdiction_btree     ON public.embeddings (md_jurisdiction);
CREATE INDEX IF NOT EXISTS idx_e_md_subjurisdiction_btree  ON public.embeddings (md_subjurisdiction);
CREATE INDEX IF NOT EXISTS idx_e_md_database_btree         ON public.embeddings (md_database);
CREATE INDEX IF NOT EXISTS idx_e_md_date_btree             ON public.embeddings (md_date);
CREATE INDEX IF NOT EXISTS idx_e_md_year_btree             ON public.embeddings (md_year);
CREATE INDEX IF NOT EXISTS idx_e_md_title_btree            ON public.embeddings (md_title);
CREATE INDEX IF NOT EXISTS idx_e_md_author_btree           ON public.embeddings (md_author);
CREATE INDEX IF NOT EXISTS idx_e_md_citation_btree         ON public.embeddings (md_citation);

-- 2) Trigram GIN for approximate text search on title/author (requires pg_trgm)
CREATE INDEX IF NOT EXISTS idx_e_md_title_trgm             ON public.embeddings USING GIN (md_title_lc gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_e_md_author_trgm            ON public.embeddings USING GIN (md_author_lc gin_trgm_ops);

-- 3) GIN on JSON arrays for membership tests (countries, titles, citations)
CREATE INDEX IF NOT EXISTS idx_e_md_countries_gin          ON public.embeddings USING GIN (md_countries);
CREATE INDEX IF NOT EXISTS idx_e_md_titles_gin             ON public.embeddings USING GIN (md_titles);
CREATE INDEX IF NOT EXISTS idx_e_md_citations_gin          ON public.embeddings USING GIN (md_citations);

-- 4) Documents.source trigram (approximate match). Add generated lower(source) to simplify indexing.
ALTER TABLE public.documents
  ADD COLUMN IF NOT EXISTS source_lc text GENERATED ALWAYS AS (lower(source)) STORED;

CREATE INDEX IF NOT EXISTS idx_documents_source_trgm       ON public.documents USING GIN (source_lc gin_trgm_ops);

-- 5) Vector index guidance (execute separately based on pgvector version):
-- Prefer HNSW on pgvector >= 0.7 for low-latency top-k:
--   CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw_cosine
--     ON public.embeddings USING hnsw (vector vector_cosine_ops)
--     WITH (m = 16, ef_construction = 200);
-- After validation, optionally drop the old IVFFLAT:
--   DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat_cosine;

-- If HNSW is not available, (re)create IVFFLAT with larger lists for better p95 (adjust lists to your size):
--   DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat_cosine;
--   CREATE INDEX idx_embeddings_vector_ivfflat_cosine
--     ON public.embeddings USING ivfflat (vector vector_cosine_ops)
--     WITH (lists = 4096);

-- 6) After all indexes are built, refresh statistics (run separately if your tool rejects ANALYZE):
--   ANALYZE public.documents;
--   ANALYZE public.embeddings;

-- Query-time recommendations (set in your app/session, not persisted here):
--   SET LOCAL jit = off;                       -- lower tail latency for short queries
--   SET LOCAL hnsw.ef_search = 60;             -- if HNSW is used
--   SET LOCAL ivfflat.probes = 10;             -- if IVFFLAT is used

-- End of schema-post-load script.
