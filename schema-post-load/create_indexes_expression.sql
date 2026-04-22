-- schema-post-load/create_indexes_expression.sql
-- Post-load indexing for AUSLegalSearch v3 using EXPRESSION INDEXES ONLY.
-- Use this variant if you want to AVOID any table rewrite (no generated columns).
-- These indexes are safe to add after data load and do not require re-ingestion.
-- Run during a low-traffic window. If your tool rejects CONCURRENTLY, keep as-is.
-- If using psql and you want non-blocking builds, change to CREATE INDEX CONCURRENTLY.

-- NOTE:
-- - Requires extensions: pg_trgm (for trigram). Install separately if needed:
--   CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- - Run ANALYZE after building indexes to refresh stats.

-- =========================
-- Equality/range filters on JSONB keys in embeddings.chunk_metadata
-- =========================
CREATE INDEX IF NOT EXISTS idx_e_md_type_expr
  ON public.embeddings ((chunk_metadata->>'type'));

CREATE INDEX IF NOT EXISTS idx_e_md_jurisdiction_expr
  ON public.embeddings ((chunk_metadata->>'jurisdiction'));

CREATE INDEX IF NOT EXISTS idx_e_md_subjurisdiction_expr
  ON public.embeddings ((chunk_metadata->>'subjurisdiction'));

CREATE INDEX IF NOT EXISTS idx_e_md_database_expr
  ON public.embeddings ((chunk_metadata->>'database'));

CREATE INDEX IF NOT EXISTS idx_e_md_date_expr
  ON public.embeddings (
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
  );

CREATE INDEX IF NOT EXISTS idx_e_md_year_expr
  ON public.embeddings (
    CASE
      WHEN translate(coalesce((chunk_metadata->>'year'), ''), '0123456789', '') = ''
           AND length(coalesce((chunk_metadata->>'year'), '')) BETWEEN 1 AND 6
      THEN (chunk_metadata->>'year')::int
      ELSE NULL
    END
  );

-- Optional: exact equality on title/author/citation
CREATE INDEX IF NOT EXISTS idx_e_md_title_expr
  ON public.embeddings ((chunk_metadata->>'title'));

CREATE INDEX IF NOT EXISTS idx_e_md_author_expr
  ON public.embeddings ((chunk_metadata->>'author'));

CREATE INDEX IF NOT EXISTS idx_e_md_citation_expr
  ON public.embeddings ((chunk_metadata->>'citation'));

-- =========================
-- Array membership: countries[], titles[], citations[]
-- =========================
CREATE INDEX IF NOT EXISTS idx_e_md_countries_gin_expr
  ON public.embeddings USING GIN ((chunk_metadata->'countries'));

CREATE INDEX IF NOT EXISTS idx_e_md_titles_gin_expr
  ON public.embeddings USING GIN ((chunk_metadata->'titles'));

CREATE INDEX IF NOT EXISTS idx_e_md_citations_gin_expr
  ON public.embeddings USING GIN ((chunk_metadata->'citations'));

-- =========================
-- Trigram GIN for approximate matching on lower(title) / lower(author)
-- =========================
CREATE INDEX IF NOT EXISTS idx_e_md_title_trgm_expr
  ON public.embeddings USING GIN ((lower(coalesce(chunk_metadata->>'title',''))) gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_e_md_author_trgm_expr
  ON public.embeddings USING GIN ((lower(coalesce(chunk_metadata->>'author',''))) gin_trgm_ops);

-- =========================
-- documents.source trigram for approximate source matching
-- =========================
CREATE INDEX IF NOT EXISTS idx_documents_source_trgm_expr
  ON public.documents USING GIN ((lower(source)) gin_trgm_ops);

-- =========================
-- Vector index guidance (choose one, run separately depending on pgvector version)
-- =========================
-- HNSW (pgvector >= 0.7, best for low-latency top-k):
-- CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw_cosine
--   ON public.embeddings USING hnsw (vector vector_cosine_ops)
--   WITH (m = 16, ef_construction = 200);
-- After validation, optionally drop IVFFLAT:
-- DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat_cosine;

-- IVFFLAT (if HNSW not available). Adjust lists to your dataset size (~sqrt(N)):
-- DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat_cosine;
-- CREATE INDEX idx_embeddings_vector_ivfflat_cosine
--   ON public.embeddings USING ivfflat (vector vector_cosine_ops)
--   WITH (lists = 4096);

-- =========================
-- Post-index maintenance (run separately if your tool does not allow ANALYZE):
-- =========================
-- ANALYZE public.embeddings;
-- ANALYZE public.documents;

-- End of expression-only index script.
