# AUSLegalSearch v3 — Tools: SQL Latency Benchmark

This directory contains the end-to-end SQL latency benchmarking tool for the AUSLegalSearch v3 stack. It measures latency and prints top results for:
- Vector similarity searches using pgvector with optional JSON metadata filters
- Full Text Search (FTS) on documents.content via documents.document_fts
- Metadata-only filtering on JSONB
- Client-side hybrid rerank combining vector and FTS
- Optimized SQL scenarios for common legal search use cases (aligned with schema-post-load/optimized_sql.sql)

Primary script:
- bench_sql_latency.py

Related documentation:
- schema-post-load/README.md — post-load indexing, generated/expression columns, HNSW/IVFFLAT guidance
- schema-post-load/optimized_sql.sql — optimized SQL templates the benchmark can exercise


## Goals

- Quickly assess end-to-end latency (p50/p95) on realistic queries and filters
- Tune vector index parameters (ivfflat.probes, hnsw.ef_search)
- Validate that metadata filters and FTS are correctly indexed and performant
- Inspect top hits for sanity checking and recall quality
- Benchmark optimized SQL scenarios used by analysts and product features


## Prerequisites

- Python 3.10+
- Dependencies: pip install -r requirements.txt
- PostgreSQL with pgvector installed and enabled
  - CREATE EXTENSION IF NOT EXISTS vector;
  - Other recommended: pg_trgm, uuid-ossp, fuzzystrmatch
- Database connection configured via environment
  - Either AUSLEGALSEARCH_DB_URL or all AUSLEGALSEARCH_DB_HOST/PORT/USER/PASSWORD/NAME
  - The repo auto-loads .env at runtime if present (exported env vars take precedence)
- Embedding model available for query embedding (defaults to nomic-ai/nomic-embed-text-v1.5, dim=768)
  - Ensure AUSLEGALSEARCH_EMBED_DIM in DB schema matches your embedding model dimension (default 768)


## Environment and Config

The benchmark uses the shared db.connector engine and the embedding/embedder interface, so it follows the same configuration model as the rest of the app.

- Database:
  - AUSLEGALSEARCH_DB_URL or:
    - AUSLEGALSEARCH_DB_HOST, AUSLEGALSEARCH_DB_PORT, AUSLEGALSEARCH_DB_USER, AUSLEGALSEARCH_DB_PASSWORD, AUSLEGALSEARCH_DB_NAME
  - Optional engine tuning:
    - AUSLEGALSEARCH_DB_POOL_SIZE, AUSLEGALSEARCH_DB_MAX_OVERFLOW, AUSLEGALSEARCH_DB_POOL_RECYCLE, AUSLEGALSEARCH_DB_POOL_TIMEOUT
    - AUSLEGALSEARCH_DB_STATEMENT_TIMEOUT_MS

- Embedding model:
  - AUSLEGALSEARCH_EMBED_MODEL (default nomic-ai/nomic-embed-text-v1.5)
  - AUSLEGALSEARCH_EMBED_DIM must match model dimension (e.g., 768)
  - AUSLEGALSEARCH_EMBED_REV to pin revisions; AUSLEGALSEARCH_HF_LOCAL_ONLY=1 for offline cache
  - HF_HOME for model cache location

- Session tuning in the script:
  - --probes sets ivfflat.probes (IVFFLAT)
  - --hnsw_ef sets hnsw.ef_search (HNSW, pgvector >= 0.7)
  - --use_jit toggles JIT (often off is better for tail latency on short queries)
  - --trgm_limit sets trigram similarity threshold (SELECT set_limit(value))


## Trigram shortlist optimization (performance)

Recent improvements add a shortlist-first pattern for trigram scenarios to cut latency from multi-seconds to sub-second:

- Use the trigram GIN index to build a top-N shortlist ordered by similarity
- Dedupe per document on just that shortlist, with ROW_NUMBER()
- Control candidate pool with:
  - --trgm_limit (SELECT set_limit(value), e.g., 0.35–0.45)
  - --shortlist (e.g., 200–1000)

Example (slow case made fast):
- Original pattern: window over entire embeddings table
- Fast pattern: shortlist top-N by similarity, then RN=1 per doc

Ensure indexes exist (see “Indexing and tuning notes”):

- GIN (pg_trgm) on md_title_lc
- Optionally partial GIN for WHERE md_type='case'
- BTree on md_type

Tune per corpus size and recall requirements.


## Baseline Quickstart

Run a case-law style query with metadata filters and vector probes:

```bash
python3 tools/bench_sql_latency.py --scenario baseline \
  --query "Angelides v James Stedman Hendersons" \
  --top_k 10 --runs 10 --probes 12
```

Treaty/journal style query with country membership and HNSW ef_search:

```bash
python3 tools/bench_sql_latency.py --scenario baseline \
  --query "Australia Peru investment agreement" \
  --top_k 10 --runs 10 --probes 10 --hnsw_ef 60 \
  --type treaty --subjurisdiction dfat --jurisdiction au --database ATS \
  --country "United States"
```

The tool prints:
- Latency Summary with p50 and p95 for each search type
- Top Vector Hits with distance and selected metadata
- Top FTS Hits with rank
- Top Metadata-only Hits with parsed JSONB fields
- Top Hybrid Hits (client-side rerank by doc)


## Optimized Scenarios (aligned with optimized_sql.sql)

The following scenarios mirror the optimized SQL templates in schema-post-load/optimized_sql.sql. Use these to validate latency and correctness after applying post-load DDLs (generated/expression md_* columns and indexes).

- cases_by_citation
  - Exact match on citations using md_citation or md_citations[].
  - Input: JSON array string (lowercased, normalized).
  - Example:
    ```bash
    python3 tools/bench_sql_latency.py --scenario cases_by_citation \
      --citations '["[1927] hca 34","(1927) 4 clr 12"]' \
      --runs 5
    ```

- cases_by_name_trgm
  - Trigram approximate match on case/party name (md_title_lc), optional filters.
  - Supports --trgm_limit and --shortlist to tune candidate set and latency.
  - Example:
    ```bash
    python3 tools/bench_sql_latency.py --scenario cases_by_name_trgm \
      --name "Angelides v Hendersons" --jurisdiction cth --year 1927 --court HCA \
      --runs 5 --trgm_limit 0.40 --shortlist 500
    ```

- cases_by_name_lev
  - Levenshtein edit-distance match on case/party name (slower; refinement).
  - Example:
    ```bash
    python3 tools/bench_sql_latency.py --scenario cases_by_name_lev \
      --name "Angelides v Hendersons" --max_dist 3 --jurisdiction cth --year 1927 --court HCA \
      --runs 5
    ```

- legislation_title_trgm
  - Legislation by approximate title using trigram, optional jurisdiction/year/database.
  - Example:
    ```bash
    python3 tools/bench_sql_latency.py --scenario legislation_title_trgm \
      --title "Crimes Act" --jurisdiction nsw --year 1990 --database consol_act \
      --limit 20 --runs 5
    ```

- types_title_trgm
  - Title search for specified types, e.g., treaty and journal.
  - Comma-separated --types (e.g., "treaty,journal"), supports --trgm_limit and --shortlist.
  - Example:
    ```bash
    python3 tools/bench_sql_latency.py --scenario types_title_trgm \
      --title "investment agreement" --types "treaty,journal" \
      --limit 20 --runs 5 --trgm_limit 0.40 --shortlist 1000
    ```

- ann_with_filters_doc_group
  - Vector ANN (cosine distance) with metadata filters, optional trigram approx filters, and doc-level grouping.
  - Embeds --query via embedding model. Supports --probes / --hnsw_ef / --trgm_limit.
  - Example:
    ```bash
    python3 tools/bench_sql_latency.py --scenario ann_with_filters_doc_group \
      --query "fiduciary duty in NSW" --top_k 10 --runs 5 \
      --type case --jurisdiction nsw --database NSWSC \
      --date_from 2000-01-01 --date_to 2024-12-31 \
      --author "Smith" --title_approx "fiduciary" --source "nswcaselaw" \
      --probes 12 --hnsw_ef 60
    ```

- title_search_doc_group
  - Approximate title search with doc-level grouping and optional filters.
  - Supports --trgm_limit and --shortlist to reduce candidate set.
  - Example:
    ```bash
    python3 tools/bench_sql_latency.py --scenario title_search_doc_group \
      --title "Succession Act" --type legislation --jurisdiction nsw --year 2006 \
      --limit 20 --runs 5 --trgm_limit 0.40 --shortlist 1000
    ```

- source_approx
  - Approximate match on documents.source (trigram).
  - Example:
    ```bash
    python3 tools/bench_sql_latency.py --scenario source_approx \
      --source "nsw legislation" --limit 20 --runs 5
    ```


## CLI Arguments (summary)

- --scenario baseline|cases_by_citation|cases_by_name_trgm|cases_by_name_lev|legislation_title_trgm|types_title_trgm|ann_with_filters_doc_group|title_search_doc_group|source_approx
- --query TEXT (baseline vector/hybrid; ann_with_filters_doc_group)
- --top_k INT, --runs INT, --use_jit, --probes INT, --hnsw_ef INT, --trgm_limit FLOAT
- --shortlist INT (for trigram scenarios: cases_by_name_trgm, types_title_trgm, title_search_doc_group)
- Filters: --type, --jurisdiction, --subjurisdiction, --database, --year, --date_from, --date_to, --title_eq, --author_eq, --citation, --title_member, --citation_member, --country
- Approx filters: --author (approx), --title_approx, --source or --source_approx
- Optimized parameters:
  - --citations JSON_ARRAY_STRING
  - --name TEXT
  - --max_dist INT
  - --title TEXT
  - --types CSV
  - --limit INT


## Output interpretation

For each scenario, the script prints:
- Latency Summary (p50, p95, count)
- Top Results using scenario-specific fields (doc_id, url, score/distance/rank, relevant metadata like title/date/court)

Use these to compare different index settings (HNSW ef_search, IVFFLAT probes) and to validate that post-load indexes and generated/expression columns are being used by the planner.


## Indexing and tuning notes

For high recall and low p95, build the right indexes and choose the right index type for your scale:

- HNSW (pgvector >= 0.7): better p95 for large N, control recall via --hnsw_ef
- IVFFLAT: lighter/faster builds, control recall via --probes and lists (set at build time)
- Always apply selective metadata equality/range filters (type, jurisdiction, database, year/date) to shrink candidate sets before vector ORDER BY
- Use partial indexes per cohort (e.g., md_type) when N is huge
- Trigram indexes:
  - CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_title_trgm ON embeddings USING GIN (md_title_lc gin_trgm_ops);
  - Optional partial (cases only): CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_title_trgm_case ON embeddings USING GIN (md_title_lc gin_trgm_ops) WHERE md_type='case';
  - BTree for cohort filters: CREATE INDEX IF NOT EXISTS idx_embeddings_md_type ON embeddings(md_type);

See schema-post-load/README.md and schema-post-load/optimized_sql.sql for:
- Choosing HNSW vs IVFFLAT at TB-scale, build-time memory (maintenance_work_mem), partial indexes
- Generated stored columns (md_*), expression indexes only alternative, and trigram/GIN index patterns
- Optimized SQL templates for citation/name/title/source searches, ANN + grouping, and doc-level grouping patterns


## Troubleshooting

- Error: relation column e.md_title_lc (or md_type, etc.) does not exist
  - You have not applied the post-load DDLs to add generated/expression columns and indexes
  - See schema-post-load/create_indexes.sql or schema-post-load/create_indexes_expression.sql
  - Re-run the benchmark after applying post-load scripts

- Error: pgvector not found
  - Ensure CREATE EXTENSION vector; has been run in the target database
  - Confirm db/store.create_all_tables() ran at least once

- Embedding dimension mismatch
  - AUSLEGALSEARCH_EMBED_DIM must match the embedding model’s dimension (e.g., 768 for nomic v1.5)

- Slow model downloads / repeated fetching
  - Set HF_HOME to a persistent, fast cache directory

- Tail latency is high
  - Turn JIT off (default) for short queries
  - Increase selectivity via filters (type/jurisdiction/database/year/date)
  - For IVFFLAT: increase lists at build time, probes at query time
  - For HNSW: increase hnsw.ef_search
  - Ensure ANALYZE has been run after bulk loads and index builds
  - For trigram scenarios: increase --trgm_limit (e.g., 0.35–0.45) and reduce/increase --shortlist to balance recall vs latency

- No results for FTS
  - Confirm the documents.document_fts column exists and the trigger is in place (db/store.create_all_tables() sets it up)
  - If document_fts is NULL for existing rows, run the backfill (disabled if AUSLEGALSEARCH_SCHEMA_LIGHT_INIT=1)


## Examples

Case by citation:
```bash
python3 tools/bench_sql_latency.py --scenario cases_by_citation \
  --citations '["[1927] hca 34","(1927) 4 clr 12"]' --runs 5
```

Cases by name (shortlist + trigram):
```bash
python3 tools/bench_sql_latency.py --scenario cases_by_name_trgm \
  --name "Leech v R" --jurisdiction cth --runs 3 \
  --trgm_limit 0.40 --shortlist 500
```

Types by title (shortlist + trigram):
```bash
python3 tools/bench_sql_latency.py --scenario types_title_trgm \
  --title "investment agreement" --types "treaty,journal" \
  --limit 10 --runs 3 --trgm_limit 0.40 --shortlist 1000
```

Legislation by title:
```bash
python3 tools/bench_sql_latency.py --scenario legislation_title_trgm \
  --title "Crimes Act" --jurisdiction nsw --year 1990 --database consol_act \
  --limit 20 --runs 5
```

ANN with filters + grouping:
```bash
python3 tools/bench_sql_latency.py --scenario ann_with_filters_doc_group \
  --query "fiduciary duty in NSW" --top_k 10 --runs 5 \
  --type case --jurisdiction nsw --database NSWSC \
  --date_from 2000-01-01 --date_to 2024-12-31 \
  --author "Smith" --title_approx "fiduciary" --source "nswcaselaw" \
  --probes 12 --hnsw_ef 60
