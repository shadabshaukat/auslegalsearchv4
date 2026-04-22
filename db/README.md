# AUSLegalSearch v4 — Database Layer

Storage and retrieval layer for ingestion, search, RAG, and analytics.

- Default backend: PostgreSQL with pgvector for embeddings and FTS (tsvector) for document text.
- Optional Oracle 26ai connector for direct SQL access.
- New optional backend: OpenSearch (knn vector + text search) for document/chunk/session/auth/search flows.

Folder contents
- connector.py — SQLAlchemy engine and session factory; .env loader; pool/timeouts; pgvector extension helper
- store.py — ORM models (users, documents, embeddings, sessions, etc.), create_all_tables(), search helpers (vector/BM25/hybrid/FTS)
- oracle23ai_connector.py / oracle26ai_connector.py — Optional Oracle 26ai connector using python-oracledb for direct Oracle SQL
- test_db_setup.py — Local setup tester (if present)
- __init__.py — package marker


## Quick start

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Configure database environment
Either use a full DSN or per-field variables. The project autoloads .env from repo root or CWD, but exported env vars take precedence.

Backend selector:
```bash
# Options: postgres (default), oracle, opensearch
export AUSLEGALSEARCH_STORAGE_BACKEND=postgres
```

- Single URL (preferred when special characters in password)
```bash
export AUSLEGALSEARCH_DB_URL='postgresql+psycopg2://user:percent%40encoded%3Apass@host:5432/dbname'
```

- Or individual fields
```bash
export AUSLEGALSEARCH_DB_HOST=localhost
export AUSLEGALSEARCH_DB_PORT=5432
export AUSLEGALSEARCH_DB_USER=postgres
export AUSLEGALSEARCH_DB_PASSWORD='YourPasswordHere'
export AUSLEGALSEARCH_DB_NAME=postgres
```

OpenSearch backend environment (when `AUSLEGALSEARCH_STORAGE_BACKEND=opensearch`):
```bash
export AUSLEGALSEARCH_OS_HOST=localhost
export AUSLEGALSEARCH_OS_PORT=9200
# Optional auth
# export AUSLEGALSEARCH_OS_USER='admin'
# export AUSLEGALSEARCH_OS_PASSWORD='admin'
# TLS options
# export AUSLEGALSEARCH_OS_USE_SSL=1
# export AUSLEGALSEARCH_OS_VERIFY_CERTS=0
# Indexing options
export AUSLEGALSEARCH_OS_INDEX_PREFIX='auslegalsearch'
export AUSLEGALSEARCH_OS_SHARDS=1
export AUSLEGALSEARCH_OS_REPLICAS=0
```

3) Bootstrap schema
- Automatically: FastAPI sets AUSLEGALSEARCH_AUTO_DDL=1 by default and calls create_all_tables() on startup
- Manually (one-off)
```python
from db.store import create_all_tables
create_all_tables()
```

4) Verify
```sql
SELECT count(*) FROM documents;
SELECT count(*) FROM embeddings;
SELECT extversion FROM pg_extension WHERE extname='vector';
```


## Connection and engine (db/connector.py)

- .env loader: Reads repo-root .env or CWD .env and only sets keys not already exported; exported envs win
- URL composition: If AUSLEGALSEARCH_DB_URL unset, builds it from per-field envs with percent-encoded credentials
- Engine tuning (all configurable via env):
  - pool_pre_ping=True
  - pool_size (AUSLEGALSEARCH_DB_POOL_SIZE, default 10)
  - max_overflow (AUSLEGALSEARCH_DB_MAX_OVERFLOW, default 20)
  - pool_recycle (AUSLEGALSEARCH_DB_POOL_RECYCLE, default 1800s)
  - pool_timeout (AUSLEGALSEARCH_DB_POOL_TIMEOUT, default 30s)
  - connect_args: connect_timeout=10s, TCP keepalives, optional server-side statement timeout via AUSLEGALSEARCH_DB_STATEMENT_TIMEOUT_MS
- Session factory: SessionLocal = sessionmaker(bind=engine)
- ensure_pgvector(): CREATE EXTENSION IF NOT EXISTS vector


## Schema and models (db/store.py)

Tables
- users(id, email unique, password_hash, registered_google, google_id, name, created_at, last_login)
- documents(id, source, content, format, document_fts tsvector)
- embeddings(id, doc_id FK, chunk_index, vector Vector(EMBEDDING_DIM), chunk_metadata JSONB)
- embedding_sessions(id, session_name unique, directory, started_at, ended_at, status, last_file, last_chunk, total_files, total_chunks, processed_chunks)
- embedding_session_files(id, session_name, filepath, status, completed_at)
- chat_sessions(id UUID PK, started_at, ended_at, username, question, chat_history JSONB, llm_params JSONB)
- conversion_files(id, session_name, src_file, dst_file, status, start_time, end_time, success, error_message)

Schema bootstrap: create_all_tables()
- Enables extensions:
  - vector (pgvector), pg_trgm (trigram), "uuid-ossp", fuzzystrmatch
- Creates tables and basic constraints
- Adds/maintains FTS:
  - documents.document_fts column
  - idx_documents_fts GIN index
  - idx_documents_content_trgm GIN trigram (content)
  - documents_fts_trigger() and tsvectorupdate trigger to refresh FTS on INSERT/UPDATE
- Vector index:
  - Builds IVFFLAT index (lists=100) on embeddings.vector unless AUSLEGALSEARCH_SCHEMA_LIGHT_INIT=1 (skip heavy ops on first boot)
- Session-file unique and status indexes:
  - UNIQUE(session_name, filepath)
  - status index

Light init mode (optional)
- Set AUSLEGALSEARCH_SCHEMA_LIGHT_INIT=1 to skip FTS backfill and vector index build during first boot on fresh setups. Apply post-load scripts later (see schema-post-load/).


## Post-load indexing and metadata surfacing

For advanced filters and TB-scale performance, apply the SQL in schema-post-load/:
- create_indexes.sql — Adds GENERATED STORED md_* columns (type, jurisdiction, database, year, date, title_lc, author_lc, arrays, etc.) and builds btree/GIN/trigram indexes; may rewrite table once to materialize columns
- create_indexes_expression.sql — Expression indexes only (no table rewrite); predicates must match expressions
- optimized_sql.sql — Additional reference/tuning patterns

Important:
- tools/bench_sql_latency.py expects these md_* columns/indexes or equivalent expression indexes for optimal performance
- Choose HNSW (pgvector >= 0.7) vs IVFFLAT based on your scale and operational constraints; detailed guidance in schema-post-load/README.md


## Embedding dimension

- EMBEDDING_DIM in db/store.py uses AUSLEGALSEARCH_EMBED_DIM (default 768). It must match the embedding model used by ingestion/search (e.g., nomic v1.5 is 768D).
- If you change models with a different dimension, you must adjust AUSLEGALSEARCH_EMBED_DIM and rebuild the vector index (and potentially re-embed data).


## Search helpers

All helpers operate server-side using SQL, returning Python dicts for convenience.

- search_vector(query_vec, top_k=5)
  - Postgres backend: cosine distance over pgvector
  - OpenSearch backend: knn vector query over `knn_vector`

- search_bm25(query, top_k=5)
  - Postgres backend: ILIKE placeholder
  - OpenSearch backend: multi_match text query across content/chunk metadata

- search_hybrid(query, top_k=5, alpha=0.5)
  - Embed query with embedding.Embedder
  - Combine vector + lexical ranks with min-max style normalization, blend with alpha
  - Returns citation-friendly output including source and chunk metadata

- search_fts(query, top_k=10, mode="both")
  - Postgres backend: FTS over documents.document_fts and/or embeddings.chunk_metadata::text
  - OpenSearch backend: multi_match + highlight over document/chunk content/metadata fields


## Example: vector search (Python)

```python
from embedding.embedder import Embedder
from db.store import search_vector

e = Embedder()
qv = e.embed(["fiduciary duty in NSW"])[0]
hits = search_vector(qv, top_k=10)
for h in hits:
    print(h["doc_id"], h["score"], h["source"], (h.get("chunk_metadata") or {}).get("title"))
```


## Example: quick SQL sanity checks (psql)

```sql
-- Basic counts
SELECT count(*) FROM documents;
SELECT count(*) FROM embeddings;

-- Check FTS exists and works
SELECT id, source
FROM documents
WHERE document_fts @@ plainto_tsquery('english', 'pharmaceutical')
LIMIT 10;

-- Recently inserted chunks with dashed-header fields
SELECT e.doc_id, e.chunk_index,
       e.chunk_metadata->>'title'     AS title,
       e.chunk_metadata->>'regulation' AS regulation,
       (e.chunk_metadata->>'tokens_est')::int AS tokens_est
FROM embeddings e
WHERE e.chunk_metadata ? 'title'
ORDER BY e.id DESC
LIMIT 10;
```


## Oracle 26ai connector (optional)

db/oracle26ai_connector.py provides a lightweight connector for Oracle 26ai databases for direct SQL calls, used by FastAPI endpoint /db/oracle26ai_query.

Environment/params:
- ORACLE_DB_USER, ORACLE_DB_PASSWORD, ORACLE_DB_DSN
- ORACLE_WALLET_LOCATION (Autonomous DB; sets TNS_ADMIN)

Usage (Python)
```python
from db.oracle26ai_connector import Oracle26AIConnector

conn = Oracle26AIConnector()  # or pass user/password/dsn/wallet_location
rows = conn.run_query("SELECT 1 AS ok FROM dual")
print(rows)
conn.close()
```

FastAPI endpoint (fastapi_app.py)
- POST /db/oracle26ai_query with JSON:
  - { "user": "...", "password": "...", "dsn": "...", "wallet_location": "...", "sql": "SELECT ...", "params": [] }
- Backward-compatible legacy route remains available (hidden from schema):
  - POST /db/oracle23ai_query


## Environment variables summary

Database core
- AUSLEGALSEARCH_STORAGE_BACKEND (postgres|oracle|opensearch)
- AUSLEGALSEARCH_DB_URL or:
  - AUSLEGALSEARCH_DB_HOST / AUSLEGALSEARCH_DB_PORT / AUSLEGALSEARCH_DB_USER / AUSLEGALSEARCH_DB_PASSWORD / AUSLEGALSEARCH_DB_NAME

OpenSearch core
- AUSLEGALSEARCH_OS_HOST / AUSLEGALSEARCH_OS_PORT
- AUSLEGALSEARCH_OS_USER / AUSLEGALSEARCH_OS_PASSWORD (optional)
- AUSLEGALSEARCH_OS_USE_SSL / AUSLEGALSEARCH_OS_VERIFY_CERTS
- AUSLEGALSEARCH_OS_INDEX_PREFIX / AUSLEGALSEARCH_OS_SHARDS / AUSLEGALSEARCH_OS_REPLICAS

Pool/timeouts
- AUSLEGALSEARCH_DB_POOL_SIZE (10)
- AUSLEGALSEARCH_DB_MAX_OVERFLOW (20)
- AUSLEGALSEARCH_DB_POOL_RECYCLE (1800)
- AUSLEGALSEARCH_DB_POOL_TIMEOUT (30)
- AUSLEGALSEARCH_DB_STATEMENT_TIMEOUT_MS (optional server-side GUC)

Schema bootstrap
- AUSLEGALSEARCH_SCHEMA_LIGHT_INIT=1 (skip heavy index/backfill during create_all_tables)

Embedding dimension
- AUSLEGALSEARCH_EMBED_DIM (default 768) — must match model used


## Operations and maintenance

- After large ingests or index builds:
  - ANALYZE public.embeddings;
  - ANALYZE public.documents;

- If using post-load scripts:
  - For HNSW builds at scale: increase maintenance_work_mem during CREATE INDEX (see schema-post-load/README.md)
  - For IVFFLAT: choose lists based on cohort size (sqrt(N) rule of thumb), tune ivfflat.probes per route

- Vacuum/bloat management:
  - Use autovacuum defaults or tune for your write rate
  - For heavy deletes, consider pg_repack if online compaction is needed


## Troubleshooting

- Missing pgvector extension
  - Run CREATE EXTENSION vector; or ensure your DB image supports it

- “Vector dimension mismatch”
  - Check AUSLEGALSEARCH_EMBED_DIM vs actual model dimension

- Statement timeout errors
  - Set AUSLEGALSEARCH_DB_STATEMENT_TIMEOUT_MS (e.g., 60000) to bound long queries
  - Improve index coverage (post-load scripts) and query selectivity

- Slow FTS results or no matches
  - Ensure FTS column and trigger exist; if backfill was skipped in light init, run the backfill (UPDATE documents SET document_fts=...) once

- JSONB filter performance poor
  - Apply post-load scripts to surface md_* columns or build expression indexes
  - Use selective equality/range filters (type/jurisdiction/database/year/date) to shrink candidate set before vector ORDER BY


## References

- schema-post-load/README.md — TB-scale indexing strategies, HNSW vs IVFFLAT, generated columns vs expression indexes
- tools/bench_sql_latency.py — p50/p95 latency measurements and top hits dump; use to tune routes
- fastapi_app.py — API endpoints for vector/hybrid/FTS search, Oracle connector, agentic chat/RAG
