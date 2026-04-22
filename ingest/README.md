# AUSLegalSearch v3 — Ingestion (Beta) Pipeline

High-throughput, multi-GPU ingestion for legal corpora. Discovers .txt/.html, parses to text + metadata, performs semantic, token-aware chunking (dashed-header aware), embeds on GPU, and persists Documents + Embeddings to PostgreSQL with pgvector and FTS maintenance. Includes multi-GPU orchestration, resumability, per-file metrics logging, and performance tuning.

Key modules
- Orchestrator: ingest/beta_orchestrator.py
- Worker: ingest/beta_worker.py
- Scanner (sample mode): ingest/beta_scanner.py
- Semantic chunker: ingest/semantic_chunker.py
- Loaders (txt/html): ingest/loader.py
- HTML utility (used by loader): legal_html2text.py
- Embeddings: embedding/embedder.py
- Database: db/connector.py, db/store.py
- Reference doc: docs/BetaDataLoad.md (end-to-end guide)


## Capabilities

- Multi-GPU orchestration with dynamic sharding and scheduling across GPUs (auto-detect via nvidia-smi; optional explicit GPU count); size-aware shard formation and per-worker size-desc processing reduce tail latency on skewed corpora
- Per-file pipeline:
  1) Parse (.txt/.html) -> base text + file-level metadata
  2) Chunk (semantic token-aware; dashed-header aware; optional RCTS fallback)
  3) Embed (batched, GPU)
  4) Insert Document + Embedding rows; maintain FTS column/trigger
  5) Update session+file progress; append per-file metrics to success log
- Resume-friendly via EmbeddingSession and EmbeddingSessionFile rows
- Robust logging (success/error + optional structured NDJSON)
- Timeouts and fallback chunker for guaranteed coverage
- CPU/GPU pipeline: CPU parses/chunks concurrently while GPU embeds next batches


## Supported inputs

- Directory tree (root) containing:
  - .txt files with optional dashed-header metadata blocks
  - .html files (BeautifulSoup stripped text) with optional dashed-header blocks
- Dashed-header block shape:
  ```
  -----------------------------------
  key: value
  key: value
  ...
  -----------------------------------
  Body...
  ```


## Components and flow

1) Orchestrator (ingest/beta_orchestrator.py)
- Discovers files:
  - Full ingest: ingest.beta_worker.find_all_supported_files(root)
  - Sample mode: ingest/beta_scanner.py find_sample_files(root, skip_year_dirs=True) — one file per folder; prunes year directories
- Partitions into shards (default GPUs*4) and dynamically schedules them across GPUs; shards balanced by equal count or greedy by total size (auto-enabled on skew)
- Ensures DB schema (create_all_tables)
- Creates child sessions: {session}-gpu0, -gpu1, ...
- Launches one worker per child (sets CUDA_VISIBLE_DEVICES=idx). Writes partition files:
  - .beta-gpu-partition-{session}-gpu{i}.txt
- Optional wait: aggregates child success/error logs into master {session} logs with header (started/ended/duration/child count/files_ok/failed)

2) Worker (ingest/beta_worker.py)
- Pipelined mode (default): ProcessPoolExecutor for CPU Stage; main process for GPU embed + DB insert
- CPU Stage per file (with deadlines):
  - parse_file() via ingest/loader.py (parse_txt or parse_html). Extracts dashed-header metadata if present
  - derive_path_metadata(): jurisdiction_guess, rel_path, series_guess, filename, etc.
  - detect_doc_type(): heuristics for case/legislation/journal/txt
  - Chunk selection (token-aware ChunkingConfig):
    1) chunk_legislation_dashed_semantic (attach dashed-header metadata to chunk_metadata)
    2) chunk_document_semantic (heading-aware blocks -> sentence merge with overlap)
    3) Optional RCTS generic (AUSLEGALSEARCH_USE_RCTS_GENERIC=1) via LangChain RecursiveCharacterTextSplitter
    - On timeout/error: fallback naive character-window chunker for coverage
  - Compute per-file metrics (chunk_count, text_len, section_count, tokens_est_total/mean, parse_ms, chunk_ms)
- GPU Stage:
  - Batched embedder.embed() with adaptive backoff on OOM (halves batch; retries)
- DB Stage:
  - Batch insert: add Document then Embedding rows (vector ndarray rows aligned to chunks)
  - Per-call deadline and retry on transient DB errors with exponential backoff
  - Update EmbeddingSessionFile status and session progress counters
- Logging:
  - Append TSV metrics line to {child}.success.log (or filepath-only when AUSLEGALSEARCH_LOG_METRICS=0)
  - Append {child}.error.log on failures
  - Optional {child}.errors.ndjson with structured details (AUSLEGALSEARCH_ERROR_DETAILS=1; include traceback if AUSLEGALSEARCH_ERROR_TRACE=1)
  - Append small footer summary to child logs (# summary files_ok / files_failed)


## Chunking semantics (ingest/semantic_chunker.py)

- Token-aware, tokenizer-agnostic estimation; config defaults:
  - target_tokens=512, overlap_tokens=64, max_tokens=640
- Hierarchy: headings -> paragraphs -> sentences; merge to target with backward overlap; enforce max_tokens (split long sentences by tokens)
- Dashed-header aware:
  - parse_dashed_blocks() extracts repeated header+body sections
  - chunk_legislation_dashed_semantic() attaches header keys (title/regulation/author/year/citation/url/etc.) + per-chunk metadata (tokens_est, section_idx/title, chunk_idx)
  - Includes preface text before first dashed header
- Optional generic fallback: chunk_generic_rcts() using LangChain splitters (token-aware with tiktoken if available; else char-based approx)


## Environment variables

Database (db/connector.py)
- AUSLEGALSEARCH_DB_URL or:
  - AUSLEGALSEARCH_DB_HOST, AUSLEGALSEARCH_DB_PORT, AUSLEGALSEARCH_DB_USER, AUSLEGALSEARCH_DB_PASSWORD, AUSLEGALSEARCH_DB_NAME
- Pooling/tuning:
  - AUSLEGALSEARCH_DB_POOL_SIZE (10), AUSLEGALSEARCH_DB_MAX_OVERFLOW (20)
  - AUSLEGALSEARCH_DB_POOL_RECYCLE (1800s), AUSLEGALSEARCH_DB_POOL_TIMEOUT (30s)
  - AUSLEGALSEARCH_DB_STATEMENT_TIMEOUT_MS (e.g., 60000)

Embedding
- AUSLEGALSEARCH_EMBED_MODEL (default nomic-ai/nomic-embed-text-v1.5)
- AUSLEGALSEARCH_EMBED_DIM (default 768; must match model dimension)
- AUSLEGALSEARCH_EMBED_BATCH (default 64)
- AUSLEGALSEARCH_EMBED_MAXLEN (HF fallback truncation)
- HF_HOME (cache), AUSLEGALSEARCH_EMBED_REV (pin), AUSLEGALSEARCH_HF_LOCAL_ONLY=1 (offline)

Worker timeouts & pipeline
- AUSLEGALSEARCH_TIMEOUT_PARSE (60)
- AUSLEGALSEARCH_TIMEOUT_CHUNK (90)
- AUSLEGALSEARCH_TIMEOUT_EMBED_BATCH (180)
- AUSLEGALSEARCH_TIMEOUT_INSERT (120)
- AUSLEGALSEARCH_TIMEOUT_SELECT (30)
- AUSLEGALSEARCH_CPU_WORKERS (default min(8, cores-1))
- AUSLEGALSEARCH_PIPELINE_PREFETCH (default 64)
- AUSLEGALSEARCH_SORT_WORKER_FILES (default 1: process this worker’s files in descending size order to reduce tail latency)

Chunking switches
- AUSLEGALSEARCH_USE_RCTS_GENERIC=1 (optional LangChain fallback)
- AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT=1
- AUSLEGALSEARCH_FALLBACK_CHARS_PER_CHUNK (4000)
- AUSLEGALSEARCH_FALLBACK_OVERLAP_CHARS (200)
- AUSLEGALSEARCH_REGEX_TIMEOUT_MS (default 200; guards regex timeouts)

Diagnostics/logs
- AUSLEGALSEARCH_LOG_METRICS=0/1 (default 1)
- AUSLEGALSEARCH_ERROR_DETAILS=0/1 (default 1)
- AUSLEGALSEARCH_ERROR_TRACE=0/1 (default 0)
- AUSLEGALSEARCH_DEBUG_COUNTS=0/1 (default 0) — prints DB counts periodically


## Quickstart

Full ingest (auto GPU detect)
```bash
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```

Force GPU count
```bash
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-2gpu" \
  --gpus 2 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --log_dir "/abs/path/to/logs"
```

Sample/preview (one file per folder; skip year dirs)
```bash
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-sample" \
  --sample_per_folder \
  --log_dir "/abs/path/to/logs"
```

Single worker (manual GPU selection)
```bash
CUDA_VISIBLE_DEVICES=0 \
python3 -m ingest.beta_worker child-session-gpu0 \
  --root "/path/to/Data_for_Beta_Launch" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```


## Resume and checkpointing

- Per-file progress tracked in embedding_session_files (unique on session_name, filepath) with status pending/complete/error
- Re-running the exact same child session name skips completed files for that child
- For multi-GPU runs, child session names include gpu index; changing GPU count changes child names
  - For a reliable resume, keep the same GPU count to reuse child names
  - Or compute remaining files by diffing partition file vs processed logs and relaunch a new child on the remainder

Remaining-files method (illustrative; see docs/BetaDataLoad.md for full commands)
```bash
session=beta-full-YYYYMMDD-HHMMSS
child=${session}-gpu3
proj=/abs/path/auslegalsearchv3
logs="$proj/logs"
part=".beta-gpu-partition-${child}.txt"

awk -F'\t' '{print $1}' "$logs/${child}.success.log" 2>/dev/null | sed '/^#/d' > /tmp/processed_g3.txt
cat "$logs/${child}.error.log" 2>/dev/null >> /tmp/processed_g3.txt
sort -u /tmp/processed_g3.txt -o /tmp/processed_g3.txt
sort -u "$part" -o /tmp/partition_g3.txt
comm -23 /tmp/partition_g3.txt /tmp/processed_g3.txt > "$proj/.beta-gpu-partition-${child}-remaining.txt"

CUDA_VISIBLE_DEVICES=3 \
python3 -m ingest.beta_worker ${child}-r1 \
  --partition_file "$proj/.beta-gpu-partition-${child}-remaining.txt" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "$logs"
```


## Logs and metrics

Per-child logs under --log_dir:
- {child}.success.log — append-only; per-file TSV metrics lines by default (or filepath-only when AUSLEGALSEARCH_LOG_METRICS=0)
  - Example fields: filepath, chunks, text_len, strategy, target/overlap/max_tokens, type, section_count, tokens_est_total/mean, parse_ms/chunk_ms/embed_ms/insert_ms
- {child}.error.log — filepaths that failed
- {child}.errors.ndjson — structured error records (AUSLEGALSEARCH_ERROR_DETAILS=1)
- Child footers: # summary files_ok=... or files_failed=...
- Orchestrator (wait mode): master {session}.success.log and {session}.error.log with header:
  - session, started_at (UTC), ended_at (UTC), duration_sec, child_sessions, files_ok/failed


## DB schema & indices

- create_all_tables():
  - Ensures extensions: vector, pg_trgm, uuid-ossp, fuzzystrmatch
  - Creates tables: documents, embeddings (Vector(EMBEDDING_DIM)), embedding_sessions, embedding_session_files, users, chat_sessions, conversion_files
  - Adds documents.document_fts and trigger to maintain FTS
  - Builds ivfflat vector index (lists=100) unless AUSLEGALSEARCH_SCHEMA_LIGHT_INIT=1
- Post-load hardening (recommended for advanced filters and tail latency):
  - Apply schema-post-load/*.sql to add md_* generated/expression columns and indexes (incl. trigram/GIN and optional HNSW)
  - tools/bench_sql_latency.py relies on these for md_* references; otherwise falls back to JSONB where possible


## Performance tuning

- GPU memory (OOM/backoff): set AUSLEGALSEARCH_EMBED_BATCH based on VRAM (e.g., 64–128 for 16GB; 128–256 for 24–40GB)
- Chunk token sizes: larger target_tokens reduce embedding calls; common values: 1500/192/1920
- Model caching: set HF_HOME to fast SSD; warm model
- CPU workers/prefetch: default CPU_WORKERS = min(8, cores-1); PIPELINE_PREFETCH=64 keeps GPU fed
- Increase AUSLEGALSEARCH_TIMEOUT_CHUNK moderately for very long/complex documents if you see timeouts


## Troubleshooting

- DB connection fails:
  - Ensure AUSLEGALSEARCH_DB_* or AUSLEGALSEARCH_DB_URL is set; passwords with special chars should be percent-encoded in URL
  - Postgres server reachable; pgvector installed
- “Vector dimension mismatch” on insert:
  - AUSLEGALSEARCH_EMBED_DIM must match the actual model dimension (e.g., 768 for nomic v1.5)
- Too many “fallback-naive” chunks:
  - Increase AUSLEGALSEARCH_TIMEOUT_CHUNK; review input file irregularities
- Resume didn’t skip:
  - Reuse the same base session and GPU count so child names match; or use remaining-files method
- Empty files or 0-chunk outputs:
  - Success is recorded with 0 chunks; verify loader/HTML parsing and content quality
- Slow ingestion:
  - Increase CPU workers (bounded), ensure fast storage for data and HF cache, tune batch size and chunk sizes


## DB sanity checks (psql)

```sql
SELECT count(*) FROM documents;
SELECT count(*) FROM embeddings;

-- Sample chunks with dashed legislation metadata
SELECT id,
       chunk_metadata->>'title' AS title,
       chunk_metadata->>'regulation' AS regulation,
       chunk_metadata->>'chunk_id' AS chunk_id,
       (chunk_metadata->>'tokens_est')::int AS tokens_est
FROM embeddings
WHERE chunk_metadata ? 'title'
ORDER BY id DESC
LIMIT 10;

-- FTS
SELECT id, source
FROM documents
WHERE document_fts @@ plainto_tsquery('english', 'pharmaceutical')
LIMIT 20;
```


## Dynamic sharding and size balancing

On skewed corpora (a few very large files mixed with many small files), static partitions cause stragglers. The orchestrator now supports:
- Sharding: split the file list into many shards (default GPUs*4) and dynamically schedule shards across GPUs (work-stealing).
- Size-aware shard formation: greedy bin-packing by total file size; auto-enabled when size skew is high.
- Per-worker size-desc ordering: each worker processes assigned files largest-first to reduce tail latency.

Example:
```bash
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-sharded-$(date +%Y%m%d-%H%M%S)" \
  --gpus 4 --shards 16 --balance_by_size \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```
Notes:
- --shards: number of shards (0=auto GPUs*4). More shards improve load balancing at the cost of more processes.
- --balance_by_size: greedy size-based shard formation; automatically enabled when size skew (Gini) is high.
- Env AUSLEGALSEARCH_SORT_WORKER_FILES=1 sorts per-worker files by size desc. Set 0 to keep natural order.

### Partition validation and manifest

To ensure complete and unique coverage of files across shards, the orchestrator validates partitions before launching workers:
- If any file is duplicated across shards or any source file is missing from all shards, the run aborts with a RuntimeError.
- Diagnostics are written to:
  - {log_dir}/{session}.partition.validation.json — includes counts, list of duplicates, and a sample of missing files
  - {log_dir}/{session}.partition.manifest.json — always written; lists total files, per-shard file counts, and uniqueness counts
- This guarantees every source file is assigned to exactly one shard.

## CLI reference

Orchestrator (ingest/beta_orchestrator.py)
- --root (required), --session (required)
- --model (embedding model name)
- --gpus (0=auto)
- --sample_per_folder (preview mode)
- --no_skip_years_in_sample (include year dirs in sample mode)
- --target_tokens / --overlap_tokens / --max_tokens
- --log_dir
- --no_wait (do not aggregate; exit after launch)
- --balance_by_size (greedy size-balanced partitions across GPUs)
- --shards (number of shards; 0=auto GPUs*4; enables dynamic scheduling across GPUs)

Worker (ingest/beta_worker.py)
- Positional: session_name (e.g., beta-...-gpu0)
- --root or --partition_file (one required)
- --model
- --target_tokens / --overlap_tokens / --max_tokens
- --log_dir


## References

- docs/BetaDataLoad.md — comprehensive Beta Data Load Guide
- tools/bench_sql_latency.py — end-to-end SQL latency benchmark
- schema-post-load/README.md — HNSW/IVFFLAT, generated/expression columns, and TB-scale guidance
