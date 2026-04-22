# AUSLegalSearch v3 — Beta Data Load Guide

End-to-end guide to run the beta ingestion pipeline: discover files, parse, chunk (semantic or dashed-header), embed on NVIDIA GPUs, write to Postgres/pgvector, and maintain FTS. Includes multi-GPU orchestration, resume, performance tuning, and verification.

- 1) Overview
- 2) Prerequisites
- 3) Supported files and layout
- 4) Multi-GPU full ingest
- 5) Sample/preview ingest
- 6) Single-worker runs
- 7) Resume and checkpointing
- 8) Chunking: strategy, functions, visuals
  - 8A) Visual chunking decision and workflow (Mermaid)
  - 8B) Visual chunking decision and workflow (ASCII)
- 9) Configuration flags and environment variables
- 10) Logs and monitoring (with start/stop time and per-file metrics)
- 11) Performance tuning on NVIDIA GPUs
- 12) Troubleshooting
- 13) Appendix: Environment variables (quick ref)
- 14) Example end-to-end command


## 1) Overview

Per-file steps:
1. Parse: .txt/.html to text and file-level metadata.
2. Chunk:
   - Primary: semantic token-aware chunking (headings → sentences; token budgets).
   - Dashed-header aware: treat each dashed block (header key: value + body) as a section; chunk body semantically; attach header metadata to chunks.
   - Fallback on error/timeout: character-window slicer for guaranteed coverage.
   - Optional generic fallback: LangChain RecursiveCharacterTextSplitter (RCTS) for irregular files (disabled by default).
3. Embed: HuggingFace model on GPU (batched).
4. Persist: write to Postgres tables documents + embeddings (pgvector), with FTS trigger/index.
5. Track: record EmbeddingSession and EmbeddingSessionFile for status, safe resume, and logs.

Key modules:
- Orchestrator (multi-GPU): ingest/beta_orchestrator.py
- Worker (per GPU): ingest/beta_worker.py
- Chunkers (beta): ingest/semantic_chunker.py
- Embeddings: embedding/embedder.py
- DB/FTS: db/store.py, db/connector.py


## 2) Prerequisites

- Python 3.10+ virtualenv and project dependencies:
  - pip install -r requirements.txt
- Postgres with pgvector and extensions enabled.
  - db/store.py create_all_tables() ensures:
    - CREATE EXTENSION IF NOT EXISTS vector, pg_trgm, uuid-ossp, fuzzystrmatch
    - FTS: documents.document_fts, trigger function, GIN index
    - Vector indexes for similarity search
- Database env vars (example):
  - export AUSLEGALSEARCH_DB_HOST=localhost
  - export AUSLEGALSEARCH_DB_PORT=5432
  - export AUSLEGALSEARCH_DB_USER=postgres
  - export AUSLEGALSEARCH_DB_PASSWORD='YourPasswordHere'
  - export AUSLEGALSEARCH_DB_NAME=postgres
  - Or set a full URL via AUSLEGALSEARCH_DB_URL (takes precedence).
- NVIDIA GPUs and CUDA visible for embedding acceleration:
  - nvidia-smi available
  - Optional: set HF_HOME to a fast SSD for model caching.

Environment loading from .env
- The code reads configuration from environment variables via os.environ at runtime and auto-loads .env at import time (db/connector.py) from either:
  - repo root: ./.env
  - current working directory: ./.env
- Exported env variables take precedence and are not overridden by .env.

Optional: export all .env variables in the shell (useful for other tools too):
```
set -a
source .env
set +a
# verify:
env | grep AUSLEGALSEARCH_DB_
python3 -c 'import os;print(os.environ.get("AUSLEGALSEARCH_DB_PASSWORD"))'
```

Notes:
- Because the code auto-loads .env, simply running python from the repo root or CWD that contains .env is sufficient. Exporting variables is still fine and overrides .env.
- AUSLEGALSEARCH_DB_URL can be used as a single DSN string (SQLAlchemy URL). If your password has special characters, percent-encode them.


## 3) Supported files and layout

- Data root: directory tree with .txt/.html files (UTF-8).
- Supported extensions: .txt, .html
- Sample of dashed-header section (preferred for legislation/journals/treaties):
  ```
  -----------------------------------
  title: Fee for providing ATAGI advice
  regulation: 7
  chunk_id: 0
  type: legislation
  -----------------------------------
  Body text...
  ... until next dashed header or EOF ...
  ```


## 4) Multi-GPU full ingest

The orchestrator auto-detects GPUs and launches one worker per GPU, partitioning files evenly. Child sessions are created as {base_session}-gpu0, -gpu1, ...

Basic full ingest (auto GPU detect):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --log_dir "/abs/path/to/logs"
```

Specify embedding model (HuggingFace):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --log_dir "/abs/path/to/logs"
```

Explicitly set GPU count:
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-2gpu" \
  --gpus 2 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --log_dir "/abs/path/to/logs"
```

Token-aware chunk sizes (cascade to all beta chunkers):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-acts-1500tok" \
  --target_tokens 1500 \
  --overlap_tokens 192 \
  --max_tokens 1920 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --log_dir "/abs/path/to/logs"
```

Launch without waiting (fire-and-forget):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-nowait" \
  --no_wait \
  --log_dir "/abs/path/to/logs"
```


## 5) Sample/preview ingest

Pick one file per folder (skipping year directories by default) for quick dry runs:
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-sample" \
  --sample_per_folder \
  --log_dir "/abs/path/to/logs"
```

Do not skip year directories in sample mode:
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-sample-all" \
  --sample_per_folder \
  --no_skip_years_in_sample \
  --log_dir "/abs/path/to/logs"
```


## 6) Single-worker runs

Directly run a worker (e.g., to test one GPU or a single partition). You control CUDA_VISIBLE_DEVICES.

Single worker over entire root:
```
CUDA_VISIBLE_DEVICES=0 \
python3 -m ingest.beta_worker child-session-gpu0 \
  --root "/path/to/Data_for_Beta_Launch" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```

Single worker from a partition file:
```
# Partition file contains absolute file paths, one per line
CUDA_VISIBLE_DEVICES=1 \
python3 -m ingest.beta_worker child-session-gpu1 \
  --partition_file ".beta-gpu-partition-child-session-gpu1.txt" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```

Flags (beta_worker):
- Positional: session_name
- --root or --partition_file (one is required)
- --model
- --target_tokens, --overlap_tokens, --max_tokens
- --log_dir


## 7) Resume and checkpointing

Resume is supported at child-session granularity. Files already marked complete for the SAME child session are skipped.

How it works:
- Each worker records EmbeddingSessionFile rows keyed by (session_name, filepath) with status:
  - pending → complete or error
- Re-running the same session_name will skip all files with status complete.

Resuming a multi-GPU run:
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-20250101-120000" \
  --gpus 2 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```

Important notes:
- Changing GPU count changes child session names. Current skip logic does not dedupe across different child sessions; use the same GPU count to resume reliably.
- Partition files are regenerated; with the same inputs/GPU count, partitions remain consistent.


## 8) Chunking: strategy, functions, visuals

The chunking pipeline is designed for legal corpora to preserve document structure and maintain consistent chunk sizes via token budgets.

- Token budgets and defaults (ChunkingConfig):
  - target_tokens: goal per chunk size (default 512)
  - overlap_tokens: carried context across chunk boundaries (default 64)
  - max_tokens: hard cap; single long sentences are word-sliced if needed (default 640)
  - min_sentence_tokens, min_chunk_tokens: filters for over-short sentences/chunks
- Common metadata per chunk:
  - section_title, section_idx, chunk_idx, tokens_est
  - Path-derived metadata: jurisdiction_guess, rel_path, series_guess, filename, etc.
- Dashed-header path:
  - Header key:value pairs (e.g., type, title, author, year, citation, url, regulation, chunk_id, jurisdiction) are merged into chunk_metadata for the corresponding section.

Supported functions (ingest/semantic_chunker.py):
- parse_dashed_blocks
- chunk_legislation_dashed_semantic
- split_into_blocks (heading-aware)
- split_into_sentences (conservative splitter)
- chunk_text_semantic (sentences → chunks)
- chunk_document_semantic (blocks → chunk_text_semantic)
- chunk_generic_rcts (optional RCTS fallback; off by default)

Treaties and journals assessment:
- If they begin with dashed headers, they follow chunk_legislation_dashed_semantic. No special handler required beyond dashed-header awareness.
- For headed prose without dashed headers, chunk_document_semantic is used.
- Optional RCTS fallback can be enabled for generic/irregular files.

### 8A) Visual chunking decision and workflow (Mermaid)

```mermaid
flowchart TD
    A[Start per file] --> B[Parse (.txt/.html) -> text + file-level meta]
    B -->|empty| Z1[Error + log] --> End
    B --> C[base_meta := derive_path_metadata + file-level meta]
    C --> D[detect_doc_type(meta,text) for analytics]
    D --> E{Chunking}
    E -->|Try 1| F[chunk_legislation_dashed_semantic(text, base_meta, cfg)]
    F -->|chunks > 0| G[Proceed with chunks]
    F -->|no chunks| H[chunk_document_semantic(text, base_meta, cfg)]
    H -->|chunks > 0| G
    H -->|no chunks| I{AUSLEGALSEARCH_USE_RCTS_GENERIC == 1?}
    I -->|yes| J[chunk_generic_rcts(text, base_meta, cfg)]
    I -->|no| K[No chunks]
    J -->|chunks > 0| G
    J -->|no chunks| K
    K --> G0[Proceed with 0 chunks]
    G --> L[Embed (batched, GPU)]
    L --> M[Insert docs+embeddings into DB]
    M --> N[Update progress + per-file metrics logs]
    N --> End

    classDef step fill:#0b7285,stroke:#0b7285,color:#fff
    classDef alt fill:#495057,stroke:#495057,color:#fff
    class G,G0,L,M,N step
    class F,H,J alt
```

- Try 1: chunk_legislation_dashed_semantic
  - Works for any dashed-header format (not restricted to a specific type). Treaties and journals with dashed headers follow this path. Header metadata is merged into chunk_metadata. Section title defaults to header "title"/"section".
- Try 2: chunk_document_semantic
  - Heading-aware blocks (Roman numerals, 1.2.3., UPPERCASE, "Section N"), then sentence merging to target_tokens with overlap. Ensures max_tokens.
- Optional Try 3: chunk_generic_rcts
  - Enabled when AUSLEGALSEARCH_USE_RCTS_GENERIC=1 and LangChain splitters are available.
  - Uses RecursiveCharacterTextSplitter (token-aware via tiktoken if present; else character-approx).
  - If a first dashed header exists, its key:value pairs are added to metadata for all chunks. strategy="rcts-generic".
- Safety: On chunk timeout/error, worker uses a character-window fallback to guarantee coverage.

### 8B) Visual chunking decision and workflow (ASCII)

```
[beta_worker.run_worker]
    |
    v
[parse_file(.txt/.html)]  -> base_doc.text + base_doc.chunk_metadata
    |
    v
[derive_path_metadata + merge] -> base_meta
    |
    v
[detect_doc_type(meta, text)] -> analytics only
    |
    v
[Chunk selector]
    |-- Try 1: chunk_legislation_dashed_semantic(text, base_meta, cfg)
    |-- Else Try 2: chunk_document_semantic(text, base_meta, cfg)
    |-- Else Try 3 (optional): chunk_generic_rcts(text, base_meta, cfg) if AUSLEGALSEARCH_USE_RCTS_GENERIC=1
    |-- Else: 0 chunks
    |
    v
[Embedder.embed (batched, GPU)]
    |
    v
[_batch_insert_chunks -> Document + Embedding rows, FTS trigger]
    |
    v
[update_session_progress + per-file success/error logs]
On timeout/error during "chunk":
    -> fallback char-window chunker (guaranteed coverage)
```

Function map (who calls what):
```
Worker:
  parse_file(.txt|.html) -> derive_path_metadata -> detect_doc_type
  -> chunk_legislation_dashed_semantic | chunk_document_semantic | chunk_generic_rcts?
  -> Embedder.embed(batched GPU) -> _batch_insert_chunks -> update_session_progress

SemanticChunker:
  parse_dashed_blocks -> _parse_dashed_header
  split_into_blocks -> split_into_sentences -> chunk_text_semantic
  chunk_legislation_dashed_semantic / chunk_document_semantic / chunk_generic_rcts (optional)

DB & Embed:
  embedding/embedder.py (GPU batched)
  db/store.py (tables/indexes/FTS, start/update/complete session)
```


## 9) Configuration flags and environment variables

New/updated environment variables (production hardening)
- Database connector (db/connector.py):
  - AUSLEGALSEARCH_DB_POOL_SIZE (default 10)
  - AUSLEGALSEARCH_DB_MAX_OVERFLOW (default 20)
  - AUSLEGALSEARCH_DB_POOL_RECYCLE (default 1800 seconds)
  - AUSLEGALSEARCH_DB_POOL_TIMEOUT (default 30 seconds)
  - AUSLEGALSEARCH_DB_STATEMENT_TIMEOUT_MS (optional server-side timeout, e.g., 60000)
- Embedding / vector:
  - AUSLEGALSEARCH_EMBED_DIM (default 768; must match your embedding model dimension)
- Worker timeouts:
  - AUSLEGALSEARCH_TIMEOUT_INSERT (default 120 seconds, per-file DB insert deadline)
- Logging/optional features (unchanged, clarified):
  - AUSLEGALSEARCH_LOG_METRICS=1 includes per-file metrics and per-stage timings
  - AUSLEGALSEARCH_USE_RCTS_GENERIC=1 enables RCTS fallback for generic text
  - AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT=1 keeps character-window fallback enabled

Resumable ingestion and “remaining files” method
- If a single GPU child stalls:
  - Build remaining list by diffing the child’s original partition file against its success and error logs.
  - Relaunch only that child on remaining files (with a fresh child session name), leaving other GPUs untouched.
- Example commands:
  ```
  session=beta-cases-full-20251014-122557
  child=${session}-gpu3
  proj=/home/ubuntu/auslegalsearchv3/auslegalsearchv3
  logs="$proj/logs"
  part=".beta-gpu-partition-${child}.txt"

  awk -F'\t' '{print $1}' "$logs/${child}.success.log" 2>/dev/null | sed '/^#/d' > /tmp/processed_g3.txt
  cat "$logs/${child}.error.log" 2>/dev/null >> /tmp/processed_g3.txt
  sort -u /tmp/processed_g3.txt -o /tmp/processed_g3.txt

  sort -u "$part" -o /tmp/partition_g3.txt
  comm -23 /tmp/partition_g3.txt /tmp/processed_g3.txt > "$proj/.beta-gpu-partition-${child}-remaining.txt"
  wc -l "$proj/.beta-gpu-partition-${child}-remaining.txt"

  pgrep -fa "ingest.beta_worker" | grep "${child}"
  kill -TERM <PID_of_${child}>; sleep 10
  pgrep -fa "ingest.beta_worker" | grep "${child}" && kill -KILL <PID_of_${child}>

  export AUSLEGALSEARCH_TIMEOUT_PARSE=30
  export AUSLEGALSEARCH_TIMEOUT_CHUNK=60
  export AUSLEGALSEARCH_TIMEOUT_EMBED_BATCH=180
  export AUSLEGALSEARCH_TIMEOUT_INSERT=120

  CUDA_VISIBLE_DEVICES=3 \
  python3 -m ingest.beta_worker ${child}-r1 \
    --partition_file "$proj/.beta-gpu-partition-${child}-remaining.txt" \
    --model "nomic-ai/nomic-embed-text-v1.5" \
    --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
    --log_dir "$logs"
  ```

Orchestrator flags (ingest/beta_orchestrator.py):
- --root (required): dataset root directory
- --session (required): base session name; child sessions are {session}-gpu0, {session}-gpu1, ...
- --model: embedding model (e.g., "nomic-ai/nomic-embed-text-v1.5")
- --gpus: number of GPUs (0 = auto-detect)
- --sample_per_folder: pick one file per folder (preview mode)
- --no_skip_years_in_sample: in sample mode, also include year directories
- --target_tokens: token target per chunk (default 512)
- --overlap_tokens: overlapping tokens between chunks (default 64)
- --max_tokens: hard max per chunk (default 640)
- --log_dir: logs directory
- --no_wait: do not wait (no aggregation at end)

Worker flags (ingest/beta_worker.py):
- Positional: session_name (e.g., "beta-...-gpu0")
- --root or --partition_file
- --model
- --target_tokens / --overlap_tokens / --max_tokens
- --log_dir

Key environment variables (full list in Appendix):
- AUSLEGALSEARCH_USE_RCTS_GENERIC=1 (optional RCTS fallback on)
- AUSLEGALSEARCH_LOG_METRICS=0/1 (per-file metrics in child success logs; default 1)
- AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT=0/1 (default 1)
- AUSLEGALSEARCH_FALLBACK_CHARS_PER_CHUNK / AUSLEGALSEARCH_FALLBACK_OVERLAP_CHARS
- AUSLEGALSEARCH_TIMEOUT_PARSE / AUSLEGALSEARCH_TIMEOUT_CHUNK / AUSLEGALSEARCH_TIMEOUT_EMBED_BATCH
- AUSLEGALSEARCH_EMBED_BATCH (default 64)


## 10) Logs and monitoring (with start/stop time and per-file metrics)

Per-worker child logs (incremental appends):
- {child_session}.success.log
- {child_session}.error.log
- {child_session}.errors.ndjson (structured; only if AUSLEGALSEARCH_ERROR_DETAILS=1)

Orchestrator aggregated logs (when wait enabled):
- {base_session}.success.log
- {base_session}.error.log

Headers included in aggregated logs:
- started_at (UTC ISO8601), ended_at (UTC ISO8601), duration_sec
- child_sessions (count), files_ok or files_failed (counts)
- Followed by aggregated child entries

Example header in {base_session}.success.log:
```
# session=beta-full-20251014-095500
# started_at=2025-10-14T10:00:01Z
# ended_at=2025-10-14T10:37:42Z
# duration_sec=2251
# child_sessions=2
# files_ok=12345
# --- aggregated child success entries ---
/abs/path/child-0.success.log lines...
```

Per-file metrics in child success logs (light overhead):
- Controlled by AUSLEGALSEARCH_LOG_METRICS (default 1)
- TSV-style line per success:
  ```
  /abs/path/file.html    chunks=12  text_len=84219  strategy=dashed-semantic  target_tokens=1500  overlap_tokens=192  max_tokens=1920  type=legislation  section_count=37  tokens_est_total=14052  tokens_est_mean=1171
  ```
- If AUSLEGALSEARCH_LOG_METRICS=0, only the file path is logged (legacy format).
- Per-stage timing fields: parse_ms, chunk_ms, embed_ms, insert_ms
- Strategy could be: dashed-semantic | semantic | rcts-generic | fallback-naive | no-chunks

Console output includes DB target, log paths, counts, and periodic progress.

Quick DB sanity checks (psql):
```
-- counts
SELECT count(*) FROM documents;
SELECT count(*) FROM embeddings;

-- sample chunks with dashed legislation metadata
SELECT id,
       chunk_metadata->>'title' AS title,
       chunk_metadata->>'regulation' AS regulation,
       chunk_metadata->>'chunk_id' AS chunk_id,
       (chunk_metadata->>'tokens_est')::int AS tokens_est
FROM embeddings
WHERE chunk_metadata ? 'title'
ORDER BY id DESC
LIMIT 10;

-- FTS search check
SELECT id, source
FROM documents
WHERE document_fts @@ plainto_tsquery('english', 'pharmaceutical')
LIMIT 20;
```


## 11) Performance tuning on NVIDIA GPUs

- Parallelism:
  - Orchestrator launches one worker per detected GPU (or per --gpus). Each worker performs parsing/semantic chunking on CPU and embedding on the assigned GPU.
- Embedding batch size:
  - AUSLEGALSEARCH_EMBED_BATCH: increase for throughput (watch memory).
    - 16 GB GPUs: 64–128
    - 24–40 GB GPUs: 128–256
- Chunk token sizes:
  - Larger target_tokens produce fewer chunks and reduce embedding calls; balance with retrieval quality.
  - Common: --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920
- Model caching:
  - Set HF_HOME to a fast SSD; warm the model before large runs.
- Chunking timeouts:
  - Increase AUSLEGALSEARCH_TIMEOUT_CHUNK moderately for very long Acts if you see timeouts.
- Logging overhead:
  - Per-file metrics compute O(chunks) sums/means and a single file append. Embedding dominates runtime, so the impact is negligible. Set AUSLEGALSEARCH_LOG_METRICS=0 to disable entirely if desired.


## 12) Troubleshooting

- No files found:
  - Verify --root path and supported extensions (.txt/.html).
- Database errors (vector/FTS):
  - Ensure db/store.py create_all_tables() ran at least once with privileges to create extensions.
- OOM during embedding:
  - Reduce AUSLEGALSEARCH_EMBED_BATCH and/or chunk sizes.
- Slow downloads / repeated model fetches:
  - Set HF_HOME to a persistent cache directory.
- Resume didn’t skip:
  - Reuse the SAME base session and SAME GPU count so child session names match.
- Too many fallback chunks:
  - Increase AUSLEGALSEARCH_TIMEOUT_CHUNK for large/complex documents.
- Verify dashed-header metadata:
  - Query embeddings where chunk_metadata ? 'title' and inspect regulation/chunk_id fields.


## 13) Appendix: Environment variables (quick ref)

- Database:
  - AUSLEGALSEARCH_DB_HOST / AUSLEGALSEARCH_DB_PORT / AUSLEGALSEARCH_DB_USER / AUSLEGALSEARCH_DB_PASSWORD / AUSLEGALSEARCH_DB_NAME
  - AUSLEGALSEARCH_DB_URL (optional override; full SQLAlchemy URL)
- Chunking:
  - AUSLEGALSEARCH_USE_RCTS_GENERIC=0/1 (default 0)
  - AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT=0/1 (default 1)
  - AUSLEGALSEARCH_FALLBACK_CHARS_PER_CHUNK (default 4000), AUSLEGALSEARCH_FALLBACK_OVERLAP_CHARS (default 200)
  - AUSLEGALSEARCH_TIMEOUT_PARSE (default 60)
  - AUSLEGALSEARCH_TIMEOUT_CHUNK (default 90)
  - AUSLEGALSEARCH_TIMEOUT_EMBED_BATCH (default 180)
- Logging & diagnostics:
  - AUSLEGALSEARCH_LOG_METRICS=0/1 (default 1)
  - AUSLEGALSEARCH_ERROR_DETAILS=0/1 (default 1)
  - AUSLEGALSEARCH_ERROR_TRACE=0/1 (default 0)
  - AUSLEGALSEARCH_DEBUG_COUNTS=0/1 (default 0)
- Embedding:
  - AUSLEGALSEARCH_EMBED_BATCH (default 64)
  - HF_HOME, TOKENIZER_PARALLELISM (optional)


## 14) Example end-to-end command

```
export AUSLEGALSEARCH_DB_HOST=localhost
export AUSLEGALSEARCH_DB_PORT=5432
export AUSLEGALSEARCH_DB_USER=postgres
export AUSLEGALSEARCH_DB_PASSWORD='YourPasswordHere'
export AUSLEGALSEARCH_DB_NAME=postgres
export AUSLEGALSEARCH_EMBED_BATCH=128
export AUSLEGALSEARCH_LOG_METRICS=1
export HF_HOME=/fast/ssd/hf_cache
# Optional: enable RCTS fallback for irregular files
# export AUSLEGALSEARCH_USE_RCTS_GENERIC=1

python3 -m ingest.beta_orchestrator \
  --root "/home/ubuntu/Data_for_Beta_Launch" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --gpus 2 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 \
  --overlap_tokens 192 \
  --max_tokens 1920 \
  --log_dir "/home/ubuntu/auslegalsearchv3/auslegalsearchv3/logs"
```

This guide reflects the current beta workflow and code paths:
- Orchestrator: ingest/beta_orchestrator.py (aggregated logs include started_at/ended_at/duration_sec headers)
- Worker: ingest/beta_worker.py (per-file metrics in child success logs; toggle via AUSLEGALSEARCH_LOG_METRICS)
- Chunkers (beta): ingest/semantic_chunker.py (dashed-header and generic semantic; optional RCTS fallback)
- Embedding: embedding/embedder.py
- DB/FTS: db/store.py, db/connector.py
