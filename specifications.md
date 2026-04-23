# AUSLegalSearch v4 — Platform Specifications, Deep-Dive, and Scale Plan

This file is the **single source of truth** for:
- platform facts and architecture,
- current implementation behavior,
- OpenSearch parity assessment vs Postgres/Oracle paths,
- scale specifications for 1M and 8M-file ingestion/search,
- improvement roadmap,
- ongoing change tracking.

---

## 1) Platform Snapshot (Facts)

## Product purpose
AUSLegalSearch v4 is a legal-search + RAG platform for Australian legal corpora (cases, legislation, journals, treaties), with:
- ingestion + chunking + embedding,
- vector + lexical + hybrid + FTS-style retrieval,
- chat/RAG via local Ollama and OCI GenAI,
- API (FastAPI) and UIs (Streamlit/Gradio),
- session/progress tracking for large ingestion jobs.

## Runtime backend modes
Backend selected by `AUSLEGALSEARCH_STORAGE_BACKEND`:
- `postgres` (SQLAlchemy + pgvector + tsvector)
- `oracle` (Oracle endpoint support retained)
- `opensearch` (OpenSearch-first storage/retrieval/session/auth)

In `db/connector.py`, when backend=`opensearch`:
- `engine = None`
- `SessionLocal = None`

So OpenSearch mode is treated as **exclusive primary storage path** (not dual-write) in current app logic.

---

## 2) OpenSearch Index Topology (Current)

Defined in `db/opensearch_connector.py::ensure_opensearch_indexes()`.

### Indexes created
1. documents
2. embeddings (or `OPENSEARCH_INDEX` override)
3. users
4. embedding_sessions
5. embedding_session_files
6. chat_sessions
7. conversion_files
8. counters

### Important behavior
- `OPENSEARCH_INDEX` controls the **embeddings/vector index name only**.
- Other indexes use `OPENSEARCH_INDEX_PREFIX` + suffix.
- Shard/replica values are applied on create using:
  - `OPENSEARCH_NUMBER_OF_SHARDS`
  - `OPENSEARCH_NUMBER_OF_REPLICAS`
- Optional enforcement:
  - `OPENSEARCH_ENFORCE_SHARDS=1` fail-fast on mismatch
- Optional destructive reset:
  - `OPENSEARCH_FORCE_RECREATE=1` drops/recreates required indexes.

### Vector mapping
Embeddings index contains:
- `vector` as `knn_vector`
- `dimension` from `AUSLEGALSEARCH_EMBED_DIM` (or `COGNEO_EMBED_DIM` fallback)
- method config from:
  - `OPENSEARCH_KNN_ENGINE`
  - `OPENSEARCH_KNN_METHOD`
  - `OPENSEARCH_KNN_SPACE`

**Conclusion:** current design is **multi-index**, not a single universal index.

---

## 3) Chunking + Embedding + Ingestion + Search Deep-Dive (OpenSearch parity)

## 3.1 Chunking logic
Primary chunking in `ingest/semantic_chunker.py`:
- sentence/paragraph/heading-aware chunking,
- token-aware budgets (`target_tokens`, `overlap_tokens`, `max_tokens`),
- dashed-header legal block parser for legislation-heavy content,
- optional RCTS fallback (`chunk_generic_rcts`) when enabled,
- robust regex timeout hardening (`AUSLEGALSEARCH_REGEX_TIMEOUT_MS`).

This chunking module is backend-agnostic and behaves the same regardless of Postgres/OpenSearch.

## 3.2 Embedding logic
`embedding/embedder.py`:
- primary SentenceTransformers path,
- HF fallback path with mean pooling,
- model configurable via env,
- output dim tied to model config.

Embedding generation is backend-agnostic and parity is preserved.

## 3.3 Ingestion logic parity
In OpenSearch mode:
- `ingest/beta_worker.py::run_worker_opensearch()` is used.
- It performs parse -> semantic chunk -> embed -> store via `db.store` helpers.
- Session file status updates (`pending/complete/error`) are preserved.

In Postgres mode:
- `run_worker()` / `run_worker_pipelined()` includes richer timeout/retry/DB session controls.

### Parity status
- Functional parity: **mostly yes**
- Operational parity at scale: **not yet full parity** (see gaps below).

## 3.4 Search parity
OpenSearch path in `db/store.py`:
- `search_vector`: kNN on embeddings index
- `search_bm25`: multi_match over documents + embeddings
- `search_hybrid`: combines vector and lexical candidates in app layer
- `search_fts`: multi_match + highlight over docs + embeddings with dedup

Postgres path uses SQL/pgvector/tsvector and differs internally but exposes similar API contracts.

---

## 4) Findings: What works well vs key scale risks

## Working well
- Backend switch is centralized and clear.
- OpenSearch index bootstrap and shard/replica controls are present.
- Ingestion includes robust chunking strategy + fallbacks.
- Sessions/progress are persisted in OpenSearch as well.

## Key scale risks for 1M–8M files
1. **Per-document refresh-heavy indexing**
   - `add_document` and `add_embedding` use `refresh=True` for every write.
   - This will heavily throttle throughput at large scale.

2. **No bulk write path in OpenSearch ingestion worker**
   - `run_worker_opensearch()` indexes one doc/chunk at a time.
   - Bulk APIs are only used in migration tool, not ingestion pipeline.

3. **Potential duplicate text storage overhead**
   - Embeddings index stores `text`, `source`, `format` duplicated from documents index.
   - Can materially increase storage for 8M-file chunk volumes.

### Deep elaboration: duplicate storage impact and pre-ingestion fix

Current OpenSearch write path stores:
- `documents` index: canonical `content` per chunk document record
- `embeddings` index: vector + duplicated `text/source/format`

For very large ingestion, this duplication can become the dominant storage cost.

If:
- average chunk text size ~= 1.2 KB (compressed source text varies),
- total chunks ~= 200M (possible at 8M files depending chunk density),

then duplicate text alone can add hundreds of GBs to TB-scale overhead after index + segment + replication effects.

#### Recommended pre-ingestion design choices

**Option A (recommended for 8M scale): vector-first embeddings index (lean documents)**
- Keep in embeddings index:
  - `embedding_id`, `doc_id`, `chunk_index`, `vector`, minimal retrieval metadata
  - optional short snippet (e.g., first 256–512 chars) only if needed for quick previews
- Remove full duplicated `text` from embeddings index
- Keep canonical full text in documents index only
- Retrieval flow:
  1) ANN returns `doc_id/chunk_index`
  2) second fetch resolves full text from documents/chunk store

**Option B (balanced): partial denormalization**
- Keep short `text_preview` in embeddings index, not full text
- Good for UI responsiveness while containing bloat

**Option C (current): full denormalization**
- Fastest one-hop read path but highest storage + merge pressure
- Not recommended for 8M unless cluster budget is oversized

#### Concrete code-level fixes before ingestion
1. Update `db/store.py::add_embedding()` OpenSearch body:
   - stop writing full `text` into embeddings index (or make configurable via env flag)
2. Add env guard (example):
   - `OPENSEARCH_EMBED_STORE_TEXT=0|1`
3. If set to preview mode:
   - store `text_preview=text[:N]` and `content_len`
4. Ensure RAG path fetches full text by `doc_id/chunk_index` when missing in embedding hit.

#### Migration note
If data already ingested with full text duplication:
- create new lean embeddings index (new alias target),
- reindex only required fields,
- switch read alias,
- drop old heavy index after validation.

4. **Hybrid scoring normalization mismatch risk**
   - OpenSearch vector `_score` semantics are not identical to Postgres distance semantics.
   - Current normalization is simple; ranking quality can drift with corpus scale.

5. **OpenSearch worker path has lighter resilience controls than SQL path**
   - Fewer retry/deadline wrappers in `run_worker_opensearch()` than `run_worker_pipelined()`.

6. **Single-hot index lifecycle**
   - No formal rollover/ILM/data-stream strategy yet for long-running 8M ingestion.

### Deep elaboration: single-hot lifecycle risk and pre-ingestion fix

"Single-hot" means continuously writing to one large embeddings index indefinitely.
At 8M-file scale this causes:
- oversized shard growth,
- slower merges and recoveries,
- high rebalance risk,
- harder zero-downtime schema evolution.

#### Recommended lifecycle model (before 8M ingestion)

1. **Use aliases from day 1**
- `als_embeddings_write` -> one active write index
- `als_embeddings_read` -> one or many searchable indexes

2. **Rollover policy**
Trigger new backing index when any threshold is reached (example):
- max docs: 30–80M per backing index (tune by chunk size/query SLA)
- max primary shard size: 40–60GB target
- max age: optional (e.g., 7d/30d) for operational hygiene

3. **ILM states (hot/warm)**
- Hot: active writes + frequent refresh tuning
- Warm: read-heavy, no writes, lower merge pressure

4. **Ingestion-time settings**
- while bulk loading active write index:
  - `refresh_interval=-1` or high (30s)
  - replicas=0
- after each ingestion window:
  - set replicas to production value (e.g., 1)
  - set refresh interval back (e.g., 1s/5s)

5. **Schema evolution without downtime**
- create next version index with updated mapping
- dual-read via read alias during validation window
- atomically switch aliases

#### Concrete implementation tasks
1. Extend `db/opensearch_connector.py` to optionally create/manage aliases.
2. Add rollover helper tool (`tools/opensearch_rollover.py`) to:
   - create next index,
   - apply aliases,
   - optionally run `_rollover` call.
3. Add ILM policy bootstrap helper (`tools/opensearch_bootstrap_ilm.py`).
4. Add startup validation:
   - fail if write alias missing when lifecycle mode enabled.

---

## 5) Scaling Specifications for 1M files (vector + RAG)

## Target operating model
- Keep multi-index architecture.
- Keep embeddings in dedicated ANN index.
- Use OpenSearch as exclusive backend in this mode.

## Required ingest settings (high priority)
During bulk ingest window:
- `index.refresh_interval = 30s` (or `-1` during heavy backfill)
- `number_of_replicas = 0` during load, restore after ingest
- use `_bulk`/`parallel_bulk` with tuned chunk sizes
- enable HTTP compression

Recommended starting points:
- bulk chunk size: 500–1500 docs/request
- bulk bytes cap: 50MB–100MB per request
- concurrency: 2–8 workers per ingest process (cluster dependent)

## Query settings
- Use ANN candidate retrieval (kNN top-N larger than user top_k, e.g., 5x–20x), then rerank hybrid.
- Keep citation + metadata in returned payloads.

## Reliability
- Dead-letter log for failed chunk upserts.
- Idempotency key for chunk docs (`source + chunk_index + hash(text)`).

---

## 6) Scaling Specifications for 8M files (vectorized legal corpus)

At 8M files, chunk count may reach tens/hundreds of millions depending chunking density.

## Architecture recommendations
1. **Partition embeddings into multiple logical indexes (by corpus/domain/time)**
   - e.g., cases, legislation, journals, treaties, or year-bucketed sets.
   - Query via aliases for unified API.

2. **Adopt write/read alias strategy**
   - `embeddings_write` alias for ingestion target
   - `embeddings_read` alias spanning active searchable indexes
   - supports zero-downtime reindex/rollover.

3. **Implement ILM/rollover policy**
   - trigger on shard size/doc count/age.
   - prevents oversized shards and difficult recoveries.

4. **Shift hybrid ranking to two-stage retrieval**
   - Stage 1: ANN + lexical retrieve (expanded candidate pool)
   - Stage 2: rerank (cross-encoder/legal reranker) + metadata priors

5. **Control index bloat**
   - keep only needed fields in embeddings index;
   - optionally store full text only once (documents index) and join by `doc_id` at serve-time or denormalize minimally.

## Pre-ingestion gate (mandatory for 8M)

Do **not** start full ingestion until these are done:
- [ ] Embeddings index storage model decided (lean vs preview vs full text)
- [ ] Write/read aliases configured and validated
- [ ] Rollover thresholds documented and tested in staging
- [ ] Bulk indexing path enabled in ingestion worker
- [ ] Per-doc refresh removed from hot ingestion path
- [ ] Replicas/refresh ingestion profile scripted (pre/post load)
- [ ] Recovery drill validated (node restart + shard relocation + alias read continuity)

## Cluster-level guidance
- target shard size typically ~20–60GB (workload dependent)
- avoid over-sharding small indexes, avoid under-sharding huge vector indexes
- benchmark HNSW params against legal query mix (recall/latency tradeoff)

---

## 7) Immediate Improvement Backlog (Prioritized)

## P0 (must-do before very large production ingest)
- [x] Add OpenSearch **bulk ingestion writer** to `run_worker_opensearch()`.
- [x] Remove per-doc `refresh=True`; use batched refresh or periodic refresh.
- [x] Add retry/backoff/deadline wrappers in OpenSearch worker path similar to SQL pipeline.
- [x] Add idempotent document/chunk keys to avoid duplicate writes on retry/resume.

### P0 implemented in code (2026-04-22)

Implemented components:

1. `db/store.py`
- Added `bulk_upsert_file_chunks_opensearch(...)` using OpenSearch helpers bulk API.
- Added deterministic idempotent keys:
  - `doc_key = sha1(source|chunk_index|text)`
  - `embedding_key = sha1(doc_key|chunk_index)`
- Added deterministic stable numeric IDs for compatibility (`doc_id`, `embedding_id`).
- Removed forced per-write refresh for doc/embed writes:
  - controlled via `OPENSEARCH_REFRESH_ON_WRITE` (default false).
- Added lean embedding storage controls:
  - `OPENSEARCH_EMBED_STORE_TEXT` (default false)
  - `OPENSEARCH_EMBED_TEXT_PREVIEW_CHARS` (default 0)
- Added retrieval fallback to documents index when embedding hit has no full text.

2. `ingest/beta_worker.py`
- OpenSearch worker now uses **bulk upsert per file** (replacing row-by-row inserts).
- Added stage deadlines and retries for parse/chunk/embed/insert in OpenSearch mode.
- Added optional RCTS fallback + naive fallback in OpenSearch path for chunking resilience.

3. `db/opensearch_connector.py`
- Extended OpenSearch mappings for new fields used by optimized path:
  - embeddings: `embedding_key`, `doc_key`, `text_preview`
  - documents: `doc_key`, `chunk_index`

5. OpenSearch failure diagnostics + targeted retry artifacts
- `db/store.py::bulk_upsert_file_chunks_opensearch(...)` now extracts and surfaces sampled item-level OpenSearch bulk failure reasons (type/reason/caused_by) in raised errors.
- `ingest/beta_worker.py` now writes real-time retry artifacts per worker session:
  - `{session}.failed.paths.txt` (failed paths, reusable as `--partition_file`)
  - `{session}.failed.ndjson` (structured failed-file event stream with stage/error metadata)
- Added helper tool `tools/reingest_failed.py` to generate retry partition files (single or multi-shard, optional size balancing) and print worker relaunch commands.

4. Search path alignment updates
- BM25/FTS paths now consider `text_preview`.
- Vector/BM25 text enrichment now resolves missing full content via documents index.

### Operational notes for rollout
- Existing indexes created before this change may not have ideal explicit mappings for new fields.
- Best practice before 8M ingest:
  1) set `OPENSEARCH_FORCE_RECREATE=1` once in a clean environment **or** create new indexes and re-alias,
  2) verify mappings,
  3) run ingestion.

---

## 7.1 Recommended environment profile for 8M ingest windows

Set for ingestion jobs (then revert for query-serving profile):

```env
# Throughput profile
OPENSEARCH_REFRESH_ON_WRITE=0
OPENSEARCH_NUMBER_OF_REPLICAS=0
OPENSEARCH_TIMEOUT=120
OPENSEARCH_MAX_RETRIES=8

# Lean vector storage
OPENSEARCH_EMBED_STORE_TEXT=0
OPENSEARCH_EMBED_TEXT_PREVIEW_CHARS=0

# Worker resilience
AUSLEGALSEARCH_DB_MAX_RETRIES=5
AUSLEGALSEARCH_TIMEOUT_PARSE=30
AUSLEGALSEARCH_TIMEOUT_CHUNK=60
AUSLEGALSEARCH_TIMEOUT_EMBED_BATCH=180
AUSLEGALSEARCH_TIMEOUT_INSERT=120
```

After ingestion window:
- restore replicas to production level (e.g., 1),
- restore refresh profile for read latency objectives.

---

## 7.2 P1 Build Scope (next production phase)

P1 focuses on internet-scale operations for 8–10M file corpora after current P0 hardening.

### P1 objectives
1. **Alias-based read/write routing**
   - Introduce dedicated aliases:
     - `embeddings_write` (single active write target)
     - `embeddings_read` (one or multiple searchable indexes)
   - Same pattern for documents index family.

2. **Rollover + lifecycle automation**
   - Add helper tooling to create new backing indexes when thresholds hit:
     - docs count,
     - primary shard size,
     - age.
   - Attach ILM-like lifecycle policy for hot->warm behavior.

3. **Zero-downtime schema evolution**
   - New index version + alias switch strategy for mapping changes.

4. **Hybrid ranking calibration improvements**
   - Replace simple min-max blend with distribution-aware score normalization.

### Expected P1 outcomes
- Safer long-running ingestion with lower shard hot-spotting risk.
- Faster recovery/reindex operations.
- Cleaner migration path for future mapping changes.
- More stable ranking quality as corpus grows.

---

## 7.3 Large-file ingestion guidance (4MB–100MB legal documents)

For 4MB–100MB text-heavy files, bottlenecks shift from vector indexing to parse/chunk memory and single-file tail latency.

### Recommended optimizations

1. **Adaptive chunk target by file size**
- For files > 20MB, use larger target token windows with controlled overlap to reduce chunk cardinality explosion.
- Keep max token hard cap, but avoid overly small chunk target on giant files.

2. **Section-aware streaming parse/chunk**
- Parse and chunk in sections/blocks incrementally rather than materializing huge intermediate strings where possible.
- Especially for legislation/cases with repeated dashed headers.

3. **Per-file circuit breakers**
- Keep existing stage deadlines.
- Add max-chunks-per-file safety threshold (`AUSLEGALSEARCH_MAX_CHUNKS_PER_FILE`) to prevent pathological outliers from stalling throughput.

4. **GPU/CPU overlap controls**
- Maintain CPU parse/chunk pool + GPU embedding overlap.
- For very large files, allow splitting a single file into chunk-batches so embedding+bulk flush can stream instead of waiting for all chunks to finish first.

5. **Bulk tuning for large files**
- Use moderate `OPENSEARCH_BULK_CHUNK_SIZE` and `OPENSEARCH_BULK_MAX_BYTES` to avoid oversized payload retries.
- Keep `OPENSEARCH_REFRESH_ON_WRITE=0` during backfill.

6. **Memory hygiene**
- Avoid retaining full `file_chunks` arrays longer than needed; flush in sub-batches for huge files.
- Keep `AUSLEGALSEARCH_EMBED_BATCH` conservative when OOM/retries appear.

### Suggested additional env knobs (future)
- `AUSLEGALSEARCH_MAX_CHUNKS_PER_FILE` (hard guardrail)
- `AUSLEGALSEARCH_LARGE_FILE_MB_THRESHOLD`
- `AUSLEGALSEARCH_LARGE_FILE_TARGET_TOKENS`
- `AUSLEGALSEARCH_LARGE_FILE_OVERLAP_TOKENS`
- `AUSLEGALSEARCH_STREAMING_CHUNK_FLUSH_SIZE`

---

## 7.4 Metadata/header persistence verification

Current behavior (verified from loader + worker + store paths):

1. `ingest/loader.py`
- `extract_metadata_block(...)` parses dashed frontmatter key:value header and returns `chunk_metadata`.

2. `ingest/beta_worker.py`
- Merges parsed header metadata with `derive_path_metadata(...)` fields (jurisdiction guesses, path-derived context).
- Semantic chunker further enriches per-chunk metadata (section indices/titles/tokens estimate).

3. `db/store.py`
- Bulk writer persists per-chunk metadata under `chunk_metadata` in embeddings index.

Result: **header-derived metadata is preserved and indexed** in current ingestion path, including path-derived enrichment.

---

## 7.5 Current parallelization model (CPU, GPU, OpenSearch bulk)

1. **GPU parallelization**
- `ingest/beta_orchestrator.py` launches one worker process per GPU via `CUDA_VISIBLE_DEVICES`.
- Supports multi-shard scheduling across detected GPUs.

2. **CPU parallelization**
- `ingest/beta_worker.py` supports CPU process pool for parse+chunk stage with prefetch buffer.
- Controlled via:
  - `AUSLEGALSEARCH_CPU_WORKERS`
  - `AUSLEGALSEARCH_PIPELINE_PREFETCH`

3. **OpenSearch bulk write parallelization**
- `db/store.py::bulk_upsert_file_chunks_opensearch` now supports helper-level parallel bulk threading when `OPENSEARCH_BULK_CONCURRENCY>1`.
- Tuned by:
  - `OPENSEARCH_BULK_CHUNK_SIZE`
  - `OPENSEARCH_BULK_MAX_BYTES`
  - `OPENSEARCH_BULK_CONCURRENCY`
  - `OPENSEARCH_BULK_QUEUE_SIZE`
  - `OPENSEARCH_CONCURRENCY_OVERSUB`

Overall: the pipeline now has **multi-GPU + CPU-prep parallelism + configurable OpenSearch bulk parallelism**.

## 7.7 Failed-file re-ingestion workflow (implemented)

For OpenSearch ingest sessions, worker logs now support first-class retry loops:

1. During failures, worker appends structured events to:
   - `{session}.failed.ndjson`
2. Worker writes failed file path list to:
   - `{session}.failed.paths.txt`
3. Operators can:
   - re-run directly with `--partition_file <session>.failed.paths.txt`, or
   - use `python -m tools.reingest_failed --shards N --balance_by_size` for multi-GPU retry partitioning.

This closes the previous observability gap where OpenSearch bulk failures only surfaced as failure counts without per-item cause hints.

## 7.6 Benchmark snapshot (2026-04-22/23, OpenSearch backend)

Environment validated after torch/CUDA correction:
- torch: `2.11.0+cu128`
- torch CUDA runtime: `12.8`
- `torch.cuda.is_available() == True`
- GPUs visible: 4

Measured ingestion sessions:

1) `os-bench-20260422-2358`
- elapsed_min: 7.16
- files_ok: 62
- chunks_indexed: 67,121
- files_per_min: 8.66
- chunks_per_min: 9,376.64

2) `os-bench-final-20260423-0001`
- elapsed_min: 3.95
- files_ok: 31
- chunks_indexed: 35,934
- files_per_min: 7.85
- chunks_per_min: 9,101.71

GPU utilization snapshot during active ingest showed high load (84–100% across 4 GPUs), confirming GPU embedding path is functioning.

### 8M-file direct extrapolation (from measured files/min)
- ~5.35 to 8.66 files/min (across captured benchmark windows)
- Approx duration for 8,000,000 files: **~641 to ~1,038 days**

Notes:
- This is a direct throughput extrapolation; real duration depends on file-size distribution, chunk density, model behavior, and OpenSearch cluster write capacity.
- For planning, treat this as baseline before additional ingest scaling changes (bulk tuning, lifecycle/rollover, larger cluster profile, more GPUs/workers).

## P1 (high impact)
- Separate ingest index settings from query-time settings (dynamic tuning hooks).
- Add aliases + rollover support in `opensearch_connector`.
- Improve hybrid scoring calibration (normalize by distribution/percentile, not raw min-max only).

## P2 (quality + operability)
- Add parity tests for postgres vs opensearch result shape/field completeness.
- Add benchmark harness for OpenSearch ANN+hybrid latency and recall.
- Add ingestion telemetry sink (per-stage throughput, failures, retry counts).

---

## 8) OpenSearch Parity Verification Checklist

Use this checklist before each release:
- [ ] same chunk counts from chunking pipeline independent of backend
- [ ] same embedding dimension assumptions (`AUSLEGALSEARCH_EMBED_DIM`)
- [ ] ingestion session tracking and resume behavior equivalent
- [ ] vector search returns expected schema fields
- [ ] hybrid and fts routes return citations + chunk metadata
- [ ] no SQLAlchemy access attempts when backend=`opensearch`

---

## 9) Change Tracking Log (Ongoing)

> Keep appending entries here for every meaningful architecture/ingest/search change.

| Date (AEST) | Area | Change | Type | Notes |
|---|---|---|---|---|
| 2026-04-22 | Backend architecture | Added exclusive OpenSearch backend switching via `AUSLEGALSEARCH_STORAGE_BACKEND` | Implemented | SQL engine/session disabled in OpenSearch mode |
| 2026-04-22 | OpenSearch indexes | Added index bootstrap + shard/replica/env controls + enforce/recreate options | Implemented | Includes vector mapping controls |
| 2026-04-22 | Ingestion | Added OpenSearch ingestion path in `beta_worker` | Implemented | Functional parity achieved; scale optimizations pending |
| 2026-04-22 | Docs | Updated naming and environment docs to v4 + OpenSearch configs | Implemented | README/db docs updated |
| 2026-04-22 | Scale plan | Added this specification file with 1M/8M optimization roadmap | Implemented | Track future changes here |
| 2026-04-22 | P1 implementation | Added alias-based docs/embeddings routing, startup alias validation, rollover helper, and ISM bootstrap tooling | Implemented | `db/opensearch_connector.py`, `db/store.py`, `tools/opensearch_rollover.py`, `tools/opensearch_bootstrap_ilm.py` |
| 2026-04-23 | Benchmarking | Added GPU/CUDA validation + OpenSearch ingest throughput snapshot + 8M extrapolation | Implemented | See `BENCHMARKING.md` and section 7.6 |
| 2026-04-23 | OpenSearch ingest resilience | Added bulk per-item error sampling, real-time failed-file artifacts, and retry partition helper tool | Implemented | `db/store.py`, `ingest/beta_worker.py`, `tools/reingest_failed.py`, README updates |

---

## 10) Governance Rule for This File

Any future change affecting:
- chunking,
- embedding model behavior,
- ingestion throughput/reliability,
- OpenSearch index topology,
- hybrid/rerank/search logic,

**must update this file in the same PR/commit** under:
1) relevant section(s), and
2) change tracking table.
