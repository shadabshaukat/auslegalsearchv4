# AUSLegalSearch Production v2 (Parallel Codebase)

This is a **parallel v2 implementation** and does not modify existing v1/beta runtime paths.

## What was added

- `production_v2/config.py` — isolated config loader (`.env.production_v2`)
- `production_v2/opensearch_v2.py` — OpenSearch v2 client + 3-index bootstrap
- `production_v2/ingest_v2.py` — ingest-from-scratch pipeline (parse/chunk/embed/index)
- `production_v2/dsl_templates.py` — scenario DSL builders for 20 capability types
- `production_v2/search_v2.py` — query routing + lexical/vector + RRF fusion + reranker stage + citation graph enrichment
- `fastapi_app_v2.py` — new API layer for v2
- `gradio_app_v2.py` — new Gradio UI for v2
- `.env.production_v2` — dedicated production environment file

## Index model (new v2)

- `austlii_authorities_v1`
- `austlii_chunks_lex_v1`
- `austlii_chunks_vec_v1`
- `austlii_citation_graph_v1`

No backfill is required. Ingest from scratch into the new index family.

## Quick start

1. Ensure `.env.production_v2` exists and is configured for your deployment.

2. Start API v2

```bash
AUSLEGALSEARCH_V2_ENV_FILE=.env.production_v2 python3 -m uvicorn fastapi_app_v2:app --host 0.0.0.0 --port 8010
```

If `uvicorn` is missing in your environment:

```bash
python3 -m pip install uvicorn
```

3. Start Gradio v2

```bash
AUSLEGALSEARCH_V2_ENV_FILE=.env.production_v2 python3 gradio_app_v2.py
```

4. Bootstrap indexes (API)

```bash
curl -u legal_api:letmein -X POST "http://localhost:8010/v2/indexes/bootstrap"
```

Hard reset indexes (delete + recreate):

```bash
curl -u legal_api:letmein -X POST "http://localhost:8010/v2/indexes/recreate"
```

### Troubleshooting: API tries `localhost:9200` instead of your cluster

Check effective runtime config loaded by API:

```bash
curl -u legal_api:letmein "http://localhost:8010/v2/config/effective"
```

Verify:
- `loaded_env_file` points to your intended `.env.production_v2`
- `opensearch_host` is your remote OpenSearch endpoint (not `http://localhost:9200`)

Recommended API launch from repo root:

```bash
cd /home/ubuntu/auslegalsearchv4
AUSLEGALSEARCH_V2_ENV_FILE=/home/ubuntu/auslegalsearchv4/.env.production_v2 \
python3 -m uvicorn fastapi_app_v2:app --host 0.0.0.0 --port 8010
```

Pre-flight check in the same shell (before launching API):

```bash
cd /home/ubuntu/auslegalsearchv4
export AUSLEGALSEARCH_V2_ENV_FILE=/home/ubuntu/auslegalsearchv4/.env.production_v2
python3 - <<'PY'
from production_v2.config import settings, loaded_env_file
print("loaded_env_file=", loaded_env_file())
print("V2_OPENSEARCH_HOST=", settings.os_host)
PY
```

If this still prints `http://localhost:9200`, check:
- file path is correct and readable by the runtime user
- no typo in variable name (`V2_OPENSEARCH_HOST`)
- process was restarted after env changes
- ensure your runtime is using updated v2 code (this version does **not** auto-load generic `.env`)
- ensure `python-dotenv` is installed, or rely on built-in fallback parser in latest v2 code

Install dotenv explicitly (recommended):

```bash
python3 -m pip install python-dotenv
```

Important behavior:
- v2 config now only loads `.env.production_v2` (or file from `AUSLEGALSEARCH_V2_ENV_FILE`).
- This prevents legacy `.env` values like `OPENSEARCH_HOST=localhost` from polluting v2.
- v2 fails fast on localhost OpenSearch unless `V2_ALLOW_LOCALHOST_OPENSEARCH=1`.

If using `systemd`, ensure the service is restarted:

```bash
sudo systemctl daemon-reload
sudo systemctl restart auslegalsearch-v2-api
sudo systemctl status auslegalsearch-v2-api --no-pager
```

5. Run ingestion from scratch

```bash
curl -u legal_api:letmein -X POST "http://localhost:8010/v2/ingest/run" \
  -H 'Content-Type: application/json' \
  -d '{"root_dir":"/abs/path/to/Data_for_Beta_Launch","include_html":true}'
```

For large corpora, prefer async ingestion endpoints (prevents UI/API timeout):

```bash
# Start job
curl -u legal_api:letmein -X POST "http://localhost:8010/v2/ingest/start" \
  -H 'Content-Type: application/json' \
  -d '{"root_dir":"/abs/path/to/Data_for_Beta_Launch","include_html":true}'

# Check job status
curl -u legal_api:letmein "http://localhost:8010/v2/ingest/status/<job_id>"

# Request stop/cancel for running job
curl -u legal_api:letmein -X POST "http://localhost:8010/v2/ingest/stop/<job_id>"

# List recent jobs / latest job (for resume after UI refresh)
curl -u legal_api:letmein "http://localhost:8010/v2/ingest/jobs?limit=20"
curl -u legal_api:letmein "http://localhost:8010/v2/ingest/jobs/latest"
```

6. Run search

```bash
curl -u legal_api:letmein -X POST "http://localhost:8010/v2/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Fair Work Act 2009 s 351","scenario":"legislation","top_k":10,"use_hybrid":true,"use_reranker":true,"rerank_top_n":50}'
```

7. Citation tracing example (uses citation graph enrichment)

```bash
curl -u legal_api:letmein -X POST "http://localhost:8010/v2/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"[2025] HCA 12","scenario":"citation_tracing","top_k":10,"use_hybrid":true,"use_reranker":true}'
```

## Local VM stack (Docker removed)

This build now runs fully local on VM (no Docker runtime for v2).

### Start/stop local v2 services

```bash
# Start FastAPI v2 + Gradio v2
bash run_legalsearch_v2_stack.sh

# Stop both services
bash stop_legalsearch_v2_stack.sh
```

Note: v1 scripts are unchanged and remain separate:
- `run_legalsearch_stack.sh`
- `stop_legalsearch_stack.sh`

Service endpoints:
- FastAPI: `http://localhost:8010`
- Gradio: `http://localhost:7861`

Logs:
- `logs/v2-fastapi.log`
- `logs/v2-gradio.log`

### Local ingestion helper script (path input)

Use the helper to submit async ingestion with a VM path:

```bash
bash scripts/v2_local_ingest.sh /home/ubuntu/auslegalsearchv4/sample-data-austlii-all-file-types 0 true
```

Arguments:
- arg1: `root_dir` (required, absolute VM path)
- arg2: `limit_files` (optional, `0` means no limit)
- arg3: `include_html` (optional, `true|false`)

### GPU acceleration (PyTorch/CUDA)

Set in `.env.production_v2`:

```env
AUSLEGALSEARCH_EMBED_USE_CUDA=1
AUSLEGALSEARCH_EMBED_AMP=1
V2_INGEST_GPU_IDS=auto
V2_INGEST_AUTO_DETECT_GPUS=1
```

The embedding path uses CUDA when available and shards embedding work across configured GPU IDs in `production_v2/ingest_v2.py`.

### Ingestion governor/autotuning (small + large datasets)

v2 now includes beta-style robustness controls:

```env
V2_INGEST_GOVERNOR_ENABLE=1
V2_INGEST_BULK_RETRIES=3
V2_INGEST_EMBED_BATCH_MIN=16
V2_INGEST_EMBED_BATCH_MAX=256
V2_INGEST_BULK_CHUNK_MIN=200
V2_INGEST_BULK_CHUNK_MAX=2000
V2_INGEST_DYNAMIC_SHARDING_MIN_FILES=64
```

Behavior summary:
- Auto-detect GPUs when `V2_INGEST_GPU_IDS=auto`.
- Avoid dynamic shard fanout for tiny datasets (faster startup, less overhead).
- Adapt embed batch size by workload size and GPU count.
- Retry/reduce OpenSearch bulk chunk size on transient pressure (429/timeouts).

## Notes

- v2 keeps existing platform untouched.
- v2 is now accuracy-first by default with optional reranker stage.
- Citation graph index is included and exposed in `/v2/search` response under `citation_graph`.
- If reranker model packages are unavailable, search gracefully falls back to fused ranking only.

## Accuracy-first notes

- Hybrid retrieval (lexical + vector) is enabled by default in v2 search flow.
- RRF fusion combines exact legal signal with semantic recall.
- Cross-encoder reranker is applied to top candidates (`V2_RERANK_TOP_N`) to improve precision.
- Citation tracing scenarios include graph-edge enrichment for authority-to-authority traversal support.

## Ingestion performance + quality recommendations (AustLII scale)

- Use async ingestion (`/v2/ingest/start`) for long-running ingest.
- Start tuning with:
  - `V2_INGEST_FILE_WORKERS=4` (increase to 6–8 on CPU-rich hosts)
  - `V2_INGEST_EMBED_BATCH=64` (raise to 96/128 if GPU memory allows)
  - `V2_INGEST_BULK_CHUNK_SIZE=800` (raise to ~1200 if OpenSearch cluster can absorb)
- For true multi-GPU embedding sharding:
  - set `V2_INGEST_GPU_IDS=0,1,2,3` (example)
  - keep `V2_INGEST_MULTIGPU_MIN_TEXTS=256` (or raise if process spin-up overhead is high)
- Keep semantic chunking defaults near legal-safe values:
  - target 512 / overlap 64 / max 640
  - min sentence 8 / min chunk 60
- Embeddings for legal accuracy:
  - Recommended legal-specialized candidate: `maastrichtlawtech/bge-legal-en-v1.5`
  - Stable baseline: `nomic-ai/nomic-embed-text-v1.5`
  - Ensure `V2_EMBED_DIM` matches model output dimension before ingest.
