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
- `.env.production_v2.example` — dedicated environment template

## Index model (new v2)

- `austlii_authorities_v1`
- `austlii_chunks_lex_v1`
- `austlii_chunks_vec_v1`
- `austlii_citation_graph_v1`

No backfill is required. Ingest from scratch into the new index family.

## Quick start

1. Create v2 env file

```bash
cp .env.production_v2.example .env.production_v2
```

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

## Production-grade Docker deployment (self-contained)

This repo now includes a full v2 container stack:

- `Dockerfile.production_v2` (CUDA runtime + all `requirements.txt` packages)
- `docker-compose.production_v2.yml` (API + Gradio in one service, persistent mounts, GPU enabled)
- `docker/production_v2/entrypoint.sh` (supervises FastAPI + Gradio, both bind to `0.0.0.0`)
- helper scripts:
  - `scripts/v2_docker_build.sh`
  - `scripts/v2_docker_start.sh`
  - `scripts/v2_docker_stop.sh`
  - `scripts/v2_docker_logs.sh`

### Prerequisites

- Docker Engine + Docker Compose plugin
- NVIDIA drivers + NVIDIA Container Toolkit (for GPU passthrough)

### Build and run

```bash
cp .env.production_v2.example .env.production_v2

# Build image
./scripts/v2_docker_build.sh

# Start containerized v2 stack
./scripts/v2_docker_start.sh

# Follow logs
./scripts/v2_docker_logs.sh

# Stop
./scripts/v2_docker_stop.sh
```

### Raw Docker Compose commands (equivalent)

```bash
docker compose -f docker-compose.production_v2.yml build --pull
docker compose -f docker-compose.production_v2.yml up -d
docker compose -f docker-compose.production_v2.yml logs -f --tail=200
docker compose -f docker-compose.production_v2.yml down
```

### Networking and resources

- API exposed on `0.0.0.0:8010`
- Gradio exposed on `0.0.0.0:7861`
- Container uses host GPU(s) via `gpus: all`
- CPU/memory scheduling remains host-native via Docker runtime; tune with compose resource options if needed.

### Persistent storage mounts

- `./logs -> /app/logs`
- `./db -> /app/db`
- `./data -> /app/data`
- `./cache/huggingface -> /root/.cache/huggingface`
- `./cache/torch -> /root/.cache/torch`
- `./cache/gradio -> /root/.gradio`

Ingestion corpus mount is configurable:

- host path: `V2_HOST_INGEST_DIR` (default `./data`)
- container path: `V2_CONTAINER_INGEST_DIR` (default `/app/data`)

When running ingestion from Gradio/API in Docker, use the **container path** (e.g. `/app/data`) as `root_dir`.
If you pass a host-only path (e.g. `/home/ubuntu/...`) that is not mounted, ingestion will fail fast with clear error.

These preserve model caches, app state, and runtime outputs across restarts/redeployments.

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
