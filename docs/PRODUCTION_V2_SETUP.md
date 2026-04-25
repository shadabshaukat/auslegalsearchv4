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
