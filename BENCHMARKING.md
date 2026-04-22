# AUSLegalSearch v4 — Benchmarking Notes (OpenSearch + 4x A10)

This document captures the recent remote benchmark and environment validation used to diagnose and improve ingestion performance.

---

## 1) Environment and root-cause fix

### Initial issue
- Ingestion was slow because embeddings were running on CPU, not GPU.
- Observed mismatch before fix:
  - `torch 2.11.0+cu130`
  - host driver/runtime reported CUDA 12.8
  - `torch.cuda.is_available() == False`

### Torch/CUDA remediation (applied)

```bash
cd /home/ubuntu/auslegalsearchv4
source venv/bin/activate

pip uninstall -y torch torchvision torchaudio triton || true
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

Validation after fix:
- `torch 2.11.0+cu128`
- `torch.version.cuda 12.8`
- `torch.cuda.is_available() == True`
- `torch.cuda.device_count() == 4`

---

## 2) Ingestion benchmark configuration

- Backend: OpenSearch (`AUSLEGALSEARCH_STORAGE_BACKEND=opensearch`)
- GPUs: 4 (`--gpus 4`)
- Model: `nomic-ai/nomic-embed-text-v1.5`
- Chunk params: `target=3000`, `overlap=250`, `max=3500`
- Resume enabled: `--resume`
- Key env:
  - `AUSLEGALSEARCH_CPU_WORKERS=4`
  - `AUSLEGALSEARCH_PIPELINE_PREFETCH=64`
  - `AUSLEGALSEARCH_EMBED_BATCH=64`
  - `OPENSEARCH_TUNE_INDEX=1`
  - `OS_METRICS_NDJSON=1`
  - `OS_INGEST_STATE_ENABLE=1`
  - `AUSLEGALSEARCH_WORKER_SCHEMA_INIT=0`
  - `OPENSEARCH_ENFORCE_SHARDS=0` (during tuned ingest windows)

---

## 3) Results captured

### Session: `os-bench-20260422-2358`
- elapsed: **7.16 min**
- files_ok: **62**
- chunks_indexed: **67,121**
- files/min: **8.66**
- chunks/min: **9,376.64**

### Session: `os-bench-final-20260423-0001`
- elapsed: **3.95 min**
- files_ok: **31**
- chunks_indexed: **35,934**
- files/min: **7.85**
- chunks/min: **9,101.71**

### Session: `os-bench-final-20260423-0001` (5-minute window check)
- elapsed: **4.99–5.79 min**
- files_ok: **31**
- chunks_indexed: **35,934**
- files/min: **5.35–6.21**
- chunks/min: **6,203.97–7,195.88**

### GPU utilization during benchmark snapshots
- GPU utilization observed around **84–100%** (with high memory usage across devices), confirming active GPU embedding.

---

## 4) 8-million file ETA (based on current observed throughput)

Using observed range across benchmark snapshots:
- Best observed: `8.66 files/min` → ~12,470 files/day
- Lower observed: `5.35 files/min` → ~7,704 files/day

Estimated wall-time for **8,000,000 files**:
- At 8.66 files/min: **~641 days**
- At 5.35 files/min: **~1,038 days**

> Important: this is a direct extrapolation of current measured run characteristics, which can vary materially with file-size distribution, chunk density, OpenSearch write performance, and model/runtime warm state.

---

## 5) Is OpenSearch Bulk API used?

Yes.

OpenSearch ingestion path uses `bulk_upsert_file_chunks_opensearch(...)` in `db/store.py`, which imports and uses:
- `opensearchpy.helpers.bulk`
- `opensearchpy.helpers.parallel_bulk` (when configured concurrency > 1)

So ingestion is already on bulk helper path (not single-document indexing loops for chunks).

---

## 6) OpenSearch-side improvements (OCI managed OpenSearch)

Recommended for faster ingest on managed OCI OpenSearch:

1. **Keep ingest tuning enabled during load windows**
   - `refresh_interval=-1` (or high interval)
   - replicas temporarily `0`, restore post-load.

2. **Tune bulk controls by load test**
   - `OPENSEARCH_BULK_CHUNK_SIZE` (start 500–1500)
   - `OPENSEARCH_BULK_MAX_BYTES` (50–100MB)
   - `OPENSEARCH_BULK_CONCURRENCY` (start 2–6 per worker)

3. **Use alias + rollover strategy**
   - write alias for active ingest index
   - read alias for query continuity
   - rollover by shard size/doc thresholds.

4. **Ensure shard sizing is balanced**
   - avoid tiny/oversized shards;
   - monitor merge pressure and threadpool rejections.

5. **Pre-provision cluster resources for ingest windows**
   - if OCI managed service supports scaling profile changes, increase ingest capacity during bulk windows.

---

## 7) Dockerization recommendation

Yes — containerizing helps reproducibility and avoids host-level drift in Python/torch dependencies.

Best practice for this stack:
- Pin exact image + CUDA runtime family in Dockerfile (matching driver compatibility).
- Pin Python package versions (including torch build) in lockfile/requirements.
- Keep runtime validation in startup health checks:
  - assert `torch.cuda.is_available()` and expected GPU count.

Docker does **not** replace NVIDIA driver requirements on host, but it greatly reduces environment drift and repeatability issues.
