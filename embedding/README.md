# AUSLegalSearch v3 — Embedding Subsystem

Vector embedding interface and runtime for the ingestion pipeline, search, and benchmarking tools. Provides a single, ergonomic API around Sentence-Transformers models with a robust HuggingFace fallback.

Primary module
- embedding/embedder.py


## Features

- Model resolution order
  1) Explicit `Embedder(model_name=...)` argument
  2) Environment variable `AUSLEGALSEARCH_EMBED_MODEL`
  3) Default: `nomic-ai/nomic-embed-text-v1.5` (768D)

- Sentence-Transformers preferred, HF fallback:
  - Attempts to load as a Sentence-Transformers repo via `SentenceTransformer(...)`
  - If that fails (model is not an ST repo or package unavailable), falls back to HuggingFace `AutoModel` + mean pooling over token embeddings
  - Exposes `.dimension` and returns `np.ndarray` of shape `[batch, dim]`

- GPU-friendly
  - Works with CUDA when available; actual device/placement is handled by underlying libraries
  - Ingestion workers perform batched embedding with adaptive backoff on OOM

- Minimal interface
  - `vecs = Embedder().embed(["text 1", "text 2"])` → `np.ndarray` float32 vectors
  - Designed to be dependency-light at callsites; all model specifics are internal


## Environment variables

- Selection and trust
  - `AUSLEGALSEARCH_EMBED_MODEL` — Model repo or checkpoint name
  - `AUSLEGALSEARCH_TRUST_REMOTE_CODE=1` — Pass `trust_remote_code=True` to allow custom model code
  - `AUSLEGALSEARCH_EMBEDDER_FLAGS` — Freeform flags; if contains `trust_remote_code`, enables trust

- Revisions and offline caches
  - `AUSLEGALSEARCH_EMBED_REV` — Specific revision/commit to load
  - `AUSLEGALSEARCH_HF_LOCAL_ONLY=1` — Force `local_files_only=True` to use local cache only
  - `HF_HOME` — Caching directory for HF/ST models (place on fast SSD)

- Fallback maximum sequence length
  - `AUSLEGALSEARCH_EMBED_MAXLEN` (default 512) — Max token length for HF fallback (AutoTokenizer truncation)

- Critical database dimension match
  - `AUSLEGALSEARCH_EMBED_DIM` — Dimension for DB schema Vector column (default 768)
  - Must match the actual model’s embedding dimension; otherwise inserts and/or queries will fail


## Usage

Basic
```python
from embedding.embedder import Embedder

embedder = Embedder()  # uses AUSLEGALSEARCH_EMBED_MODEL or default
vecs = embedder.embed(["example legal text 1", "example legal text 2"])
print(vecs.shape)  # (2, embedder.dimension)
```

Explicit model (Sentence-Transformers repo)
```python
embedder = Embedder("maastrichtlawtech/bge-legal-en-v1.5")
vecs = embedder.embed(["..."])
```

Plain HF checkpoint (mean pooling fallback)
```python
embedder = Embedder("nlpaueb/legal-bert-base-uncased")
vecs = embedder.embed(["..."])  # .dimension inferred from model.hidden_size (e.g., 768)
```

Environment-based
```bash
export AUSLEGALSEARCH_EMBED_MODEL="nomic-ai/nomic-embed-text-v1.5"
python -c "from embedding.embedder import Embedder; e=Embedder(); print(e.dimension)"
```


## Integration points

- Ingestion workers (`ingest/beta_worker.py`)
  - Embedder is initialized once per worker process (after CPU process pool is created)
  - Batching controlled by `AUSLEGALSEARCH_EMBED_BATCH` (default 64)
  - Adaptive backoff halves batch size on OOM until it fits

- Search (`db/store.py`)
  - `search_vector` and `search_hybrid` call the `Embedder` to vectorize queries

- Bench tool (`tools/bench_sql_latency.py`)
  - Uses `Embedder` to generate query vectors for vector similarity benchmarking


## Performance guidance

- Batch size vs VRAM
  - 16 GB GPUs: 64–128
  - 24–40 GB GPUs: 128–256
- Model cache
  - Set `HF_HOME` to a fast SSD and pre-warm models to avoid cold-start latency
- Text length
  - Default HF fallback truncates at 512 tokens; adjust `AUSLEGALSEARCH_EMBED_MAXLEN` if needed
- Normalization
  - Embeddings are returned unnormalized by default; pgvector cosine distance works with raw vectors. If you need strict unit-norm, add `_l2_normalize` (already implemented and commented) where appropriate


## Troubleshooting

- “Vector dimension mismatch” in DB operations
  - Ensure `AUSLEGALSEARCH_EMBED_DIM` (used by DB schema) equals the actual model dimension reported by `Embedder().dimension`
  - Recreate DB vector column/index if you change model dimension

- Model download failures / repeated downloads
  - Confirm network/firewall; set `HF_HOME` to a persistent cache
  - Use `AUSLEGALSEARCH_HF_LOCAL_ONLY=1` to force local cache usage (requires prior download)

- Sentence-Transformers import errors
  - The fallback path uses HF `AutoModel` + mean pooling. Ensure `transformers` and `torch` are installed
  - Verify GPU drivers/CUDA compatibility with your `torch` distribution

- Slow first inference
  - The first call includes model initialization; warm-up by embedding a small batch at startup


## Notes

- Some ST models internally normalize vectors; this wrapper does not re-normalize by default to avoid double-normalization
- If you need strict normalization (e.g., for dot-product), consider re-enabling `_l2_normalize(vecs)` in the code where applicable
