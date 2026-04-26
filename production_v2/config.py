from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_LOADED_ENV_FILE = ""


def _simple_load_env_file(path: Path, override: bool) -> bool:
    """Minimal .env loader fallback when python-dotenv isn't installed."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return False

    loaded_any = False
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        val = v.strip()
        if not key:
            continue
        # Remove wrapping quotes
        if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
            val = val[1:-1]
        if override or os.environ.get(key) is None:
            os.environ[key] = val
        loaded_any = True
    return loaded_any


def _load_v2_env() -> str:
    """
    Load v2 env in a robust way even when process cwd is not repo root.
    Priority (strict, v2-only):
      1) AUSLEGALSEARCH_V2_ENV_FILE (absolute or cwd-relative)
      2) .env.production_v2 in cwd
      3) .env.production_v2 at project root (parent of production_v2 package)

    Note: We intentionally do NOT auto-fallback to generic .env here,
    to avoid accidental contamination from legacy OPENSEARCH_HOST=localhost.
    """
    load_dotenv = None
    try:
        from dotenv import load_dotenv as _ld  # type: ignore

        load_dotenv = _ld
    except Exception:
        load_dotenv = None

    explicit = os.environ.get("AUSLEGALSEARCH_V2_ENV_FILE", "").strip()
    cwd = Path.cwd()
    here = Path(__file__).resolve().parent
    project_root = here.parent

    candidates = []
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            candidates.append(((cwd / p).resolve(), True))
            candidates.append(((project_root / p).resolve(), True))
        else:
            candidates.append((p, True))

    candidates.extend(
        [
            ((cwd / ".env.production_v2").resolve(), True),
            ((project_root / ".env.production_v2").resolve(), True),
        ]
    )

    seen = set()
    for p, force_override in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            # Prefer v2 env file values to avoid stale service env variables.
            if load_dotenv is not None:
                load_dotenv(dotenv_path=str(p), override=bool(force_override))
            else:
                _simple_load_env_file(path=p, override=bool(force_override))
            return str(p)
    return ""

try:
    _LOADED_ENV_FILE = _load_v2_env()
except Exception:
    pass


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and str(v).strip() != "" else default


def _dequote(v: str) -> str:
    s = str(v or "").strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_localhost_url(v: str) -> bool:
    s = _dequote(v)
    if not s:
        return True
    p = urlparse(s if "://" in s else f"http://{s}")
    host = (p.hostname or "").lower()
    return host in {"", "localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class V2Settings:
    os_host: str = _dequote(_env("V2_OPENSEARCH_HOST", _env("OPENSEARCH_HOST", "http://localhost:9200")))
    os_user: str = _dequote(_env("V2_OPENSEARCH_USER", _env("OPENSEARCH_USER", "")))
    os_pass: str = _dequote(_env("V2_OPENSEARCH_PASS", _env("OPENSEARCH_PASS", "")))
    os_verify_certs: bool = _env_bool("V2_OPENSEARCH_VERIFY_CERTS", _env_bool("OPENSEARCH_VERIFY_CERTS", True))
    os_timeout: int = int(_env("V2_OPENSEARCH_TIMEOUT", _env("OPENSEARCH_TIMEOUT", "120")))
    allow_localhost_opensearch: bool = _env_bool("V2_ALLOW_LOCALHOST_OPENSEARCH", False)

    index_authorities: str = _env("V2_INDEX_AUTHORITIES", "austlii_authorities_v1")
    index_chunks_lex: str = _env("V2_INDEX_CHUNKS_LEX", "austlii_chunks_lex_v1")
    index_chunks_vec: str = _env("V2_INDEX_CHUNKS_VEC", "austlii_chunks_vec_v1")
    index_citation_graph: str = _env("V2_INDEX_CITATION_GRAPH", "austlii_citation_graph_v1")

    embed_model: str = _env("V2_EMBED_MODEL", _env("AUSLEGALSEARCH_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"))
    embed_dim: int = int(_env("V2_EMBED_DIM", _env("AUSLEGALSEARCH_EMBED_DIM", "768")))
    embed_batch: int = int(_env("V2_EMBED_BATCH", _env("AUSLEGALSEARCH_EMBED_BATCH", "64")))

    reranker_enable_default: bool = _env_bool("V2_RERANK_ENABLE_DEFAULT", True)
    reranker_model: str = _env("V2_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_n: int = int(_env("V2_RERANK_TOP_N", "50"))

    chunk_target_tokens: int = int(_env("V2_CHUNK_TARGET_TOKENS", "512"))
    chunk_overlap_tokens: int = int(_env("V2_CHUNK_OVERLAP_TOKENS", "64"))
    chunk_max_tokens: int = int(_env("V2_CHUNK_MAX_TOKENS", "640"))
    chunk_min_sentence_tokens: int = int(_env("V2_CHUNK_MIN_SENTENCE_TOKENS", "8"))
    chunk_min_chunk_tokens: int = int(_env("V2_CHUNK_MIN_CHUNK_TOKENS", "60"))

    ingest_file_workers: int = int(_env("V2_INGEST_FILE_WORKERS", "4"))
    ingest_embed_batch: int = int(_env("V2_INGEST_EMBED_BATCH", _env("V2_EMBED_BATCH", "64")))
    ingest_bulk_chunk_size: int = int(_env("V2_INGEST_BULK_CHUNK_SIZE", "800"))
    ingest_status_poll_seconds: int = int(_env("V2_INGEST_STATUS_POLL_SECONDS", "2"))
    ingest_gpu_ids: str = _env("V2_INGEST_GPU_IDS", "")  # e.g. "0,1,2,3"
    ingest_multigpu_min_texts: int = int(_env("V2_INGEST_MULTIGPU_MIN_TEXTS", "256"))
    ingest_offload_enable: bool = _env_bool("V2_INGEST_OFFLOAD_ENABLE", False)
    ingest_offload_start_cmd: str = _env("V2_INGEST_OFFLOAD_START_CMD", "")
    ingest_offload_stop_cmd: str = _env("V2_INGEST_OFFLOAD_STOP_CMD", "")
    ingest_offload_workdir: str = _env("V2_INGEST_OFFLOAD_WORKDIR", "")
    host_ingest_dir: str = _env("V2_HOST_INGEST_DIR", "")
    container_ingest_dir: str = _env("V2_CONTAINER_INGEST_DIR", "/app/data")

    api_host: str = _env("V2_API_HOST", "0.0.0.0")
    api_port: int = int(_env("V2_API_PORT", "8010"))
    api_user: str = _env("V2_API_USER", "legal_api")
    api_pass: str = _env("V2_API_PASS", "letmein")

    gradio_host: str = _env("V2_GRADIO_HOST", "0.0.0.0")
    gradio_port: int = int(_env("V2_GRADIO_PORT", "7861"))
    gradio_api_url: str = _env("V2_GRADIO_API_URL", f"http://localhost:{int(_env('V2_API_PORT', '8010'))}")


settings = V2Settings()


def loaded_env_file() -> str:
    return _LOADED_ENV_FILE


def validate_v2_runtime() -> None:
    """Fail fast for misconfigured production runtime."""
    if not settings.allow_localhost_opensearch and _is_localhost_url(settings.os_host):
        raise RuntimeError(
            "V2 OpenSearch host resolved to localhost/empty. "
            f"loaded_env_file={loaded_env_file() or '<none>'}; "
            "set AUSLEGALSEARCH_V2_ENV_FILE to your .env.production_v2 and ensure "
            "V2_OPENSEARCH_HOST is set to remote endpoint. "
            "If local dev is intentional, set V2_ALLOW_LOCALHOST_OPENSEARCH=1."
        )
