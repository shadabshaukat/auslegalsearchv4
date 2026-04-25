from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_LOADED_ENV_FILE = ""


def _load_v2_env() -> str:
    """
    Load v2 env in a robust way even when process cwd is not repo root.
    Priority:
      1) AUSLEGALSEARCH_V2_ENV_FILE (absolute or cwd-relative)
      2) .env.production_v2 in cwd
      3) .env.production_v2 at project root (parent of production_v2 package)
      4) .env in cwd / project root (fallback)
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return ""

    explicit = os.environ.get("AUSLEGALSEARCH_V2_ENV_FILE", "").strip()
    cwd = Path.cwd()
    here = Path(__file__).resolve().parent
    project_root = here.parent

    candidates = []
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            candidates.append((cwd / p).resolve())
            candidates.append((project_root / p).resolve())
        else:
            candidates.append(p)

    candidates.extend(
        [
            (cwd / ".env.production_v2").resolve(),
            (project_root / ".env.production_v2").resolve(),
            (cwd / ".env").resolve(),
            (project_root / ".env").resolve(),
        ]
    )

    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            # Do not clobber already-exported env vars from process manager.
            load_dotenv(dotenv_path=str(p), override=False)
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


@dataclass(frozen=True)
class V2Settings:
    os_host: str = _dequote(_env("V2_OPENSEARCH_HOST", _env("OPENSEARCH_HOST", "http://localhost:9200")))
    os_user: str = _dequote(_env("V2_OPENSEARCH_USER", _env("OPENSEARCH_USER", "")))
    os_pass: str = _dequote(_env("V2_OPENSEARCH_PASS", _env("OPENSEARCH_PASS", "")))
    os_verify_certs: bool = _env_bool("V2_OPENSEARCH_VERIFY_CERTS", _env_bool("OPENSEARCH_VERIFY_CERTS", True))
    os_timeout: int = int(_env("V2_OPENSEARCH_TIMEOUT", _env("OPENSEARCH_TIMEOUT", "120")))

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
