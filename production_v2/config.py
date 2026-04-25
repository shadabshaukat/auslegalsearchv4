from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv(os.environ.get("AUSLEGALSEARCH_V2_ENV_FILE", ".env.production_v2"))
except Exception:
    pass


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and str(v).strip() != "" else default


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class V2Settings:
    os_host: str = _env("V2_OPENSEARCH_HOST", _env("OPENSEARCH_HOST", "http://localhost:9200"))
    os_user: str = _env("V2_OPENSEARCH_USER", _env("OPENSEARCH_USER", ""))
    os_pass: str = _env("V2_OPENSEARCH_PASS", _env("OPENSEARCH_PASS", ""))
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
