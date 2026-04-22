"""
OpenSearch connector and index bootstrap helpers for AUSLegalSearch v3.

This module is intentionally lightweight and optional. It only activates when
`AUSLEGALSEARCH_STORAGE_BACKEND=opensearch`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Any


def _env(*names: str, default: str = "") -> str:
    for name in names:
        v = os.environ.get(name)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    v = _env(*names, default="1" if default else "0").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


@lru_cache(maxsize=1)
def get_opensearch_client():
    try:
        from opensearchpy import OpenSearch  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenSearch backend requested but opensearch-py is not installed. "
            "Install requirements and retry."
        ) from e

    host_uri = _env("OPENSEARCH_HOST", default="")
    use_ssl = _env_bool("OPENSEARCH_USE_SSL", "AUSLEGALSEARCH_OS_USE_SSL", default=False)
    if host_uri:
        # accepts host:port OR https://host:port
        if "://" in host_uri:
            from urllib.parse import urlparse
            p = urlparse(host_uri)
            host = p.hostname or "localhost"
            port = int(p.port or (443 if p.scheme == "https" else 9200))
            use_ssl = (p.scheme == "https")
        else:
            hp = host_uri.replace("/", "")
            if ":" in hp:
                host, port_s = hp.rsplit(":", 1)
                port = int(port_s)
            else:
                host, port = hp, 9200
    else:
        host = _env("AUSLEGALSEARCH_OS_HOST", default="localhost")
        port = int(_env("AUSLEGALSEARCH_OS_PORT", default="9200"))

    user = _env("OPENSEARCH_USER", "AUSLEGALSEARCH_OS_USER", default="")
    password = _env("OPENSEARCH_PASS", "AUSLEGALSEARCH_OS_PASSWORD", default="")
    verify_certs = _env_bool("OPENSEARCH_VERIFY_CERTS", "AUSLEGALSEARCH_OS_VERIFY_CERTS", default=False)
    http_compress = _env_bool("OPENSEARCH_HTTP_COMPRESS", default=True)
    timeout_sec = int(_env("OPENSEARCH_TIMEOUT", "AUSLEGALSEARCH_OS_TIMEOUT", default="30"))
    max_retries = int(_env("OPENSEARCH_MAX_RETRIES", default="3"))

    auth = (user, password) if user else None

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=auth,
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        http_compress=http_compress,
        timeout=timeout_sec,
        max_retries=max_retries,
        retry_on_timeout=True,
    )


def index_name(suffix: str) -> str:
    # For embedding/vector index, allow explicit single index name
    if suffix == "embeddings":
        explicit = _env("OPENSEARCH_INDEX", default="")
        if explicit:
            return explicit
    prefix = _env("OPENSEARCH_INDEX_PREFIX", "AUSLEGALSEARCH_OS_INDEX_PREFIX", default="auslegalsearch")
    return f"{prefix}_{suffix}"


def _embedding_dim() -> int:
    return int(_env("COGNEO_EMBED_DIM", "AUSLEGALSEARCH_EMBED_DIM", default="768"))


def _knn_engine() -> str:
    return _env("OPENSEARCH_KNN_ENGINE", default="lucene")


def _knn_method() -> str:
    return _env("OPENSEARCH_KNN_METHOD", default="hnsw")


def _knn_space() -> str:
    return _env("OPENSEARCH_KNN_SPACE", default="cosinesimil")


def _index_body_for(name: str) -> Dict[str, Any]:
    dim = _embedding_dim()
    if name == index_name("embeddings") or name.endswith("_embeddings"):
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": int(_env("OPENSEARCH_NUMBER_OF_SHARDS", "AUSLEGALSEARCH_OS_SHARDS", default="1")),
                    "number_of_replicas": int(_env("OPENSEARCH_NUMBER_OF_REPLICAS", "AUSLEGALSEARCH_OS_REPLICAS", default="0")),
                }
            },
            "mappings": {
                "properties": {
                    "embedding_id": {"type": "long"},
                    "doc_id": {"type": "long"},
                    "chunk_index": {"type": "integer"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": _knn_method(),
                            "space_type": _knn_space(),
                            "engine": _knn_engine(),
                        },
                    },
                    "chunk_metadata": {"type": "object", "enabled": True},
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "format": {"type": "keyword"},
                }
            },
        }

    if name.endswith("_documents"):
        return {
            "settings": {
                "index": {
                    "number_of_shards": int(_env("OPENSEARCH_NUMBER_OF_SHARDS", "AUSLEGALSEARCH_OS_SHARDS", default="1")),
                    "number_of_replicas": int(_env("OPENSEARCH_NUMBER_OF_REPLICAS", "AUSLEGALSEARCH_OS_REPLICAS", default="0")),
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "long"},
                    "source": {"type": "keyword"},
                    "format": {"type": "keyword"},
                    "content": {"type": "text"},
                    "created_at": {"type": "date"},
                }
            },
        }

    # Generic mappings for auth/sessions/chat/conversion metadata.
    return {
        "settings": {
            "index": {
                "number_of_shards": int(_env("OPENSEARCH_NUMBER_OF_SHARDS", "AUSLEGALSEARCH_OS_SHARDS", default="1")),
                "number_of_replicas": int(_env("OPENSEARCH_NUMBER_OF_REPLICAS", "AUSLEGALSEARCH_OS_REPLICAS", default="0")),
            }
        },
        "mappings": {"dynamic": True},
    }


def _validate_shards_replicas(client, idx: str) -> None:
    if not _env_bool("OPENSEARCH_ENFORCE_SHARDS", default=False):
        return
    desired_s = int(_env("OPENSEARCH_NUMBER_OF_SHARDS", "AUSLEGALSEARCH_OS_SHARDS", default="1"))
    desired_r = int(_env("OPENSEARCH_NUMBER_OF_REPLICAS", "AUSLEGALSEARCH_OS_REPLICAS", default="0"))
    st = client.indices.get_settings(index=idx)
    cfg = (st.get(idx) or {}).get("settings", {}).get("index", {})
    got_s = int(cfg.get("number_of_shards", desired_s))
    got_r = int(cfg.get("number_of_replicas", desired_r))
    if got_s != desired_s or got_r != desired_r:
        raise RuntimeError(
            f"Index {idx} shard/replica mismatch: got {got_s}/{got_r}, expected {desired_s}/{desired_r}."
        )


def ensure_opensearch_indexes() -> None:
    client = get_opensearch_client()
    required = [
        index_name("documents"),
        index_name("embeddings"),
        index_name("users"),
        index_name("embedding_sessions"),
        index_name("embedding_session_files"),
        index_name("chat_sessions"),
        index_name("conversion_files"),
        index_name("counters"),
    ]
    force_recreate = _env_bool("OPENSEARCH_FORCE_RECREATE", default=False)
    for idx in required:
        if client.indices.exists(index=idx) and force_recreate:
            client.indices.delete(index=idx)
        if not client.indices.exists(index=idx):
            client.indices.create(index=idx, body=_index_body_for(idx))
        _validate_shards_replicas(client, idx)
