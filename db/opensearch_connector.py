"""
OpenSearch connector and index bootstrap helpers for AUSLegalSearch v4.

This module is intentionally lightweight and optional. It only activates when
`AUSLEGALSEARCH_STORAGE_BACKEND=opensearch`.
"""

from __future__ import annotations

import os
import re
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


def aliases_enabled() -> bool:
    return _env_bool("OPENSEARCH_USE_ALIASES", default=False)


def alias_name(suffix: str, mode: str) -> str:
    mode = (mode or "read").strip().lower()
    if mode not in {"read", "write"}:
        raise ValueError(f"Unsupported alias mode: {mode}")
    explicit = _env(f"OPENSEARCH_{suffix.upper()}_{mode.upper()}_ALIAS", default="")
    if explicit:
        return explicit
    prefix = _env("OPENSEARCH_ALIAS_PREFIX", "OPENSEARCH_INDEX_PREFIX", "AUSLEGALSEARCH_OS_INDEX_PREFIX", default="auslegalsearch")
    return f"{prefix}_{suffix}_{mode}"


def index_target(suffix: str, purpose: str = "read") -> str:
    """
    Returns the concrete index or alias target depending on alias mode.
    purpose: read|write
    """
    purpose = (purpose or "read").strip().lower()
    if purpose not in {"read", "write"}:
        purpose = "read"
    if aliases_enabled() and suffix in {"documents", "embeddings"}:
        return alias_name(suffix, purpose)
    return index_name(suffix)


def _is_embeddings_index_name(name: str) -> bool:
    base = index_name("embeddings")
    return bool(name == base or re.search(r"_embeddings(?:-\d+)?$", name))


def _is_documents_index_name(name: str) -> bool:
    base = index_name("documents")
    return bool(name == base or re.search(r"_documents(?:-\d+)?$", name))


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
    if _is_embeddings_index_name(name):
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
                    "embedding_key": {"type": "keyword"},
                    "doc_id": {"type": "long"},
                    "doc_key": {"type": "keyword"},
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
                    "text_preview": {"type": "text"},
                    "source": {"type": "keyword"},
                    "format": {"type": "keyword"},
                }
            },
        }

    if _is_documents_index_name(name):
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
                    "doc_key": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
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


def _alias_backing_indexes(client, alias: str) -> list:
    try:
        data = client.indices.get_alias(name=alias)
    except Exception:
        return []
    return sorted(list((data or {}).keys()))


def _alias_write_indexes(client, alias: str) -> list:
    try:
        data = client.indices.get_alias(name=alias)
    except Exception:
        return []
    out = []
    for idx, payload in (data or {}).items():
        aliases = (payload or {}).get("aliases") or {}
        cfg = aliases.get(alias) or {}
        if cfg.get("is_write_index") is True:
            out.append(idx)
    return sorted(out)


def _bootstrap_rollover_family(client, suffix: str, force_recreate: bool = False) -> None:
    write_alias = alias_name(suffix, "write")
    read_alias = alias_name(suffix, "read")
    base = index_name(suffix)
    pad = int(_env("OPENSEARCH_ROLLOVER_INDEX_PAD", default="6"))
    first_index = f"{base}-{str(1).zfill(max(3, pad))}"
    auto_create = _env_bool("OPENSEARCH_ALIAS_AUTO_CREATE", default=True)
    validate = _env_bool("OPENSEARCH_ALIAS_VALIDATE_STARTUP", default=True)

    if force_recreate:
        for alias in (write_alias, read_alias):
            for idx in _alias_backing_indexes(client, alias):
                if client.indices.exists(index=idx):
                    client.indices.delete(index=idx)

    write_backing = _alias_backing_indexes(client, write_alias)
    write_marked = _alias_write_indexes(client, write_alias)
    if not write_backing and auto_create:
        if not client.indices.exists(index=first_index):
            client.indices.create(index=first_index, body=_index_body_for(first_index))
        client.indices.put_alias(index=first_index, name=write_alias, body={"is_write_index": True})
        client.indices.put_alias(index=first_index, name=read_alias)
        write_backing = [first_index]
        write_marked = [first_index]

    # If alias exists but no explicit write marker, set first backing index as write index.
    if write_backing and not write_marked and auto_create:
        client.indices.put_alias(index=write_backing[0], name=write_alias, body={"is_write_index": True})
        write_marked = [write_backing[0]]

    # Keep read alias covering write backing indexes
    if auto_create:
        for idx in write_backing:
            try:
                client.indices.put_alias(index=idx, name=read_alias)
            except Exception:
                pass

    if validate:
        wb = _alias_backing_indexes(client, write_alias)
        rb = _alias_backing_indexes(client, read_alias)
        ww = _alias_write_indexes(client, write_alias)
        for idx in sorted(set(wb + rb)):
            try:
                _validate_shards_replicas(client, idx)
            except Exception:
                raise
        if not wb:
            raise RuntimeError(f"OpenSearch alias validation failed: write alias '{write_alias}' has no backing index")
        if not rb:
            raise RuntimeError(f"OpenSearch alias validation failed: read alias '{read_alias}' has no backing index")
        if len(ww) != 1:
            raise RuntimeError(
                f"OpenSearch alias validation failed: write alias '{write_alias}' must have exactly one write index, found {len(ww)}"
            )


def ensure_opensearch_indexes() -> None:
    client = get_opensearch_client()
    use_aliases = aliases_enabled()
    required = [
        index_name("users"),
        index_name("embedding_sessions"),
        index_name("embedding_session_files"),
        index_name("chat_sessions"),
        index_name("conversion_files"),
        index_name("counters"),
    ]
    if not use_aliases:
        required = [index_name("documents"), index_name("embeddings")] + required

    force_recreate = _env_bool("OPENSEARCH_FORCE_RECREATE", default=False)

    if use_aliases:
        _bootstrap_rollover_family(client, "documents", force_recreate=force_recreate)
        _bootstrap_rollover_family(client, "embeddings", force_recreate=force_recreate)

    for idx in required:
        if client.indices.exists(index=idx) and force_recreate:
            client.indices.delete(index=idx)
        if not client.indices.exists(index=idx):
            client.indices.create(index=idx, body=_index_body_for(idx))
        _validate_shards_replicas(client, idx)
