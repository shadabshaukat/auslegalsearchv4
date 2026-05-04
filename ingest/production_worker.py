"""
Production ingestion worker wrapper.

Uses all beta worker logic (parsing/chunking/pipeline/governor/resume/logging),
but routes OpenSearch writes into one of 5 production indexes by file type:
  - cases
  - treaties
  - journals
  - legislation
  - hca

No beta files are modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from opensearchpy.helpers import bulk, parallel_bulk  # type: ignore

from db import store as _store
from db.opensearch_connector import _embedding_dim, _env, _index_body_for, get_opensearch_client
from ingest import beta_worker as _beta
from ingest.production_index_router import bucket_index_name, resolve_bucket


_ENSURED: set[str] = set()


def _ensure_type_index(index_name: str) -> None:
    if index_name in _ENSURED:
        return
    client = get_opensearch_client()
    if not client.indices.exists(index=index_name):
        dim = _embedding_dim()
        total_fields_limit = int(_env("OPENSEARCH_TOTAL_FIELDS_LIMIT", default="5000"))
        body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": int(_env("OPENSEARCH_NUMBER_OF_SHARDS", "AUSLEGALSEARCH_OS_SHARDS", default="1")),
                    "number_of_replicas": int(_env("OPENSEARCH_NUMBER_OF_REPLICAS", "AUSLEGALSEARCH_OS_REPLICAS", default="0")),
                    "mapping.total_fields.limit": total_fields_limit,
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
                    "text_preview": {"type": "text"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": _env("OPENSEARCH_KNN_METHOD", default="hnsw"),
                            "space_type": _env("OPENSEARCH_KNN_SPACE", default="cosinesimil"),
                            "engine": _env("OPENSEARCH_KNN_ENGINE", default="lucene"),
                        },
                    },
                    "chunk_metadata": {"type": "object", "enabled": True},
                    "chunk_metadata_text": {"type": "text"},
                    "title": {"type": "text"},
                    "titles": {"type": "keyword"},
                    "citations": {"type": "keyword"},
                    "date": {"type": "date"},
                    "type": {"type": "keyword"},
                    "subjurisdiction": {"type": "keyword"},
                    "jurisdiction": {"type": "keyword"},
                    "data_quality": {"type": "keyword"},
                    "database": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "prod_bucket": {"type": "keyword"},
                    "created_at": {"type": "date"},
                }
            },
        }
        client.indices.create(index=index_name, body=body)
    _ENSURED.add(index_name)


def _bulk_upsert_file_chunks_opensearch_production(
    source_path: str,
    fmt: str,
    chunks,
    vectors,
    max_retries: int = 3,
    chunk_start_index: int = 0,
) -> int:
    """
    Same call signature as db.store.bulk_upsert_file_chunks_opensearch,
    but routes each file into exactly one production type index.
    """
    if not chunks:
        return 0
    if vectors is None or len(vectors) != len(chunks):
        raise ValueError("Vector batch does not match chunks in bulk upsert")

    md_items = [((c or {}).get("chunk_metadata") or {}) for c in chunks]
    bucket = resolve_bucket(source_path=source_path, chunk_metadata_items=md_items)
    idx = bucket_index_name(bucket)
    _ensure_type_index(idx)

    client = _store._os_client()  # noqa: SLF001 - intentional reuse
    timeout_sec = _store._os_int("OPENSEARCH_TIMEOUT", 120)  # noqa: SLF001
    bulk_chunk_size = max(1, _store._os_int("OPENSEARCH_BULK_CHUNK_SIZE", 500))  # noqa: SLF001
    bulk_max_bytes = max(1024 * 1024, _store._os_int("OPENSEARCH_BULK_MAX_BYTES", 104857600))  # noqa: SLF001
    bulk_concurrency = max(1, _store._os_int("OPENSEARCH_BULK_CONCURRENCY", 2))  # noqa: SLF001
    bulk_queue_size = max(1, _store._os_int("OPENSEARCH_BULK_QUEUE_SIZE", 8))  # noqa: SLF001
    refresh_on_write = _store._os_bool("OPENSEARCH_REFRESH_ON_WRITE", False)  # noqa: SLF001

    def _iter_actions():
        for i, chunk in enumerate(chunks):
            abs_i = int(chunk_start_index) + int(i)
            text_body = (chunk or {}).get("text", "")
            md = _store._sanitize_chunk_metadata_for_os((chunk or {}).get("chunk_metadata") or {})  # noqa: SLF001
            if _store._os_bool("OPENSEARCH_DROP_METADATA", False):  # noqa: SLF001
                md = None

            doc_key = _store._chunk_doc_key(source_path, abs_i, text_body)  # noqa: SLF001
            doc_id = _store._stable_int_id("doc:" + doc_key)  # noqa: SLF001
            emb_key = hashlib.sha1(f"{doc_key}|{abs_i}".encode("utf-8")).hexdigest()

            src = {
                "doc_id": int(doc_id),
                "doc_key": doc_key,
                "chunk_index": abs_i,
                "source": source_path,
                "content": text_body,
                "text_preview": text_body[:500],
                "format": fmt,
                "vector": list(vectors[i]),
                "chunk_metadata": md,
                "chunk_metadata_text": json.dumps(md or {}, ensure_ascii=False),
                "title": (md or {}).get("title"),
                "titles": (md or {}).get("titles") or [],
                "citations": (md or {}).get("citations") or [],
                "date": (md or {}).get("date"),
                "type": (md or {}).get("type"),
                "subjurisdiction": (md or {}).get("subjurisdiction"),
                "jurisdiction": (md or {}).get("jurisdiction"),
                "data_quality": (md or {}).get("data_quality"),
                "database": (md or {}).get("database"),
                "url": (md or {}).get("url"),
                "prod_bucket": bucket,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            yield {"_op_type": "index", "_index": idx, "_id": str(emb_key), "_source": src}

    last_err = None
    for attempt in range(max(1, int(max_retries))):
        try:
            if bulk_concurrency > 1:
                errors_count = 0
                for ok, _item in parallel_bulk(
                    client,
                    _iter_actions(),
                    thread_count=bulk_concurrency,
                    queue_size=bulk_queue_size,
                    chunk_size=bulk_chunk_size,
                    max_chunk_bytes=bulk_max_bytes,
                    refresh=refresh_on_write,
                    request_timeout=timeout_sec,
                    raise_on_error=False,
                    raise_on_exception=False,
                ):
                    if not ok:
                        errors_count += 1
                if errors_count > 0:
                    raise RuntimeError(f"bulk upsert errors: {errors_count} failures")
            else:
                _, errors = bulk(
                    client,
                    _iter_actions(),
                    chunk_size=bulk_chunk_size,
                    max_chunk_bytes=bulk_max_bytes,
                    refresh=refresh_on_write,
                    request_timeout=timeout_sec,
                    raise_on_error=False,
                    raise_on_exception=False,
                )
                if errors:
                    raise RuntimeError(f"bulk upsert errors: {len(errors)} failures")

            return len(chunks)
        except Exception as e:
            last_err = e
            if attempt + 1 < max(1, int(max_retries)):
                time.sleep(min(8, 2 ** attempt))
            else:
                break

    raise RuntimeError(f"OpenSearch production bulk upsert failed after retries: {last_err}")


def run_worker_opensearch(*args, **kwargs):
    _beta.bulk_upsert_file_chunks_opensearch = _bulk_upsert_file_chunks_opensearch_production
    return _beta.run_worker_opensearch(*args, **kwargs)


def run_worker_opensearch_pipelined(*args, **kwargs):
    _beta.bulk_upsert_file_chunks_opensearch = _bulk_upsert_file_chunks_opensearch_production
    return _beta.run_worker_opensearch_pipelined(*args, **kwargs)


def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    return _beta._parse_cli_args(argv)


if __name__ == "__main__":
    import sys

    args = _parse_cli_args(sys.argv[1:])
    run_worker_opensearch(
        session_name=args["session_name"],
        root_dir=args.get("root"),
        partition_file=args.get("partition_file"),
        embedding_model=args.get("model"),
        token_target=args.get("target_tokens") or 512,
        token_overlap=args.get("overlap_tokens") or 64,
        token_max=args.get("max_tokens") or 640,
        log_dir=args.get("log_dir") or "./logs",
        resume=bool(args.get("resume")),
    )
