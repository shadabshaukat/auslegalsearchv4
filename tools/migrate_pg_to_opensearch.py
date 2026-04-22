"""
Migrate existing PostgreSQL documents/embeddings data into OpenSearch indexes.

This utility is intentionally standalone so you can run it even when
`AUSLEGALSEARCH_STORAGE_BACKEND=opensearch`.

Usage example:
    python -m tools.migrate_pg_to_opensearch \
      --pg-url "postgresql+psycopg2://user:pass@host:5432/db" \
      --batch-size 500
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

from sqlalchemy import create_engine, text

from db.opensearch_connector import ensure_opensearch_indexes, get_opensearch_client, index_name


def _resolve_pg_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    env_url = os.environ.get("AUSLEGALSEARCH_PG_MIGRATION_URL") or os.environ.get("AUSLEGALSEARCH_DB_URL")
    if env_url:
        return env_url
    raise RuntimeError(
        "Missing PostgreSQL URL. Provide --pg-url or set AUSLEGALSEARCH_PG_MIGRATION_URL (or AUSLEGALSEARCH_DB_URL)."
    )


def _parse_vector_text(vtxt: Any) -> List[float]:
    if vtxt is None:
        return []
    if isinstance(vtxt, (list, tuple)):
        return [float(x) for x in vtxt]
    s = str(vtxt).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def _parse_json_text(v: Any):
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except Exception:
        return None


def _chunked_fetch(conn, sql: str, batch_size: int) -> Iterable[List[Any]]:
    result = conn.execution_options(stream_results=True).execute(text(sql))
    while True:
        rows = result.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def migrate_documents(conn, client, batch_size: int) -> int:
    idx = index_name("documents")
    total = 0
    from opensearchpy.helpers import bulk  # type: ignore

    sql = """
        SELECT id, source, content, format
        FROM documents
        ORDER BY id
    """
    for rows in _chunked_fetch(conn, sql, batch_size):
        actions = []
        for r in rows:
            doc_id = int(r[0])
            actions.append(
                {
                    "_op_type": "index",
                    "_index": idx,
                    "_id": str(doc_id),
                    "_source": {
                        "doc_id": doc_id,
                        "source": r[1],
                        "content": r[2],
                        "format": r[3],
                    },
                }
            )
        if actions:
            bulk(client, actions, refresh=False)
            total += len(actions)
            print(f"[migrate] documents indexed: {total}")
    client.indices.refresh(index=idx)
    return total


def migrate_embeddings(conn, client, batch_size: int) -> int:
    idx = index_name("embeddings")
    total = 0
    from opensearchpy.helpers import bulk  # type: ignore

    sql = """
        SELECT e.id, e.doc_id, e.chunk_index, e.vector::text, e.chunk_metadata::text,
               d.content, d.source, d.format
        FROM embeddings e
        LEFT JOIN documents d ON d.id = e.doc_id
        ORDER BY e.id
    """
    for rows in _chunked_fetch(conn, sql, batch_size):
        actions = []
        for r in rows:
            emb_id = int(r[0])
            actions.append(
                {
                    "_op_type": "index",
                    "_index": idx,
                    "_id": str(emb_id),
                    "_source": {
                        "embedding_id": emb_id,
                        "doc_id": int(r[1]) if r[1] is not None else None,
                        "chunk_index": int(r[2]) if r[2] is not None else 0,
                        "vector": _parse_vector_text(r[3]),
                        "chunk_metadata": _parse_json_text(r[4]),
                        "text": r[5] or "",
                        "source": r[6] or "",
                        "format": r[7] or "",
                    },
                }
            )
        if actions:
            bulk(client, actions, refresh=False)
            total += len(actions)
            print(f"[migrate] embeddings indexed: {total}")
    client.indices.refresh(index=idx)
    return total


def sync_counters(conn, client):
    counters = index_name("counters")
    doc_max = conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM documents")).scalar() or 0
    emb_max = conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM embeddings")).scalar() or 0
    client.index(index=counters, id="documents", body={"value": int(doc_max)}, refresh=True)
    client.index(index=counters, id="embeddings", body={"value": int(emb_max)}, refresh=True)
    print(f"[migrate] counters synced: documents={doc_max}, embeddings={emb_max}")


def main():
    ap = argparse.ArgumentParser(description="Migrate PostgreSQL documents/embeddings into OpenSearch")
    ap.add_argument("--pg-url", default=None, help="PostgreSQL SQLAlchemy URL")
    ap.add_argument("--batch-size", type=int, default=500, help="Bulk indexing batch size")
    args = ap.parse_args()

    pg_url = _resolve_pg_url(args.pg_url)
    print("[migrate] Ensuring OpenSearch indexes...")
    ensure_opensearch_indexes()
    client = get_opensearch_client()

    print("[migrate] Connecting to PostgreSQL source...")
    engine = create_engine(pg_url)
    with engine.connect() as conn:
        docs = migrate_documents(conn, client, max(1, int(args.batch_size)))
        embs = migrate_embeddings(conn, client, max(1, int(args.batch_size)))
        sync_counters(conn, client)

    print(f"[migrate] COMPLETE: documents={docs}, embeddings={embs}")


if __name__ == "__main__":
    main()
