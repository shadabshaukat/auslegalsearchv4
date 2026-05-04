"""
Production search across type-routed OpenSearch indexes.

Supports querying across:
  - cases
  - treaties
  - journals
  - legislation
  - hca

with optional per-type filtering and hybrid (lexical + vector) fusion.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, List, Optional

from db.opensearch_connector import get_opensearch_client
from ingest.production_index_router import BUCKETS, bucket_index_name


def _normalize_types(types: Optional[Iterable[str]]) -> List[str]:
    if not types:
        return list(BUCKETS)
    out = []
    for t in types:
        s = str(t or "").strip().lower()
        if s in BUCKETS and s not in out:
            out.append(s)
    return out or list(BUCKETS)


def _rrf_merge(lex_hits: List[Dict[str, Any]], vec_hits: List[Dict[str, Any]], top_k: int, k: int = 60) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for rank, h in enumerate(lex_hits, start=1):
        hid = str(h.get("_id"))
        rec = merged.setdefault(hid, {"hit": h, "rrf": 0.0, "lex_rank": None, "vec_rank": None})
        rec["lex_rank"] = rank
        rec["rrf"] += 1.0 / float(k + rank)

    for rank, h in enumerate(vec_hits, start=1):
        hid = str(h.get("_id"))
        rec = merged.setdefault(hid, {"hit": h, "rrf": 0.0, "lex_rank": None, "vec_rank": None})
        rec["vec_rank"] = rank
        rec["rrf"] += 1.0 / float(k + rank)

    ranked = sorted(merged.values(), key=lambda x: float(x.get("rrf") or 0.0), reverse=True)[: int(top_k)]
    out = []
    for r in ranked:
        h = r.get("hit") or {}
        src = h.get("_source") or {}
        out.append(
            {
                "id": h.get("_id"),
                "index": h.get("_index"),
                "score": r.get("rrf"),
                "title": src.get("title"),
                "citations": src.get("citations") or [],
                "date": src.get("date"),
                "type": src.get("type"),
                "jurisdiction": src.get("jurisdiction"),
                "subjurisdiction": src.get("subjurisdiction"),
                "database": src.get("database"),
                "url": src.get("url"),
                "source": src.get("source"),
                "chunk_index": src.get("chunk_index"),
                "text_preview": src.get("text_preview") or (src.get("content") or "")[:500],
                "prod_bucket": src.get("prod_bucket"),
                "lex_rank": r.get("lex_rank"),
                "vec_rank": r.get("vec_rank"),
            }
        )
    return out


def search_production(
    query: str,
    top_k: int = 10,
    types: Optional[Iterable[str]] = None,
    use_vector: bool = True,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    selected_types = _normalize_types(types)
    indices = [bucket_index_name(t) for t in selected_types]

    client = get_opensearch_client()

    lex_body = {
        "size": max(int(top_k) * 5, 25),
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^4",
                    "citations^3",
                    "content^2",
                    "text_preview^2",
                    "chunk_metadata_text",
                ],
            }
        },
    }
    lex_res = client.search(index=indices, body=lex_body)
    lex_hits = (lex_res.get("hits") or {}).get("hits") or []

    vec_hits: List[Dict[str, Any]] = []
    vector_used = False
    vector_error = None
    if bool(use_vector):
        try:
            from embedding.embedder import Embedder

            emb = Embedder()
            qv = emb.embed([query])[0]
            qv = qv.tolist() if hasattr(qv, "tolist") else list(qv)
            vec_body = {
                "size": max(int(top_k) * 5, 25),
                "query": {
                    "knn": {
                        "vector": {
                            "vector": qv,
                            "k": max(int(top_k) * 5, 25),
                        }
                    }
                },
            }
            vec_res = client.search(index=indices, body=vec_body)
            vec_hits = (vec_res.get("hits") or {}).get("hits") or []
            vector_used = True
        except Exception as e:
            vector_error = str(e)

    results = _rrf_merge(lex_hits=lex_hits, vec_hits=vec_hits, top_k=int(top_k), k=int(rrf_k))
    if not results:
        # lexical-only fallback shape
        results = _rrf_merge(lex_hits=lex_hits, vec_hits=[], top_k=int(top_k), k=int(rrf_k))

    return {
        "query": query,
        "types": selected_types,
        "indexes": indices,
        "top_k": int(top_k),
        "vector_used": vector_used,
        "vector_error": vector_error,
        "results": results,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search production type-routed indexes")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--types", default="", help="comma list: cases,treaties,journals,legislation,hca")
    ap.add_argument("--no_vector", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    import json

    ns = _parse_args()
    types = [x.strip() for x in str(ns.types or "").split(",") if x.strip()]
    out = search_production(
        query=ns.query,
        top_k=ns.top_k,
        types=types,
        use_vector=not bool(ns.no_vector),
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
