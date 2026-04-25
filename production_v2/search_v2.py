from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from embedding.embedder import Embedder
from production_v2.config import settings
from production_v2.dsl_templates import SCENARIO_TO_BUILDER
from production_v2.opensearch_v2 import get_client


SEMANTIC_HEAVY_SCENARIOS = {
    "semantic",
    "natural_language",
    "precedent",
    "argument_based",
    "fact_pattern",
    "argument_drafting",
    "counterargument",
    "multi_step",
}

_EMBEDDER: Optional[Embedder] = None
_RERANKER = None


def _embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(settings.embed_model)
    return _EMBEDDER


def _reranker():
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        _RERANKER = CrossEncoder(settings.reranker_model)
        return _RERANKER
    except Exception:
        _RERANKER = False
        return None


def classify_query(query: str) -> str:
    q = (query or "").strip()
    low = q.lower()
    if re.search(r"\[\d{4}\]\s*[A-Z]{2,}\s*\d+", q) or re.search(r"\b\d+\s+clr\s+\d+\b", low):
        return "citation"
    if re.search(r"\b(v|vs|versus)\b", low):
        return "case_name"
    if " act " in f" {low} " or re.search(r"\bs\.?\s*\d+\b", low) or "section " in low:
        return "legislation"
    if any(op in low for op in [" and ", " or ", " not ", " near ", "(", ")"]):
        return "boolean"
    if len(low.split()) >= 8 or low.endswith("?"):
        return "natural_language"
    return "keyword"


def _extract_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    src = hit.get("_source") or {}
    return {
        "id": hit.get("_id"),
        "score": hit.get("_score", 0.0),
        "chunk_id": src.get("chunk_id") or hit.get("_id"),
        "authority_id": src.get("authority_id"),
        "chunk_index": src.get("chunk_index"),
        "title": src.get("title"),
        "text": src.get("text"),
        "text_preview": src.get("text_preview"),
        "source": src.get("source"),
        "url": src.get("url"),
        "type": src.get("type"),
        "jurisdiction": src.get("jurisdiction"),
        "database": src.get("database"),
        "citations": src.get("citations") or [],
        "raw": src,
    }


def lexical_search(
    query: str,
    scenario: str,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    client = get_client()
    builder = SCENARIO_TO_BUILDER.get(scenario) or SCENARIO_TO_BUILDER["keyword"]
    dsl = builder(query=query, top_k=top_k, filters=filters)
    resp = client.search(index=settings.index_chunks_lex, body=dsl)
    hits = [_extract_hit(h) for h in ((resp.get("hits") or {}).get("hits") or [])]
    return dsl, hits


def vector_search(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    client = get_client()
    qvec = _embedder().embed([query])[0]
    f = filters or {}
    filter_clauses: List[Dict[str, Any]] = []
    for k in ["type", "jurisdiction", "subjurisdiction", "database", "court"]:
        v = f.get(k)
        if v:
            filter_clauses.append({"term": {k: str(v).lower()}})
    if f.get("date_from") or f.get("date_to"):
        filter_clauses.append({"range": {"date": {kk: vv for kk, vv in {"gte": f.get("date_from"), "lte": f.get("date_to")}.items() if vv}}})

    dsl: Dict[str, Any]
    if filter_clauses:
        dsl = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"knn": {"vector": {"vector": qvec.tolist(), "k": top_k}}}],
                    "filter": filter_clauses,
                }
            },
        }
    else:
        dsl = {"size": top_k, "query": {"knn": {"vector": {"vector": qvec.tolist(), "k": top_k}}}}

    resp = client.search(index=settings.index_chunks_vec, body=dsl)
    hits = [_extract_hit(h) for h in ((resp.get("hits") or {}).get("hits") or [])]
    return dsl, hits


def _hydrate_from_lex(chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    client = get_client()
    docs = client.mget(index=settings.index_chunks_lex, body={"ids": chunk_ids}).get("docs") or []
    out: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        if d.get("found"):
            out[d.get("_id")] = d.get("_source") or {}
    return out


def rrf_fuse(
    lexical_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    top_k: int,
    k: int = 60,
    w_lex: float = 1.0,
    w_vec: float = 1.0,
) -> List[Dict[str, Any]]:
    combined: Dict[str, Dict[str, Any]] = {}
    for rank, h in enumerate(lexical_hits, 1):
        cid = h.get("chunk_id") or h.get("id")
        if not cid:
            continue
        rec = combined.setdefault(cid, {"chunk_id": cid, "rrf_score": 0.0, "lex_rank": None, "vec_rank": None})
        rec["rrf_score"] += w_lex * (1.0 / (k + rank))
        rec["lex_rank"] = rank
        rec.update({kk: vv for kk, vv in h.items() if vv is not None})

    for rank, h in enumerate(vector_hits, 1):
        cid = h.get("chunk_id") or h.get("id")
        if not cid:
            continue
        rec = combined.setdefault(cid, {"chunk_id": cid, "rrf_score": 0.0, "lex_rank": None, "vec_rank": None})
        rec["rrf_score"] += w_vec * (1.0 / (k + rank))
        rec["vec_rank"] = rank
        for kk, vv in h.items():
            if vv is not None and (rec.get(kk) is None or rec.get(kk) == ""):
                rec[kk] = vv

    needs_hydrate = [cid for cid, rec in combined.items() if not rec.get("text")]
    hydrated = _hydrate_from_lex(needs_hydrate)
    for cid in needs_hydrate:
        if cid in hydrated:
            src = hydrated[cid]
            rec = combined[cid]
            rec["text"] = src.get("text")
            rec["text_preview"] = src.get("text_preview")
            rec["title"] = rec.get("title") or src.get("title")
            rec["source"] = rec.get("source") or src.get("source")
            rec["url"] = rec.get("url") or src.get("url")
            rec["type"] = rec.get("type") or src.get("type")
            rec["jurisdiction"] = rec.get("jurisdiction") or src.get("jurisdiction")
            rec["database"] = rec.get("database") or src.get("database")
            rec["citations"] = rec.get("citations") or src.get("citations") or []
            rec["authority_id"] = rec.get("authority_id") or src.get("authority_id")
            rec["chunk_index"] = rec.get("chunk_index") or src.get("chunk_index")

    ranked = sorted(combined.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    return ranked[:top_k]


def _apply_reranker(query: str, hits: List[Dict[str, Any]], top_k: int, rerank_top_n: int) -> Tuple[List[Dict[str, Any]], bool]:
    rr = _reranker()
    if rr is None or rr is False or not hits:
        return hits[:top_k], False
    n = max(1, min(int(rerank_top_n), len(hits)))
    front = hits[:n]
    rest = hits[n:]
    pairs = [(query, (h.get("text") or h.get("text_preview") or "")) for h in front]
    try:
        scores = rr.predict(pairs)
        for i, h in enumerate(front):
            h["rerank_score"] = float(scores[i])
        front = sorted(front, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return (front + rest)[:top_k], True
    except Exception:
        return hits[:top_k], False


def _citation_graph_enrich(query: str, merged: List[Dict[str, Any]], top_n_edges: int = 20) -> List[Dict[str, Any]]:
    client = get_client()
    auth_ids = [str(h.get("authority_id")) for h in merged if h.get("authority_id")]
    cites = []
    for h in merged:
        for c in (h.get("citations") or []):
            if c:
                cites.append(str(c).lower())

    should = []
    if auth_ids:
        should.append({"terms": {"from_authority_id": auth_ids}})
        should.append({"terms": {"to_authority_id": auth_ids}})
    if cites:
        should.append({"terms": {"from_citation": cites}})
        should.append({"terms": {"to_citation": cites}})

    if not should:
        return []

    dsl = {
        "size": int(top_n_edges),
        "query": {
            "bool": {
                "should": should,
                "minimum_should_match": 1,
            }
        },
        "sort": [{"date": {"order": "asc", "unmapped_type": "date"}}],
    }

    try:
        resp = client.search(index=settings.index_citation_graph, body=dsl)
        rows = (resp.get("hits") or {}).get("hits") or []
        return [
            {
                "edge_id": r.get("_id"),
                **(r.get("_source") or {}),
            }
            for r in rows
        ]
    except Exception:
        # Fallback lookup by query citation pattern
        citation_match = re.findall(r"\[(?:19|20)\d{2}\]\s*[A-Z]{2,}\s*\d+|\b\d+\s+CLR\s+\d+\b|\bHCA\s+\d+\b", query or "", flags=re.IGNORECASE)
        if not citation_match:
            return []
        try:
            resp = client.search(
                index=settings.index_citation_graph,
                body={
                    "size": int(top_n_edges),
                    "query": {
                        "bool": {
                            "should": [
                                {"terms": {"from_citation": [c.lower() for c in citation_match]}},
                                {"terms": {"to_citation": [c.lower() for c in citation_match]}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                },
            )
            rows = (resp.get("hits") or {}).get("hits") or []
            return [{"edge_id": r.get("_id"), **(r.get("_source") or {})} for r in rows]
        except Exception:
            return []


def run_search(
    query: str,
    scenario: Optional[str],
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    use_hybrid: bool = True,
    use_reranker: Optional[bool] = None,
    rerank_top_n: Optional[int] = None,
) -> Dict[str, Any]:
    resolved = scenario or classify_query(query)
    lex_dsl, lex_hits = lexical_search(query=query, scenario=resolved, top_k=max(20, top_k), filters=filters)

    vec_dsl: Optional[Dict[str, Any]] = None
    vec_hits: List[Dict[str, Any]] = []
    if use_hybrid or resolved in SEMANTIC_HEAVY_SCENARIOS:
        vec_dsl, vec_hits = vector_search(query=query, top_k=max(20, top_k), filters=filters)

    if vec_hits:
        merged = rrf_fuse(lexical_hits=lex_hits, vector_hits=vec_hits, top_k=top_k)
    else:
        merged = lex_hits[:top_k]

    rerank_enabled = settings.reranker_enable_default if use_reranker is None else bool(use_reranker)
    reranked = False
    if rerank_enabled:
        merged, reranked = _apply_reranker(
            query=query,
            hits=merged,
            top_k=top_k,
            rerank_top_n=int(rerank_top_n or settings.reranker_top_n),
        )

    citation_graph = []
    if resolved in {"citation_tracing", "citation", "precedent", "related"}:
        citation_graph = _citation_graph_enrich(query=query, merged=merged, top_n_edges=20)

    return {
        "scenario": resolved,
        "query": query,
        "top_k": top_k,
        "filters": filters or {},
        "dsl": {
            "lexical": lex_dsl,
            "vector": vec_dsl,
        },
        "counts": {
            "lexical_hits": len(lex_hits),
            "vector_hits": len(vec_hits),
            "returned": len(merged),
            "citation_graph_edges": len(citation_graph),
            "reranked": bool(reranked),
        },
        "reranker": {
            "enabled": bool(rerank_enabled),
            "applied": bool(reranked),
            "model": settings.reranker_model,
            "top_n": int(rerank_top_n or settings.reranker_top_n),
        },
        "citation_graph": citation_graph,
        "results": merged,
    }
