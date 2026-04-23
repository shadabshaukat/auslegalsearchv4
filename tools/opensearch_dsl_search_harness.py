"""Professional OpenSearch DSL harness for legal search use-cases.

Features
- Loads use-cases from JSON (query phrase/natural language + expected terms + optional filters)
- Builds OpenSearch DSL per use-case type
- Executes search against OpenSearch documents+embeddings read targets
- Computes lightweight accuracy proxies:
  - hits_returned
  - hit_rate (hits>0)
  - topk_term_match_rate (share of hits containing >=1 expected term)
  - expected_term_coverage (how many expected terms are seen in top-k text)
- Saves full JSON report including DSL bodies for reproducibility
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from db.opensearch_connector import get_opensearch_client, index_target


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _safe_text(hit: Dict[str, Any]) -> str:
    src = hit.get("_source") or {}
    parts = [
        src.get("content") or "",
        src.get("text") or "",
        src.get("text_preview") or "",
        json.dumps(src.get("chunk_metadata") or {}, ensure_ascii=False),
        src.get("source") or "",
    ]
    return "\n".join([p for p in parts if p])


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _metadata_filters(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    for k, v in (filters or {}).items():
        val = str(v)
        # Try both top-level and chunk_metadata dynamic fields
        clauses.append(
            {
                "bool": {
                    "should": [
                        {"term": {f"{k}.keyword": val}},
                        {"term": {k: val}},
                        {"term": {f"chunk_metadata.{k}.keyword": val}},
                        {"term": {f"chunk_metadata.{k}": val}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    return clauses


def build_dsl(case: Dict[str, Any], size: int) -> Dict[str, Any]:
    q = case.get("query", "")
    t = (case.get("type") or "keyword").strip().lower()
    filters = case.get("filters") or {}
    filter_clauses = _metadata_filters(filters)

    if t in {"keyword", "case_name", "citation", "legislation", "natural_language"}:
        return {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": q,
                                "fields": [
                                    "content^3",
                                    "text^2",
                                    "text_preview",
                                    "source^1.2",
                                    "chunk_metadata.*",
                                ],
                                "type": "best_fields",
                                "operator": "and" if t in {"keyword", "citation", "legislation"} else "or",
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            },
            "highlight": {"fields": {"content": {}, "text": {}, "text_preview": {}}},
        }

    if t == "boolean":
        # Use query_string so legal connector style operators are respected.
        return {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "query_string": {
                                "query": q,
                                "fields": ["content^3", "text^2", "text_preview", "chunk_metadata.*"],
                                "default_operator": "AND",
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            },
        }

    if t in {"conceptual", "semantic", "precedent", "fact_pattern", "argument_based", "related"}:
        # Best-effort efficient hybrid-like lexical query using dis_max + phrase matching boosts.
        return {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "dis_max": {
                                "queries": [
                                    {
                                        "multi_match": {
                                            "query": q,
                                            "fields": ["content^3", "text^2", "text_preview", "chunk_metadata.*"],
                                            "type": "most_fields",
                                            "operator": "or",
                                        }
                                    },
                                    {
                                        "multi_match": {
                                            "query": q,
                                            "fields": ["content^4", "text^3"],
                                            "type": "phrase",
                                            "slop": 2,
                                        }
                                    },
                                ],
                                "tie_breaker": 0.2,
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            },
        }

    if t == "metadata_filter":
        return {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": q,
                                "fields": ["content^2", "text^2", "text_preview", "chunk_metadata.*"],
                                "operator": "or",
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            },
        }

    # fallback
    return {
        "size": size,
        "query": {
            "multi_match": {
                "query": q,
                "fields": ["content^2", "text^2", "text_preview", "chunk_metadata.*"],
            }
        },
    }


def score_hits(hits: List[Dict[str, Any]], expected_terms: List[str], top_k: int) -> Dict[str, Any]:
    n = len(hits)
    top_hits = hits[:top_k]
    terms = [str(t).lower() for t in (expected_terms or []) if str(t).strip()]
    if not terms:
        return {
            "hits_returned": n,
            "hit_rate": 1.0 if n > 0 else 0.0,
            "topk_term_match_rate": 0.0,
            "expected_term_coverage": 0.0,
        }

    matched_hits = 0
    term_seen = {t: False for t in terms}
    for h in top_hits:
        txt = _safe_text(h).lower()
        hit_has_any = False
        for t in terms:
            if t in txt:
                hit_has_any = True
                term_seen[t] = True
        if hit_has_any:
            matched_hits += 1

    topk_term_match_rate = (matched_hits / len(top_hits)) if top_hits else 0.0
    expected_term_coverage = (sum(1 for v in term_seen.values() if v) / len(term_seen)) if term_seen else 0.0

    return {
        "hits_returned": n,
        "hit_rate": 1.0 if n > 0 else 0.0,
        "topk_term_match_rate": round(topk_term_match_rate, 4),
        "expected_term_coverage": round(expected_term_coverage, 4),
    }


def run_case(client, indices: List[str], case: Dict[str, Any], default_top_k: int) -> Dict[str, Any]:
    top_k = int(case.get("top_k") or default_top_k)
    dsl = build_dsl(case, size=top_k)
    res = client.search(index=indices, body=dsl)
    hits = ((res or {}).get("hits") or {}).get("hits") or []
    metrics = score_hits(hits, case.get("expected_terms") or [], top_k=top_k)
    return {
        "id": case.get("id"),
        "type": case.get("type"),
        "query": case.get("query"),
        "top_k": top_k,
        "dsl": dsl,
        "metrics": metrics,
        "sample_hits": [
            {
                "score": h.get("_score"),
                "source": (h.get("_source") or {}).get("source"),
                "doc_id": (h.get("_source") or {}).get("doc_id"),
                "chunk_index": (h.get("_source") or {}).get("chunk_index"),
            }
            for h in hits[:3]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenSearch DSL test harness for legal search use-cases")
    ap.add_argument("--usecases", default="tools/opensearch_search_usecases.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--json", action="store_true", help="Print full report JSON to stdout")
    args = ap.parse_args()

    p = Path(args.usecases)
    if not p.exists():
        raise SystemExit(f"usecase file not found: {p}")
    usecases = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(usecases, list):
        raise SystemExit("usecase file must be a JSON array")

    client = get_opensearch_client()
    indices = [index_target("documents", purpose="read"), index_target("embeddings", purpose="read")]

    results = []
    for case in usecases:
        try:
            results.append(run_case(client, indices, case, default_top_k=args.top_k))
        except Exception as e:
            results.append(
                {
                    "id": case.get("id"),
                    "type": case.get("type"),
                    "query": case.get("query"),
                    "error": str(e),
                }
            )

    valid = [r for r in results if not r.get("error")]
    summary = {
        "total_cases": len(results),
        "ok_cases": len(valid),
        "error_cases": len(results) - len(valid),
        "avg_hit_rate": round(sum((r["metrics"]["hit_rate"] for r in valid), 0.0) / len(valid), 4) if valid else 0.0,
        "avg_topk_term_match_rate": round(sum((r["metrics"]["topk_term_match_rate"] for r in valid), 0.0) / len(valid), 4) if valid else 0.0,
        "avg_expected_term_coverage": round(sum((r["metrics"]["expected_term_coverage"] for r in valid), 0.0) / len(valid), 4) if valid else 0.0,
        "indices": indices,
    }
    report = {"summary": summary, "results": results}

    out_path = Path(args.out) if args.out else Path("logs") / f"opensearch_search_harness_report-{_now_stamp()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(summary, indent=2))
        print(f"report_path={out_path}")
        for r in results:
            if r.get("error"):
                print(f"[ERROR] {r.get('id')} :: {r.get('error')}")
            else:
                m = r["metrics"]
                print(
                    f"[{r.get('id')}] hits={m['hits_returned']} hit_rate={m['hit_rate']} "
                    f"term_match={m['topk_term_match_rate']} coverage={m['expected_term_coverage']}"
                )


if __name__ == "__main__":
    main()
