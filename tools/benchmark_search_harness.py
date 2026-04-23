"""Benchmark OpenSearch search latency and simple recall-proxy overlap metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from embedding.embedder import Embedder
from db import store


def _doc_key(hit: Dict) -> Tuple:
    return (hit.get("doc_id"), hit.get("chunk_index", 0))


def _latency_ms(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms


def _load_queries(args) -> List[str]:
    if args.query:
        return [args.query]
    if args.queries_file:
        p = Path(args.queries_file)
        if not p.exists():
            raise SystemExit(f"queries file not found: {p}")
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
        return [ln for ln in lines if ln]
    raise SystemExit("provide --query or --queries-file")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark vector/bm25/hybrid latency + overlap proxies")
    ap.add_argument("--query", default="")
    ap.add_argument("--queries-file", default="")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1, help="Runs per query")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    queries = _load_queries(args)
    emb = Embedder()

    q_reports: List[Dict] = []
    lat_v: List[float] = []
    lat_b: List[float] = []
    lat_h: List[float] = []
    ov_h_v: List[float] = []
    ov_h_b: List[float] = []

    for q in queries:
        for run_i in range(max(1, int(args.repeat))):
            rec: Dict[str, object] = {"query": q, "run": run_i + 1, "error": None}
            try:
                qv = emb.embed([q])[0]
                v_hits, v_ms = _latency_ms(store.search_vector, qv, top_k=args.top_k)
                b_hits, b_ms = _latency_ms(store.search_bm25, q, top_k=args.top_k)
                h_hits, h_ms = _latency_ms(store.search_hybrid, q, top_k=args.top_k)

                v_keys = {_doc_key(h) for h in (v_hits or [])}
                b_keys = {_doc_key(h) for h in (b_hits or [])}
                h_keys = {_doc_key(h) for h in (h_hits or [])}

                overlap_h_v = (len(h_keys & v_keys) / len(v_keys)) if v_keys else 0.0
                overlap_h_b = (len(h_keys & b_keys) / len(b_keys)) if b_keys else 0.0

                lat_v.append(v_ms)
                lat_b.append(b_ms)
                lat_h.append(h_ms)
                ov_h_v.append(overlap_h_v)
                ov_h_b.append(overlap_h_b)

                rec.update({
                    "latency_ms_vector": round(v_ms, 2),
                    "latency_ms_bm25": round(b_ms, 2),
                    "latency_ms_hybrid": round(h_ms, 2),
                    "count_vector": len(v_hits or []),
                    "count_bm25": len(b_hits or []),
                    "count_hybrid": len(h_hits or []),
                    "overlap_hybrid_vs_vector": round(overlap_h_v, 4),
                    "overlap_hybrid_vs_bm25": round(overlap_h_b, 4),
                })
            except Exception as e:
                rec["error"] = str(e)
            q_reports.append(rec)

    summary = {
        "queries": len(queries),
        "repeat": max(1, int(args.repeat)),
        "runs": len(q_reports),
        "avg_latency_ms_vector": round(mean(lat_v), 2) if lat_v else 0.0,
        "avg_latency_ms_bm25": round(mean(lat_b), 2) if lat_b else 0.0,
        "avg_latency_ms_hybrid": round(mean(lat_h), 2) if lat_h else 0.0,
        "avg_overlap_hybrid_vs_vector": round(mean(ov_h_v), 4) if ov_h_v else 0.0,
        "avg_overlap_hybrid_vs_bm25": round(mean(ov_h_b), 4) if ov_h_b else 0.0,
        "errors": sum(1 for r in q_reports if r.get("error")),
    }

    out = {"summary": summary, "runs": q_reports}
    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(json.dumps(summary, indent=2))
    for r in q_reports:
        if r.get("error"):
            print(f"[ERROR] query={r.get('query')} run={r.get('run')} err={r.get('error')}")
        else:
            print(
                f"query={r.get('query')!r} run={r.get('run')} "
                f"v={r.get('latency_ms_vector')}ms b={r.get('latency_ms_bm25')}ms h={r.get('latency_ms_hybrid')}ms "
                f"ov(h,v)={r.get('overlap_hybrid_vs_vector')} ov(h,b)={r.get('overlap_hybrid_vs_bm25')}"
            )


if __name__ == "__main__":
    main()
