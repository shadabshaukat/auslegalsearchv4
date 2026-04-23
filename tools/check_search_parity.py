"""Compare search result shape/completeness between two backends.

This script runs lightweight search calls under backend A and backend B in
separate subprocesses, then compares field presence/shape.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List


CHILD = r'''
import json
from embedding.embedder import Embedder
from db import store

query = {query!r}
top_k = int({top_k})

out = {{"backend": {backend!r}, "error": None, "modes": {{}}}}
try:
    emb = Embedder()
    qv = emb.embed([query])[0]
    modes = {{
        "vector": store.search_vector(qv, top_k=top_k),
        "bm25": store.search_bm25(query, top_k=top_k),
        "hybrid": store.search_hybrid(query, top_k=top_k),
        "fts": store.search_fts(query, top_k=top_k),
    }}
    for name, hits in modes.items():
        sample = hits[0] if hits else {{}}
        out["modes"][name] = {{
            "count": len(hits),
            "fields": sorted(list(sample.keys())),
            "sample": sample,
        }}
except Exception as e:
    out["error"] = str(e)

print(json.dumps(out))
'''


def _run_backend(backend: str, query: str, top_k: int) -> Dict:
    env = os.environ.copy()
    env["AUSLEGALSEARCH_STORAGE_BACKEND"] = backend
    code = CHILD.format(query=query, top_k=top_k, backend=backend)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"backend": backend, "error": proc.stderr.strip() or proc.stdout.strip(), "modes": {}}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"backend": backend, "error": "failed to parse child JSON", "raw": proc.stdout}


def _compare(a: Dict, b: Dict) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    modes = sorted(set((a.get("modes") or {}).keys()) | set((b.get("modes") or {}).keys()))
    for m in modes:
        af = set(((a.get("modes") or {}).get(m) or {}).get("fields") or [])
        bf = set(((b.get("modes") or {}).get(m) or {}).get("fields") or [])
        out[m] = {
            "fields_only_in_a": sorted(list(af - bf)),
            "fields_only_in_b": sorted(list(bf - af)),
            "count_a": ((a.get("modes") or {}).get(m) or {}).get("count", 0),
            "count_b": ((b.get("modes") or {}).get(m) or {}).get("count", 0),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare search result shape between two backends")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--backend-a", default="postgres", choices=["postgres", "opensearch", "oracle"])
    ap.add_argument("--backend-b", default="opensearch", choices=["postgres", "opensearch", "oracle"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    a = _run_backend(args.backend_a, args.query, args.top_k)
    b = _run_backend(args.backend_b, args.query, args.top_k)
    cmp = _compare(a, b)

    out = {"a": a, "b": b, "comparison": cmp}
    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"backend_a={args.backend_a} error={a.get('error')}")
    print(f"backend_b={args.backend_b} error={b.get('error')}")
    for mode, c in cmp.items():
        print(f"[{mode}] count_a={c['count_a']} count_b={c['count_b']}")
        if c["fields_only_in_a"]:
            print(f"  fields_only_in_a: {c['fields_only_in_a']}")
        if c["fields_only_in_b"]:
            print(f"  fields_only_in_b: {c['fields_only_in_b']}")


if __name__ == "__main__":
    main()
