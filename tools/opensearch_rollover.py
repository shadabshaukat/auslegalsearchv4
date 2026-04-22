"""
OpenSearch rollover helper for AUSLegalSearch v4.

Supports:
- inspect aliases/backing indexes
- perform explicit rollover for documents/embeddings write aliases
- dry-run rollover condition checks
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Any

from db.opensearch_connector import get_opensearch_client, alias_name, aliases_enabled


def _default_conditions(args) -> Dict[str, Any]:
    cond: Dict[str, Any] = {}
    if args.max_docs and args.max_docs > 0:
        cond["max_docs"] = int(args.max_docs)
    if args.max_primary_shard_size:
        cond["max_primary_shard_size"] = str(args.max_primary_shard_size)
    if args.max_age:
        cond["max_age"] = str(args.max_age)
    if not cond:
        cond = {"max_docs": 50_000_000}
    return cond


def _print_alias_state(client, suffix: str) -> None:
    w = alias_name(suffix, "write")
    r = alias_name(suffix, "read")
    out = {"suffix": suffix, "write_alias": w, "read_alias": r, "write_backing": [], "read_backing": []}
    try:
        wa = client.indices.get_alias(name=w)
        out["write_backing"] = sorted(list((wa or {}).keys()))
    except Exception:
        pass
    try:
        ra = client.indices.get_alias(name=r)
        out["read_backing"] = sorted(list((ra or {}).keys()))
    except Exception:
        pass
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Rollover helper for OpenSearch alias families")
    ap.add_argument("--suffix", choices=["documents", "embeddings", "both"], default="both")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--max-primary-shard-size", default="")
    ap.add_argument("--max-age", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-state", action="store_true")
    args = ap.parse_args()

    if not aliases_enabled():
        raise SystemExit("OPENSEARCH_USE_ALIASES is disabled. Enable aliases before using rollover helper.")

    client = get_opensearch_client()
    suffixes = ["documents", "embeddings"] if args.suffix == "both" else [args.suffix]
    conditions = _default_conditions(args)

    if args.show_state:
        for s in suffixes:
            _print_alias_state(client, s)

    for s in suffixes:
        write_alias = alias_name(s, "write")
        body = {"conditions": conditions}
        res = client.indices.rollover(alias=write_alias, body=body, dry_run=bool(args.dry_run))
        print(json.dumps({"suffix": s, "write_alias": write_alias, "dry_run": bool(args.dry_run), "result": res}, indent=2))


if __name__ == "__main__":
    main()
