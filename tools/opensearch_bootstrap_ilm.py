"""
Bootstrap OpenSearch ISM policy and attach it to alias backing indexes.

Note: OpenSearch uses Index State Management (ISM), not Elasticsearch ILM.
This helper creates/updates an ISM policy and applies it to current backing indexes
for documents/embeddings read/write alias families.
"""

from __future__ import annotations

import argparse
import json

from db.opensearch_connector import get_opensearch_client, alias_name, aliases_enabled


def _default_policy(policy_id: str, min_rollover_age: str, min_warm_age: str):
    return {
        "policy": {
            "policy_id": policy_id,
            "description": "AUSLegalSearch rollover/warm policy",
            "default_state": "hot",
            "states": [
                {
                    "name": "hot",
                    "actions": [
                        {
                            "rollover": {
                                "min_index_age": min_rollover_age,
                            }
                        }
                    ],
                    "transitions": [
                        {
                            "state_name": "warm",
                            "conditions": {
                                "min_index_age": min_warm_age,
                            },
                        }
                    ],
                },
                {
                    "name": "warm",
                    "actions": [],
                    "transitions": [],
                },
            ],
            "ism_template": [
                {
                    "index_patterns": ["*documents-*", "*embeddings-*"],
                    "priority": 100,
                }
            ],
        }
    }


def _attach_policy(client, index_name: str, policy_id: str):
    body = {"index.plugins.index_state_management.policy_id": policy_id}
    return client.indices.put_settings(index=index_name, body={"index": {"plugins.index_state_management.policy_id": policy_id}})


def _alias_backing(client, alias: str):
    try:
        payload = client.indices.get_alias(name=alias)
    except Exception:
        return []
    return sorted(list((payload or {}).keys()))


def main():
    ap = argparse.ArgumentParser(description="Bootstrap OpenSearch ISM policy and attach to alias backing indexes")
    ap.add_argument("--policy-id", default="auslegalsearch-hot-warm")
    ap.add_argument("--min-rollover-age", default="7d")
    ap.add_argument("--min-warm-age", default="30d")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not aliases_enabled():
        raise SystemExit("OPENSEARCH_USE_ALIASES is disabled. Enable aliases before ISM bootstrap.")

    client = get_opensearch_client()
    pol = _default_policy(args.policy_id, args.min_rollover_age, args.min_warm_age)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "policy": pol}, indent=2))
        return

    put_res = client.transport.perform_request("PUT", f"/_plugins/_ism/policies/{args.policy_id}", body=pol)
    print(json.dumps({"policy_put": put_res}, indent=2))

    aliases = [
        alias_name("documents", "write"),
        alias_name("documents", "read"),
        alias_name("embeddings", "write"),
        alias_name("embeddings", "read"),
    ]
    seen = set()
    for a in aliases:
        for idx in _alias_backing(client, a):
            if idx in seen:
                continue
            seen.add(idx)
            try:
                res = _attach_policy(client, idx, args.policy_id)
                print(json.dumps({"index": idx, "attached": True, "response": res}, indent=2))
            except Exception as e:
                print(json.dumps({"index": idx, "attached": False, "error": str(e)}, indent=2))


if __name__ == "__main__":
    main()
