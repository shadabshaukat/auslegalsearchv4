# AUSLegalSearch v4 — Tools Index

This directory contains developer tools for AUSLegalSearch v4.

Available tools and documentation:
- SQL Latency Benchmark (p50/p95; vector/FTS/metadata; optimized SQL scenarios)
  - Readme: README-bench-sql-latency.md
  - Script: bench_sql_latency.py

- Delete by URL Utility (single and bulk; literal --show-sql; safe orphan document deletion)
  - Readme: README-delete-url.md
  - Script: delete_url_records.py

- PostgreSQL -> OpenSearch Migration Utility
  - Script: migrate_pg_to_opensearch.py
  - Purpose: copy existing `documents` and `embeddings` rows from PostgreSQL into OpenSearch indexes
  - Example:
    - `python -m tools.migrate_pg_to_opensearch --pg-url 'postgresql+psycopg2://user:pass@host:5432/db' --batch-size 500`

- OpenSearch Rollover Helper
  - Script: opensearch_rollover.py
  - Purpose: inspect alias state and execute/dry-run rollover for documents/embeddings write aliases
  - Example:
    - `python -m tools.opensearch_rollover --suffix both --max-docs 50000000 --dry-run --show-state`

- OpenSearch ISM Bootstrap Helper
  - Script: opensearch_bootstrap_ilm.py
  - Purpose: create/update ISM policy and attach it to alias backing indexes
  - Example:
    - `python -m tools.opensearch_bootstrap_ilm --policy-id auslegalsearch-hot-warm --min-rollover-age 7d --min-warm-age 30d`

Notes:
- All tools inherit database configuration via the shared connector (db/connector.py) and .env. Recommended:
  set -a; source .env; set +a
- For details and examples, see each tool’s README above.
