# AUSLegalSearch v3 — Tools Index

This directory contains developer tools for AUSLegalSearch v3.

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

Notes:
- All tools inherit database configuration via the shared connector (db/connector.py) and .env. Recommended:
  set -a; source .env; set +a
- For details and examples, see each tool’s README above.
