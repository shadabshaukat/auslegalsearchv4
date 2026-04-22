# AUSLegalSearch v3 — Delete by URL Tool

This utility locates and deletes all records for one or more URL values from the ingestion tables:
- Deletes rows in `embeddings` where `chunk_metadata->>'url' = :url`
- Then deletes any rows in `documents` whose `id` is referenced by the deleted embeddings and no embeddings remain for that document (safe-orphan delete)

It supports single-URL mode and bulk mode (a file with one URL per line). It shares the same Postgres connection setup as the rest of the project and loads `.env` via the shared connector.

Tool path:
- tools/delete_url_records.py


## Prerequisites

- Python 3.10+
- Project dependencies installed (`pip install -r requirements.txt`)
- PostgreSQL database with the project schema
- Database configuration via environment
  - Recommended to export `.env` before running:
    ```bash
    set -a; source .env; set +a
    ```
  - Alternatively provide a single DSN in `AUSLEGALSEARCH_DB_URL`
- The tool inherits DB connection from `db/connector.py`, which has a built-in `.env` loader


## Environment variables

Either a single DSN:
- `AUSLEGALSEARCH_DB_URL='postgresql+psycopg2://user:pass@host:5432/dbname'`

Or individual variables:
- `AUSLEGALSEARCH_DB_HOST`, `AUSLEGALSEARCH_DB_PORT`, `AUSLEGALSEARCH_DB_USER`, `AUSLEGALSEARCH_DB_PASSWORD`, `AUSLEGALSEARCH_DB_NAME`

Note: The project’s `db/connector.py` reads `.env` when present (exported env vars take precedence). Running from the repo root is recommended.


## Usage

Single URL — dry run (preview only):
```bash
python -m tools.delete_url_records \
  --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html" \
  --dry-run
```

Single URL — delete with prompt:
```bash
python -m tools.delete_url_records \
  --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html"
```

Single URL — delete without prompt:
```bash
python -m tools.delete_url_records \
  --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html" \
  --yes
```

Bulk delete — file with one URL per line:
- Create a text file containing URLs, one per line. Blank lines and lines starting with `#` are ignored.
- Example `urls.txt`:
  ```
  https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html
  https://austlii.edu.au/cgi-bin/viewdoc/au/cases/cth/HCA/2022/35.html
  # comment line is ignored
  ```
- Run in dry-run mode first:
  ```bash
  python -m tools.delete_url_records --url-file "/abs/path/urls.txt" --dry-run
  ```
- Perform deletion (no prompt):
  ```bash
  python -m tools.delete_url_records --url-file "/abs/path/urls.txt" --yes
  ```

Show the SQL used and exit (single and bulk)

- Single URL (prints literal SQL with the provided URL substituted):
```bash
python -m tools.delete_url_records \
  --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html" \
  --show-sql
```

- Bulk URLs (prints literal SQL for the first N URLs; cap controlled by AUSLEGALSEARCH_SHOWSQL_MAXURLS, default 20):
```bash
python -m tools.delete_url_records \
  --url-file "/abs/path/urls.txt" \
  --show-sql
```

- Template output (when no URL is provided, a parameterized :url template is printed):
```bash
python -m tools.delete_url_records --show-sql
# prints parameterized SQL with :url placeholders
```


## What the tool does (logic)

For each URL:
1) Preview:
   - Count embeddings that would be deleted
   - Count documents that are referenced by those embeddings
   - Show distinct doc_ids and example embeddings (up to 10)
2) If confirmed:
   - Delete embeddings where `chunk_metadata->>'url' = :url`, returning `(id, doc_id)`
   - Delete documents with `id` in the returned `doc_id` set, but only if no embeddings remain for that document (`NOT EXISTS` on embeddings)
3) Post-check:
   - Show how many embeddings/documents remain for the same URL (should be zero)


## SQL — Locate and Delete by URL

Locate embeddings for a URL:
```sql
SELECT e.id, e.doc_id, e.chunk_index
FROM embeddings e
WHERE e.chunk_metadata->>'url' = 'https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html'
ORDER BY e.id;
```

Distinct document ids impacted:
```sql
SELECT DISTINCT e.doc_id
FROM embeddings e
WHERE e.chunk_metadata->>'url' = 'https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html';
```

Documents impacted:
```sql
SELECT d.*
FROM documents d
WHERE EXISTS (
  SELECT 1
  FROM embeddings e
  WHERE e.doc_id = d.id
    AND e.chunk_metadata->>'url' = 'https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html'
);
```

Counts:
```sql
SELECT COUNT(*) AS embeddings_to_delete
FROM embeddings e
WHERE e.chunk_metadata->>'url' = 'https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html';

SELECT COUNT(*) AS documents_affected
FROM documents d
WHERE EXISTS (
  SELECT 1
  FROM embeddings e
  WHERE e.doc_id = d.id
    AND e.chunk_metadata->>'url' = 'https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html'
);
```

Atomic deletion (CTE; recommended):
```sql
WITH del_emb AS (
  DELETE FROM embeddings
  WHERE chunk_metadata->>'url' = 'https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html'
  RETURNING doc_id
),
del_docs AS (
  DELETE FROM documents d
  WHERE d.id IN (SELECT DISTINCT doc_id FROM del_emb)
    AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.doc_id = d.id)
  RETURNING d.id
)
SELECT
  (SELECT COUNT(*) FROM del_emb)  AS embeddings_deleted,
  (SELECT COUNT(*) FROM del_docs) AS documents_deleted;
```

Two-step transaction variant:
```sql
BEGIN;

CREATE TEMP TABLE _del_doc_ids AS
SELECT DISTINCT doc_id
FROM (
  DELETE FROM embeddings
  WHERE chunk_metadata->>'url' = 'https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html'
  RETURNING doc_id
) t;

DELETE FROM documents d
WHERE d.id IN (SELECT doc_id FROM _del_doc_ids)
  AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.doc_id = d.id);

COMMIT;
```


## Exit codes and idempotency

- Exit code 0: success
- Exit code 1: unhandled exception
- Exit code 2: CLI argument error (e.g., invalid `--url-file` path)

The operations are idempotent by URL: once embeddings for a URL are removed, subsequent runs for the same URL will have no effect; documents are only deleted when they have no remaining embeddings.


## Safety and recommendations

- Always run with `--dry-run` first to inspect the number of rows and doc_ids involved.
- Consider taking a DB snapshot or running inside a transaction if you want quick rollback beyond the tool’s checks.
- If you maintain additional referencing tables (beyond `embeddings` -> `documents`), add corresponding cleanup steps before the document delete.
- Run from the repo root or export `.env` so the connector can load connection settings:
  ```bash
  set -a; source .env; set +a
  ```


## Example session

```bash
# Preview a single URL
python -m tools.delete_url_records --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/cth/HCA/2022/35.html" --dry-run

# Delete the same URL without prompt
python -m tools.delete_url_records --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/cth/HCA/2022/35.html" --yes

# Bulk delete from list
python -m tools.delete_url_records --url-file "./deletion-urls.txt" --dry-run
python -m tools.delete_url_records --url-file "./deletion-urls.txt" --yes

# Print SQLs only
python -m tools.delete_url_records --show-sql
```

If you intend to reload fresh versions of the same cases, delete first using this tool, then run the ingestion pipeline to insert the updated records.
