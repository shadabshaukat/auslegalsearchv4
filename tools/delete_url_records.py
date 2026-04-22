#!/usr/bin/env python3
"""
Delete all records for a given URL across embeddings (by chunk_metadata->>'url')
and the corresponding documents rows in auslegalsearchv3.

- Locates embeddings where chunk_metadata->>'url' = :url
- Deletes those embeddings
- Deletes any documents referenced by the deleted embeddings' doc_id
  (only where there are no remaining embeddings for that doc_id)

Environment/DB connection:
- Inherits DB connection settings from .env via db.connector's built-in .env loader.
- Alternatively, export AUSLEGALSEARCH_DB_URL or AUSLEGALSEARCH_DB_* (HOST/PORT/USER/PASSWORD/NAME).

Usage:
  # Dry run: preview what would be deleted
  python -m tools.delete_url_records --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html" --dry-run

  # Delete with confirmation prompt
  python -m tools.delete_url_records --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html"

  # Delete without prompt
  python -m tools.delete_url_records --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCA/2011/428.html" --yes

  # Bulk delete from a file containing one URL per line (blank lines and lines starting with # are ignored)
  python -m tools.delete_url_records --url-file "/abs/path/to/urls.txt" --dry-run
  python -m tools.delete_url_records --url-file "/abs/path/to/urls.txt" --yes
"""

from __future__ import annotations

import argparse
import sys
import os
from typing import List, Dict, Any, Tuple

from sqlalchemy import text
from db.connector import engine  # inherits .env via db.connector's loader


SQL_PREVIEW = {
    "doc_ids": """
        SELECT DISTINCT e.doc_id
        FROM embeddings e
        WHERE e.chunk_metadata->>'url' = :url
    """,
    "count_embeddings": """
        SELECT COUNT(*) AS emb_count
        FROM embeddings e
        WHERE e.chunk_metadata->>'url' = :url
    """,
    "count_documents": """
        SELECT COUNT(*) AS doc_count
        FROM documents d
        WHERE EXISTS (
            SELECT 1
            FROM embeddings e
            WHERE e.doc_id = d.id AND e.chunk_metadata->>'url' = :url
        )
    """,
    "sample_embeddings": """
        SELECT e.id, e.doc_id, e.chunk_index
        FROM embeddings e
        WHERE e.chunk_metadata->>'url' = :url
        ORDER BY e.id
        LIMIT 10
    """,
}

# Two-phase deletion (atomic in one transaction):
# 1) Delete embeddings for the URL (returning (id, doc_id))
# 2) Delete documents for those doc_ids, but only if no embeddings remain for that doc_id
SQL_DELETE_EMBEDDINGS = """
    DELETE FROM embeddings
    WHERE chunk_metadata->>'url' = :url
    RETURNING id, doc_id
"""

SQL_DELETE_DOCUMENTS = """
    DELETE FROM documents d
    WHERE d.id = ANY(:ids)
      AND NOT EXISTS (
          SELECT 1 FROM embeddings e WHERE e.doc_id = d.id
      )
    RETURNING d.id
"""


def preview(url: str) -> Dict[str, Any]:
    """
    Preview what will be deleted for a given URL.
    """
    out: Dict[str, Any] = {}
    with engine.begin() as conn:
        emb_count = conn.execute(text(SQL_PREVIEW["count_embeddings"]), {"url": url}).scalar() or 0
        doc_count = conn.execute(text(SQL_PREVIEW["count_documents"]), {"url": url}).scalar() or 0
        doc_ids = [r[0] for r in conn.execute(text(SQL_PREVIEW["doc_ids"]), {"url": url}).fetchall()]
        sample = [
            {"embedding_id": r[0], "doc_id": r[1], "chunk_index": r[2]}
            for r in conn.execute(text(SQL_PREVIEW["sample_embeddings"]), {"url": url}).fetchall()
        ]
    out["url"] = url
    out["embeddings_count"] = int(emb_count)
    out["documents_count"] = int(doc_count)
    out["doc_ids"] = doc_ids
    out["sample_embeddings"] = sample
    return out


def delete_for_url(url: str) -> Tuple[int, int, List[int], List[int]]:
    """
    Delete embeddings (by url) and then documents for the affected doc_ids (safely).
    Returns:
      (embeddings_deleted, documents_deleted, deleted_embedding_ids, deleted_document_ids)
    """
    with engine.begin() as conn:
        # Phase 1: delete embeddings referencing the URL
        emb_rows = conn.execute(text(SQL_DELETE_EMBEDDINGS), {"url": url}).fetchall()
        deleted_emb_ids = [int(r[0]) for r in emb_rows]
        doc_ids = sorted({int(r[1]) for r in emb_rows})

        # Phase 2: delete documents whose id is among deleted embeddings' doc_ids,
        #          but only if no embeddings remain for that document.
        deleted_doc_ids: List[int] = []
        if doc_ids:
            res_docs = conn.execute(text(SQL_DELETE_DOCUMENTS), {"ids": doc_ids}).fetchall()
            deleted_doc_ids = [int(r[0]) for r in res_docs]

    return (len(deleted_emb_ids), len(deleted_doc_ids), deleted_emb_ids, deleted_doc_ids)


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _read_urls_from_file(path: str) -> List[str]:
    """
    Read URLs line-by-line, ignoring blanks and lines starting with '#'.
    Returns unique URLs preserving order.
    """
    urls: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = (raw or "").strip()
            if not line or line.startswith("#"):
                continue
            if line not in seen:
                urls.append(line)
                seen.add(line)
    return urls


def _sql_quote(value: str) -> str:
    """
    Minimal SQL single-quote escaping for literal printing in --show-sql.
    Do not use for parameter binding; engine uses parameters for real execution.
    """
    return (value or "").replace("'", "''")


def main():
    ap = argparse.ArgumentParser(description="Locate and delete records for a given URL (or a list of URLs) across embeddings and documents.")
    ap.add_argument("--url", required=False, help="URL value to match against embeddings.chunk_metadata->>'url'")
    ap.add_argument("--url-file", required=False, help="Path to a file containing URL(s), one per line")
    ap.add_argument("--dry-run", action="store_true", help="Preview only; do not delete")
    ap.add_argument("--yes", action="store_true", help="Proceed without confirmation")
    ap.add_argument("--show-sql", action="store_true", help="Print the SQL statements for reference and exit")
    args = ap.parse_args()

    # show-sql printing is handled after URL resolution so we can print literal SQL when URLs are provided

    # Resolve URL list (single or bulk)
    urls: List[str] = []
    if args.url_file:
        if not os.path.isfile(args.url_file):
            print(f"[error] url-file not found: {args.url_file}")
            sys.exit(2)
        urls = _read_urls_from_file(args.url_file)
    elif args.url:
        urls = [args.url.strip()]
    else:
        print("[error] Either --url or --url-file must be provided.")
        sys.exit(2)

    # --show-sql: if URLs are provided, print literal SQL per URL; otherwise print template with :url
    if args.show_sql:
        if urls:
            max_urls = int(os.environ.get("AUSLEGALSEARCH_SHOWSQL_MAXURLS", "20"))
            to_print = urls[:max_urls]
            for u in to_print:
                uq = _sql_quote(u)
                print("SQL to locate and delete by URL (literal):")
                print("\n-- Locate target doc_ids")
                print(f"SELECT DISTINCT e.doc_id\n        FROM embeddings e\n        WHERE e.chunk_metadata->>'url' = '{uq}'")
                print("\n-- Count embeddings to be deleted")
                print(f"SELECT COUNT(*) AS emb_count\n        FROM embeddings e\n        WHERE e.chunk_metadata->>'url' = '{uq}'")
                print("\n-- Count documents referencing those embeddings")
                print("SELECT COUNT(*) AS doc_count\n        FROM documents d\n        WHERE EXISTS (\n            SELECT 1\n            FROM embeddings e\n            WHERE e.doc_id = d.id AND e.chunk_metadata->>'url' = '" + uq + "'\n        )")
                print("\n-- Sample embeddings")
                print(f"SELECT e.id, e.doc_id, e.chunk_index\n        FROM embeddings e\n        WHERE e.chunk_metadata->>'url' = '{uq}'\n        ORDER BY e.id\n        LIMIT 10")
                print("\n-- Atomic delete (CTE; recommended)")
                print("WITH del_emb AS (\n  DELETE FROM embeddings\n  WHERE chunk_metadata->>'url' = '" + uq + "'\n  RETURNING doc_id\n),\n del_docs AS (\n  DELETE FROM documents d\n  WHERE d.id IN (SELECT DISTINCT doc_id FROM del_emb)\n    AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.doc_id = d.id)\n  RETURNING d.id\n)\nSELECT\n  (SELECT COUNT(*) FROM del_emb)  AS embeddings_deleted,\n  (SELECT COUNT(*) FROM del_docs) AS documents_deleted;")
                print("\n-- Two-step variant (transactional)")
                print("BEGIN;\n\nCREATE TEMP TABLE _del_doc_ids AS\nSELECT DISTINCT doc_id\nFROM (\n  DELETE FROM embeddings\n  WHERE chunk_metadata->>'url' = '" + uq + "'\n  RETURNING doc_id\n) t;\n\nDELETE FROM documents d\nWHERE d.id IN (SELECT doc_id FROM _del_doc_ids)\n  AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.doc_id = d.id);\n\nCOMMIT;")
                print("\n" + ("-" * 80) + "\n")
            if len(urls) > max_urls:
                print(f"-- Note: {len(urls)-max_urls} additional URL(s) omitted from --show-sql output.")
        else:
            # Template (no URL provided)
            print("SQL to locate and delete by URL (template with :url):")
            print("\n-- Locate target doc_ids")
            print(SQL_PREVIEW["doc_ids"].strip())
            print("\n-- Count embeddings to be deleted")
            print(SQL_PREVIEW["count_embeddings"].strip())
            print("\n-- Count documents referencing those embeddings")
            print(SQL_PREVIEW["count_documents"].strip())
            print("\n-- Sample embeddings")
            print(SQL_PREVIEW["sample_embeddings"].strip())
            print("\n-- Delete embeddings (returning id, doc_id)")
            print(SQL_DELETE_EMBEDDINGS.strip())
            print("\n-- Delete documents for the deleted doc_ids (only if no embeddings remain)")
            print(SQL_DELETE_DOCUMENTS.strip())
        sys.exit(0)

    if not urls:
        print("[info] No URL(s) to process.")
        return

    # Preview phase
    previews: List[Dict[str, Any]] = []
    total_emb = 0
    total_doc = 0
    for u in urls:
        info = preview(u)
        previews.append(info)
        total_emb += info.get("embeddings_count", 0)
        total_doc += info.get("documents_count", 0)
        # Per-URL concise preview
        print(f"[preview] url={u} emb_del={info.get('embeddings_count',0)} docs_aff={info.get('documents_count',0)}")

    print(f"[preview-summary] urls={len(urls)} embeddings_to_delete={total_emb} documents_affected={total_doc}")

    if args.dry_run:
        print("[dry-run] No changes made.")
        return

    # Confirmation (single prompt for bulk)
    if not args.yes:
        if not _confirm(f"Proceed to delete embeddings/documents for {len(urls)} URL(s)?"):
            print("Aborted.")
            return

    # Deletion phase
    grand_emb = 0
    grand_doc = 0
    err_urls: List[str] = []
    for u in urls:
        try:
            emb_del, doc_del, emb_ids, doc_ids = delete_for_url(u)
            grand_emb += emb_del
            grand_doc += doc_del
            print(f"[deleted] url={u} emb={emb_del} docs={doc_del}")
        except Exception as e:
            err_urls.append(u)
            print(f"[error] url={u} :: {e}")

    print(f"[deleted-summary] urls={len(urls)} emb_total={grand_emb} docs_total={grand_doc} errors={len(err_urls)}")
    if err_urls:
        print("[errors] The following URL(s) failed during deletion:")
        for u in err_urls[:50]:
            print(f"  {u}")
        if len(err_urls) > 50:
            print(f"  ... and {len(err_urls)-50} more")

    # Optional post-check for a few URLs
    check_n = min(5, len(urls))
    for u in urls[:check_n]:
        leftover = preview(u)
        print(f"[post-check] url={u} remaining_embeddings={leftover.get('embeddings_count',0)} remaining_docs_aff={leftover.get('documents_count',0)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)
