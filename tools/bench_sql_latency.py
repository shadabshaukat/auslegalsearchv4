#!/usr/bin/env python3
"""
End-to-end SQL latency benchmark for AUSLegalSearch v3

Baseline targets (existing):
- Vector similarity (pgvector) + JSON metadata filters
- Full-text search (FTS) on documents.content (documents.document_fts)
- JSON metadata-only filtering (no vector, no FTS)
- Hybrid (client-side combine)

Optimized SQL scenarios (from schema-post-load/optimized_sql.sql):
- cases_by_citation: exact citation match across md_citation or md_citations[]
- cases_by_name_trgm: approximate case/party name match using trigram on md_title_lc
- cases_by_name_lev: Levenshtein edit-distance refinement/alternative
- legislation_title_trgm: legislation by approximate title (trigram)
- types_title_trgm: title search for specified types (e.g., treaty, journal)
- ann_with_filters_doc_group: ANN vector search with metadata filters, doc-level grouping
- title_search_doc_group: approximate title search with doc-level grouping
- source_approx: approximate documents.source match (trigram)

This tool reports p50/p95 across multiple runs.

Usage examples:
  # Baseline vector/fts/metadata/hybrid
  python3 tools/bench_sql_latency.py --scenario baseline \
    --query "Angelides v James Stedman Hendersons" \
    --top_k 10 --runs 10 --probes 12

  # Cases by citation (exact; ARRAY of normalized lowercase citations)
  python3 tools/bench_sql_latency.py --scenario cases_by_citation \
    --citations "[\"[1927] hca 34\",\"(1927) clr 12\"]" \
    --runs 5

  # Cases by name (trigram), with optional filters
  python3 tools/bench_sql_latency.py --scenario cases_by_name_trgm \
    --name "Angelides v Hendersons" --jurisdiction cth --year 1927 --court HCA \
    --runs 5 --trgm_limit 0.25

  # Legislation by approximate title (trigram)
  python3 tools/bench_sql_latency.py --scenario legislation_title_trgm \
    --title "Crimes Act" --jurisdiction nsw --year 1990 --database consol_act \
    --limit 20 --runs 5

  # Treaty/journal approximate title search
  python3 tools/bench_sql_latency.py --scenario types_title_trgm \
    --title "investment agreement" --types "treaty,journal" --limit 20 --runs 5

  # ANN with metadata filters + doc-level grouping (uses query embedding)
  python3 tools/bench_sql_latency.py --scenario ann_with_filters_doc_group \
    --query "fiduciary duty in NSW" --top_k 10 --runs 5 \
    --type case --jurisdiction nsw --database NSWSC \
    --date_from 2000-01-01 --date_to 2024-12-31 \
    --author_approx "Smith" --title_approx "fiduciary" --source_approx "nswcaselaw" \
    --probes 12 --hnsw_ef 60

  # Title search with doc-level grouping
  python3 tools/bench_sql_latency.py --scenario title_search_doc_group \
    --title "Succession Act" --type legislation --jurisdiction nsw --year 2006 \
    --limit 20 --runs 5

  # Approximate source (documents.source)
  python3 tools/bench_sql_latency.py --scenario source_approx \
    --source "nsw legislation" --limit 20 --runs 5

Notes:
- Uses SQLAlchemy engine from db.connector (reads .env automatically).
- Embedding model comes from AUSLEGALSEARCH_EMBED_MODEL env or default in embedding/embedder.py.
- Tune ivfflat.probes (IVFFLAT) or hnsw.ef_search (HNSW) per run.
"""

import time
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import os, sys
# Ensure project root on sys.path when invoked as a script (e.g., "python3 tools/bench_sql_latency.py").
# This allows "from db.connector import engine" and "from embedding.embedder import Embedder" to resolve.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sqlalchemy import text
from sqlalchemy.engine import Connection
from db.connector import engine
from embedding.embedder import Embedder


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _set_ivf_probes(conn: Connection, probes: Optional[int]) -> None:
    if probes and int(probes) > 0:
        conn.execute(text("SET LOCAL ivfflat.probes = :p"), {"p": int(probes)})


def _set_hnsw_ef(conn: Connection, ef_search: Optional[int]) -> None:
    if ef_search and int(ef_search) > 0:
        conn.execute(text("SET LOCAL hnsw.ef_search = :ef"), {"ef": int(ef_search)})


def _set_session_tuning(conn: Connection, use_jit: bool = False) -> None:
    # Disable JIT for short queries to reduce tail latency
    conn.execute(text("SET LOCAL jit = :jit"), {"jit": "on" if use_jit else "off"})


def _set_trgm_limit(conn: Connection, trgm_limit: Optional[float]) -> None:
    if trgm_limit is not None:
        # guard to sane range [0,1]
        limit = max(0.0, min(1.0, float(trgm_limit)))
        conn.execute(text("SELECT set_limit(:l)"), {"l": limit})

def _set_misc(conn: Connection, disable_seqscan: bool = False, effective_io: Optional[int] = None) -> None:
    """
    Optional session-level diagnostics/tuning:
      - disable_seqscan: SET LOCAL enable_seqscan = off (forces index usage if possible; diagnostic only)
      - effective_io: SET LOCAL effective_io_concurrency = N (improves I/O prefetch on fast storage)
    """
    if disable_seqscan:
        conn.execute(text("SET LOCAL enable_seqscan = off"))
    if effective_io and int(effective_io) > 0:
        conn.execute(text("SET LOCAL effective_io_concurrency = :n"), {"n": int(effective_io)})

def _explain_vector_query(
    conn: Connection,
    query_vec: List[float],
    top_k: int,
    filters: Dict[str, Any],
    probes: Optional[int] = None,
    hnsw_ef: Optional[int] = None,
    use_jit: bool = False,
) -> None:
    """
    Print EXPLAIN ANALYZE for the vector query (baseline path) with current filters.
    Helps validate index usage (partial IVFFLAT/HNSW), estimates, and latency.
    """
    _set_session_tuning(conn, use_jit=use_jit)
    _set_ivf_probes(conn, probes)
    _set_hnsw_ef(conn, hnsw_ef)
    array_sql, arr_params = _build_vector_array_sql(query_vec)

    where_clauses = []
    params: Dict[str, Any] = {"topk": int(top_k)}
    # Match the same filter shape as run_vector_query
    if filters.get("type"):
        where_clauses.append("(e.md_type = :type_exact)")
        params["type_exact"] = str(filters["type"])
    if filters.get("jurisdiction"):
        where_clauses.append("(e.md_jurisdiction = :jurisdiction_exact)")
        params["jurisdiction_exact"] = str(filters["jurisdiction"])
    if filters.get("subjurisdiction"):
        where_clauses.append("(e.md_subjurisdiction = :subjurisdiction_exact)")
        params["subjurisdiction_exact"] = str(filters["subjurisdiction"])
    if filters.get("database"):
        where_clauses.append("(e.md_database = :database_exact)")
        params["database_exact"] = str(filters["database"])
    if filters.get("year") is not None:
        where_clauses.append("(e.md_year = :year_exact)")
        params["year_exact"] = int(filters["year"])
    if filters.get("date_from") and filters.get("date_to"):
        where_clauses.append("(e.md_date BETWEEN :df AND :dt)")
        params["df"] = str(filters["date_from"])
        params["dt"] = str(filters["date_to"])
    if filters.get("title_eq"):
        where_clauses.append("(e.md_title = :title_eq OR (e.chunk_metadata->>'title') = :title_eq)")
        params["title_eq"] = str(filters["title_eq"])
    if filters.get("author_eq"):
        where_clauses.append("(e.md_author = :author_eq OR (e.chunk_metadata->>'author') = :author_eq)")
        params["author_eq"] = str(filters["author_eq"])
    if filters.get("citation"):
        where_clauses.append("(e.md_citation = :citation_eq OR (e.chunk_metadata->>'citation') = :citation_eq)")
        params["citation_eq"] = str(filters["citation"])
    if filters.get("country"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_countries, e.chunk_metadata->'countries')) AS c(val)
              WHERE LOWER(c.val) = LOWER(:country)
            )
        """)
        params["country"] = str(filters["country"])
    if filters.get("title_member"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_titles, e.chunk_metadata->'titles')) AS t(val)
              WHERE LOWER(t.val) = LOWER(:title_member)
            )
        """)
        params["title_member"] = str(filters["title_member"])
    if filters.get("citation_member"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_citations, e.chunk_metadata->'citations')) AS c(val)
              WHERE LOWER(c.val) = LOWER(:citation_member)
            )
        """)
        params["citation_member"] = str(filters["citation_member"])
    if filters.get("author"):
        where_clauses.append("(e.md_author_lc % LOWER(:author_approx) OR LOWER(coalesce(e.chunk_metadata->>'author','')) % LOWER(:author_approx))")
        params["author_approx"] = str(filters["author"])
    if filters.get("title"):
        where_clauses.append("(e.md_title_lc % LOWER(:title_approx) OR LOWER(coalesce(e.chunk_metadata->>'title','')) % LOWER(:title_approx))")
        params["title_approx"] = str(filters["title"])
    if filters.get("source_approx"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM documents d
              WHERE d.id = e.doc_id
                AND (d.source_lc % LOWER(:src_approx) OR LOWER(d.source) % LOWER(:src_approx))
            )
        """)
        params["src_approx"] = str(filters["source_approx"])
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.update(arr_params)
    explain_sql = f"""
    EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, WAL)
    SELECT e.doc_id, e.chunk_index,
           (e.vector <=> {array_sql}) AS distance
    FROM embeddings e
    {where_sql}
    ORDER BY e.vector <=> {array_sql} ASC
    LIMIT :topk
    """
    plan = conn.execute(text(explain_sql), params).fetchall()
    print("\n--- EXPLAIN ANALYZE (vector query) ---")
    for row in plan:
        print(row[0])


def _build_vector_array_sql(vec: List[float]) -> Tuple[str, Dict[str, Any]]:
    """
    Return a single bind param usable as (:qv)::vector to maximize index-ability.
    Using parentheses ensures SQLAlchemy/psycopg2 recognizes the :qv bind and
    PostgreSQL parses the cast correctly. This typically yields better plans
    than constructing ARRAY[:v0,:v1,...]::vector in the ORDER BY expression.
    """
    s = "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"
    return "(:qv)::vector", {"qv": s}


# =========================
# Baseline existing queries
# =========================

def run_vector_query(
    conn: Connection,
    query_vec: List[float],
    top_k: int,
    filters: Dict[str, Any],
    probes: Optional[int] = None,
    hnsw_ef: Optional[int] = None,
    use_jit: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_session_tuning(conn, use_jit=use_jit)
    _set_ivf_probes(conn, probes)
    _set_hnsw_ef(conn, hnsw_ef)

    where_clauses = []
    params: Dict[str, Any] = {"topk": int(top_k)}

    # Equality filters on JSON metadata (prefer stored columns if present; fallback to JSONB)
    if filters.get("type"):
        where_clauses.append("(e.md_type = :type_exact)")
        params["type_exact"] = str(filters["type"])

    if filters.get("jurisdiction"):
        where_clauses.append("(e.md_jurisdiction = :jurisdiction_exact)")
        params["jurisdiction_exact"] = str(filters["jurisdiction"])

    if filters.get("subjurisdiction"):
        where_clauses.append("(e.md_subjurisdiction = :subjurisdiction_exact)")
        params["subjurisdiction_exact"] = str(filters["subjurisdiction"])

    if filters.get("database"):
        where_clauses.append("(e.md_database = :database_exact)")
        params["database_exact"] = str(filters["database"])

    if filters.get("year") is not None:
        where_clauses.append("(e.md_year = :year_exact)")
        params["year_exact"] = int(filters["year"])

    if filters.get("date_from") and filters.get("date_to"):
        where_clauses.append("(e.md_date BETWEEN :df AND :dt)")
        params["df"] = str(filters["date_from"])
        params["dt"] = str(filters["date_to"])

    # Exact equality on title/author/citation
    if filters.get("title_eq"):
        where_clauses.append("(e.md_title = :title_eq OR (e.chunk_metadata->>'title') = :title_eq)")
        params["title_eq"] = str(filters["title_eq"])

    if filters.get("author_eq"):
        where_clauses.append("(e.md_author = :author_eq OR (e.chunk_metadata->>'author') = :author_eq)")
        params["author_eq"] = str(filters["author_eq"])

    if filters.get("citation"):
        where_clauses.append("(e.md_citation = :citation_eq OR (e.chunk_metadata->>'citation') = :citation_eq)")
        params["citation_eq"] = str(filters["citation"])

    # Membership tests on arrays
    if filters.get("country"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_countries, e.chunk_metadata->'countries')) AS c(val)
              WHERE LOWER(c.val) = LOWER(:country)
            )
        """)
        params["country"] = str(filters["country"])

    if filters.get("title_member"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_titles, e.chunk_metadata->'titles')) AS t(val)
              WHERE LOWER(t.val) = LOWER(:title_member)
            )
        """)
        params["title_member"] = str(filters["title_member"])

    if filters.get("citation_member"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_citations, e.chunk_metadata->'citations')) AS c(val)
              WHERE LOWER(c.val) = LOWER(:citation_member)
            )
        """)
        params["citation_member"] = str(filters["citation_member"])

    # Approximate matches (trigram); rely on md_*_lc or JSONB fallback
    if filters.get("author"):
        where_clauses.append("(e.md_author_lc % LOWER(:author_approx) OR LOWER(coalesce(e.chunk_metadata->>'author','')) % LOWER(:author_approx))")
        params["author_approx"] = str(filters["author"])

    if filters.get("title"):
        where_clauses.append("(e.md_title_lc % LOWER(:title_approx) OR LOWER(coalesce(e.chunk_metadata->>'title','')) % LOWER(:title_approx))")
        params["title_approx"] = str(filters["title"])

    if filters.get("source_approx"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM documents d
              WHERE d.id = e.doc_id
                AND (d.source_lc % LOWER(:src_approx) OR LOWER(d.source) % LOWER(:src_approx))
            )
        """)
        params["src_approx"] = str(filters["source_approx"])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Vector array
    array_sql, arr_params = _build_vector_array_sql(query_vec)
    params.update(arr_params)

    sql = f"""
    SELECT e.doc_id, e.chunk_index,
           (e.vector <=> {array_sql}) AS distance,
           d.source,
           e.chunk_metadata
    FROM embeddings e
    JOIN documents d ON d.id = e.doc_id
    {where_sql}
    ORDER BY e.vector <=> {array_sql} ASC
    LIMIT :topk
    """
    rows = conn.execute(text(sql), params).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0],
            "chunk_index": r[1],
            "distance": float(r[2]),
            "source": r[3],
            "chunk_metadata": r[4],
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_fts_query(
    conn: Connection,
    query: str,
    top_k: int,
    use_jit: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_session_tuning(conn, use_jit=use_jit)
    sql = """
    WITH q AS (SELECT plainto_tsquery('english', :q) AS ts)
    SELECT d.id, d.source,
           ts_rank(d.document_fts, (SELECT ts FROM q)) AS rank
    FROM documents d, q
    WHERE d.document_fts @@ (SELECT ts FROM q)
    ORDER BY rank DESC
    LIMIT :k
    """
    rows = conn.execute(text(sql), {"q": query, "k": int(top_k)}).fetchall()
    hits = [{"doc_id": r[0], "source": r[1], "rank": float(r[2])} for r in rows]
    t1 = _now_ms()
    return hits, (t1 - t0)


def hybrid_rerank(vec_hits: List[Dict[str, Any]], fts_hits: List[Dict[str, Any]], alpha: float = 0.5, top_k: int = 10) -> List[Dict[str, Any]]:
    # Map FTS ranks to a normalized score (simple min-max)
    f_map = {}
    if fts_hits:
        ranks = [h["rank"] for h in fts_hits]
        rmin, rmax = min(ranks), max(ranks)
        for h in fts_hits:
            f_map[(h["doc_id"])] = 1.0 if rmin == rmax else (h["rank"] - rmin) / (rmax - rmin)

    # Normalize vector distances to similarity
    v_map = {}
    if vec_hits:
        dists = [h["distance"] for h in vec_hits]
        dmin, dmax = min(dists), max(dists)
        for h in vec_hits:
            sim = 1.0 if dmax == dmin else 1.0 - ((h["distance"] - dmin) / (dmax - dmin))
            v_map[(h["doc_id"], h.get("chunk_index", 0))] = sim

    # Combine by doc_id; keep the best chunk per doc for display
    combined: Dict[int, Dict[str, Any]] = {}
    for vh in vec_hits:
        key = vh["doc_id"]
        vs = v_map.get((vh["doc_id"], vh.get("chunk_index", 0)), 0.0)
        fs = f_map.get(key, 0.0)
        score = alpha * vs + (1 - alpha) * fs
        prev = combined.get(key)
        if (prev is None) or (score > prev["score"]):
            combined[key] = {"doc_id": key, "source": vh["source"], "score": score, "chunk_index": vh.get("chunk_index", 0)}

    for fh in fts_hits:
        key = fh["doc_id"]
        vs = 0.0  # if not in vec map
        fs = f_map.get(key, 0.0)
        score = alpha * vs + (1 - alpha) * fs
        prev = combined.get(key)
        if (prev is None) or (score > prev["score"]):
            combined[key] = {"doc_id": key, "source": fh["source"], "score": score}

    out = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return out


def run_metadata_filter_query(
    conn: Connection,
    top_k: int,
    filters: Dict[str, Any],
    use_jit: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Measure latency for JSONB metadata filtering only (no vector, no FTS).
    Useful to validate btree/GIN indexes and selectivity.
    """
    t0 = _now_ms()
    _set_session_tuning(conn, use_jit=use_jit)

    where_clauses = []
    params: Dict[str, Any] = {"topk": int(top_k)}

    # Equality filters on stored columns or JSONB fallback
    if filters.get("type"):
        where_clauses.append("(e.md_type = :type_exact OR (e.chunk_metadata->>'type') = :type_exact)")
        params["type_exact"] = str(filters["type"])

    if filters.get("jurisdiction"):
        where_clauses.append("(e.md_jurisdiction = :jurisdiction_exact OR (e.chunk_metadata->>'jurisdiction') = :jurisdiction_exact)")
        params["jurisdiction_exact"] = str(filters["jurisdiction"])

    if filters.get("subjurisdiction"):
        where_clauses.append("(e.md_subjurisdiction = :subjurisdiction_exact OR (e.chunk_metadata->>'subjurisdiction') = :subjurisdiction_exact)")
        params["subjurisdiction_exact"] = str(filters["subjurisdiction"])

    if filters.get("database"):
        where_clauses.append("(e.md_database = :database_exact OR (e.chunk_metadata->>'database') = :database_exact)")
        params["database_exact"] = str(filters["database"])

    if filters.get("year") is not None:
        where_clauses.append("(e.md_year = :year_exact OR ((e.chunk_metadata->>'year')::int) = :year_exact)")
        params["year_exact"] = int(filters["year"])

    if filters.get("date_from") and filters.get("date_to"):
        where_clauses.append("((e.md_date BETWEEN :df AND :dt) OR ((e.chunk_metadata->>'date')::date BETWEEN :df AND :dt))")
        params["df"] = str(filters["date_from"])
        params["dt"] = str(filters["date_to"])

    # Exact equality on title/author/citation
    if filters.get("title_eq"):
        where_clauses.append("(e.md_title = :title_eq OR (e.chunk_metadata->>'title') = :title_eq)")
        params["title_eq"] = str(filters["title_eq"])

    if filters.get("author_eq"):
        where_clauses.append("(e.md_author = :author_eq OR (e.chunk_metadata->>'author') = :author_eq)")
        params["author_eq"] = str(filters["author_eq"])

    if filters.get("citation"):
        where_clauses.append("(e.md_citation = :citation_eq OR (e.chunk_metadata->>'citation') = :citation_eq)")
        params["citation_eq"] = str(filters["citation"])

    # Membership tests on arrays
    if filters.get("country"):
        where_clauses.append("""
          EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(e.md_countries, e.chunk_metadata->'countries')) AS c(val)
            WHERE LOWER(c.val) = LOWER(:country)
          )
        """)
        params["country"] = str(filters["country"])

    if filters.get("title_member"):
        where_clauses.append("""
          EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(e.md_titles, e.chunk_metadata->'titles')) AS t(val)
            WHERE LOWER(t.val) = LOWER(:title_member)
          )
        """)
        params["title_member"] = str(filters["title_member"])

    if filters.get("citation_member"):
        where_clauses.append("""
          EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(e.md_citations, e.chunk_metadata->'citations')) AS c(val)
            WHERE LOWER(c.val) = LOWER(:citation_member)
          )
        """)
        params["citation_member"] = str(filters["citation_member"])

    # Approximate matches (trigram)
    if filters.get("author"):
        where_clauses.append("(e.md_author_lc % LOWER(:author_approx) OR LOWER(coalesce(e.chunk_metadata->>'author','')) % LOWER(:author_approx))")
        params["author_approx"] = str(filters["author"])

    if filters.get("title"):
        where_clauses.append("(e.md_title_lc % LOWER(:title_approx) OR LOWER(coalesce(e.chunk_metadata->>'title','')) % LOWER(:title_approx))")
        params["title_approx"] = str(filters["title"])

    if filters.get("source_approx"):
        where_clauses.append("""
          EXISTS (
            SELECT 1 FROM documents d
            WHERE d.id = e.doc_id
              AND (d.source_lc % LOWER(:src_approx) OR LOWER(d.source) % LOWER(:src_approx))
          )
        """)
        params["src_approx"] = str(filters["source_approx"])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
    SELECT e.doc_id, e.chunk_index, e.chunk_metadata
    FROM embeddings e
    {where_sql}
    LIMIT :topk
    """
    rows = conn.execute(text(sql), params).fetchall()
    hits = [{"doc_id": r[0], "chunk_index": r[1], "chunk_metadata": r[2]} for r in rows]
    t1 = _now_ms()
    return hits, (t1 - t0)


# ===========================================
# Optimized SQL scenarios (templated queries)
# ===========================================

def run_cases_by_citation(conn: Connection, citations: List[str]) -> Tuple[List[Dict[str, Any]], float]:
    """
    citations: list of normalized lowercase citations (e.g., ["[1927] hca 34", "(1927) 4 clr 12"])
    """
    t0 = _now_ms()
    sql = """
    WITH matched AS (
      SELECT DISTINCT e.doc_id
      FROM embeddings e
      WHERE e.md_type = 'case'
        AND (
          (e.md_citation IS NOT NULL AND lower(e.md_citation) = ANY(:citations))
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(e.md_citations, '[]'::jsonb)) AS c(val)
            WHERE lower(c.val) = ANY(:citations)
          )
        )
    ),
    names AS (
      SELECT
        e2.doc_id,
        MAX(e2.md_date) AS case_date,
        MAX(e2.md_jurisdiction) AS jurisdiction,
        MAX(e2.md_database) AS court,
        MIN(
          COALESCE(
            e2.md_citation,
            (SELECT min(x) FROM jsonb_array_elements_text(COALESCE(e2.md_citations,'[]'::jsonb)) AS x(x))
          )
        ) AS citation,
        MIN(e2.md_title) FILTER (WHERE e2.md_title IS NOT NULL) AS case_name
      FROM embeddings e2
      JOIN matched m ON m.doc_id = e2.doc_id
      GROUP BY e2.doc_id
    )
    SELECT
      n.doc_id,
      d.source AS url,
      n.jurisdiction,
      n.case_date,
      n.court,
      n.citation,
      n.case_name
    FROM names n
    JOIN documents d ON d.id = n.doc_id
    ORDER BY n.case_date DESC
    """
    rows = conn.execute(text(sql), {"citations": citations}).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0], "url": r[1], "jurisdiction": r[2], "case_date": str(r[3]),
            "court": r[4], "citation": r[5], "case_name": r[6]
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_cases_by_name_trgm(conn: Connection, name: str, jurisdiction: Optional[str], year: Optional[int], court: Optional[str], trgm_limit: Optional[float], shortlist: int) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_trgm_limit(conn, trgm_limit)
    sql = """
    WITH params AS (
      SELECT LOWER(:q::text) AS q,
             :jurisdiction::text AS jurisdiction,
             :year::int AS year,
             :court::text AS court,
             :shortlist::int AS shortlist
    ),
    seed AS (
      SELECT
        e.doc_id AS doc_id,
        d.source AS url,
        e.md_jurisdiction AS jurisdiction,
        e.md_date AS case_date,
        e.md_database AS court,
        e.md_title AS case_name,
        similarity(e.md_title_lc, p.q) AS name_similarity
      FROM embeddings e
      JOIN documents d ON d.id = e.doc_id
      CROSS JOIN params p
      WHERE e.md_type = 'case'
        AND (e.md_title_lc % p.q)
        AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
        AND (p.year IS NULL OR e.md_year = p.year)
        AND (p.court IS NULL OR e.md_database = p.court)
      ORDER BY similarity(e.md_title_lc, p.q) DESC, e.md_date DESC
      LIMIT (SELECT shortlist FROM params)
    ),
    ranked AS (
      SELECT s.*, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY name_similarity DESC, case_date DESC) AS rn
      FROM seed s
    )
    SELECT doc_id, url, jurisdiction, case_date, court, case_name, name_similarity
    FROM ranked
    WHERE rn = 1
    ORDER BY name_similarity DESC, case_date DESC
    """
    rows = conn.execute(text(sql), {"q": name, "jurisdiction": jurisdiction, "year": year, "court": court, "shortlist": int(shortlist)}).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0], "url": r[1], "jurisdiction": r[2], "case_date": str(r[3]),
            "court": r[4], "case_name": r[5], "name_similarity": float(r[6])
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_cases_by_name_lev(conn: Connection, name: str, max_dist: int, jurisdiction: Optional[str], year: Optional[int], court: Optional[str]) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    sql = """
    WITH params AS (
      SELECT LOWER(:q::text) AS q, :maxd::int AS max_dist, :jurisdiction::text AS jurisdiction, :year::int AS year, :court::text AS court
    )
    SELECT
      d.id AS doc_id,
      d.source AS url,
      e.md_jurisdiction AS jurisdiction,
      e.md_date AS case_date,
      e.md_database AS court,
      (
        SELECT string_agg(DISTINCT t, '; ')
        FROM (
          SELECT e2.md_title AS t
          FROM embeddings e2
          WHERE e2.doc_id = e.doc_id AND e2.md_title IS NOT NULL
        ) s
      ) AS case_name,
      MIN(levenshtein(e.md_title_lc, p.q)) AS distance
    FROM embeddings e
    JOIN documents d ON d.id = e.doc_id
    CROSS JOIN params p
    WHERE e.md_type = 'case'
      AND (
        levenshtein(e.md_title_lc, p.q) < p.max_dist
        OR e.md_title_lc ILIKE p.q
      )
      AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
      AND (p.year IS NULL OR e.md_year = p.year)
      AND (p.court IS NULL OR e.md_database = p.court)
    GROUP BY d.id, d.source, e.md_jurisdiction, e.md_date, e.md_database
    ORDER BY MIN(levenshtein(e.md_title_lc, p.q)) ASC, e.md_date DESC
    """
    rows = conn.execute(text(sql), {"q": name, "maxd": max_dist, "jurisdiction": jurisdiction, "year": year, "court": court}).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0], "url": r[1], "jurisdiction": r[2], "case_date": str(r[3]),
            "court": r[4], "case_name": r[5], "distance": float(r[6])
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_legislation_title_trgm(conn: Connection, title: str, jurisdiction: Optional[str], year: Optional[int], database: Optional[str], limit: int, trgm_limit: Optional[float]) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_trgm_limit(conn, trgm_limit)
    sql = """
    WITH params AS (
      SELECT LOWER(:q::text) AS q, :jurisdiction::text AS jurisdiction, :year::int AS year, :database::text AS database, :lim::int AS lim
    )
    SELECT
      d.id AS doc_id,
      d.source AS url,
      e.md_jurisdiction AS jurisdiction,
      e.md_date AS enacted_date,
      e.md_title AS name,
      e.md_database AS database,
      similarity(e.md_title_lc, p.q) AS score
    FROM embeddings e
    JOIN documents d ON d.id = e.doc_id
    CROSS JOIN params p
    WHERE e.md_type = 'legislation'
      AND (e.md_title_lc % p.q)
      AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
      AND (p.year IS NULL OR e.md_year = p.year)
      AND (p.database IS NULL OR e.md_database = p.database)
    ORDER BY score DESC, e.md_date DESC
    LIMIT (SELECT lim FROM params)
    """
    rows = conn.execute(text(sql), {"q": title, "jurisdiction": jurisdiction, "year": year, "database": database, "lim": int(limit)}).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0], "url": r[1], "jurisdiction": r[2], "enacted_date": str(r[3]),
            "name": r[4], "database": r[5], "score": float(r[6])
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_types_title_trgm(conn: Connection, title: str, types: List[str], limit: int, trgm_limit: Optional[float], shortlist: int) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_trgm_limit(conn, trgm_limit)
    sql = """
    WITH scored AS (
      SELECT
        e.doc_id,
        e.md_type AS type, e.md_title AS title, e.md_author AS author, e.md_date AS date,
        similarity(e.md_title_lc, LOWER(:q)) AS score
      FROM embeddings e
      WHERE e.md_type = ANY(:types)
        AND (e.md_title_lc % LOWER(:q))
      ORDER BY score DESC, date DESC
      LIMIT :shortlist
    ),
    ranked AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY score DESC, date DESC) AS rn
      FROM scored
    )
    SELECT r.doc_id, d.source AS url, r.type, r.title, r.author, r.date, r.score
    FROM ranked r
    JOIN documents d ON d.id = r.doc_id
    WHERE r.rn = 1
    ORDER BY r.score DESC, r.date DESC
    LIMIT :lim
    """
    rows = conn.execute(text(sql), {"q": title, "types": types, "shortlist": int(shortlist), "lim": int(limit)}).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0], "url": r[1], "type": r[2], "title": r[3], "author": r[4], "date": str(r[5]), "score": float(r[6])
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_ann_with_filters_doc_group(
    conn: Connection,
    query_vec: List[float],
    top_k: int,
    _type: Optional[str],
    database: Optional[str],
    jurisdiction: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    country: Optional[str],
    author_approx: Optional[str],
    title_approx: Optional[str],
    source_approx: Optional[str],
    probes: Optional[int],
    hnsw_ef: Optional[int],
    use_jit: bool,
    trgm_limit: Optional[float],
) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_session_tuning(conn, use_jit=use_jit)
    _set_ivf_probes(conn, probes)
    _set_hnsw_ef(conn, hnsw_ef)
    _set_trgm_limit(conn, trgm_limit)
    # Build vector array param separately to avoid DRIVER issues
    array_sql, arr_params = _build_vector_array_sql(query_vec)
    sql = f"""
    WITH params AS (
      SELECT
        :top_k::int AS top_k,
        :type::text AS type,
        :database::text AS database,
        :jurisdiction::text AS jurisdiction,
        :date_from::date AS date_from,
        :date_to::date AS date_to,
        :country::text AS country,
        LOWER(:author_approx::text) AS author_approx,
        LOWER(:title_approx::text) AS title_approx,
        LOWER(:source_approx::text) AS source_approx
    ),
    ann AS (
      SELECT
        e.doc_id,
        e.chunk_index,
        e.vector <=> {array_sql} AS distance,
        e.md_type, e.md_database, e.md_jurisdiction, e.md_date, e.md_year,
        e.md_title, e.md_author, e.md_countries
      FROM embeddings e, params p
      WHERE
        (p.type IS NULL OR e.md_type = p.type)
        AND (p.database IS NULL OR e.md_database = p.database)
        AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
        AND (p.date_from IS NULL OR e.md_date >= p.date_from)
        AND (p.date_to IS NULL OR e.md_date <= p.date_to)
        AND (
          p.country IS NULL OR EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(COALESCE(e.md_countries, '[]'::jsonb)) AS c(val)
            WHERE LOWER(c.val) = LOWER(p.country)
          )
        )
      ORDER BY e.vector <=> {array_sql}
      LIMIT (SELECT top_k FROM params)
    )
    SELECT
      a.doc_id, a.chunk_index, a.distance,
      d.source AS url,
      a.md_type, a.md_database AS court, a.md_jurisdiction AS jurisdiction,
      a.md_date AS date, a.md_year AS year,
      a.md_title AS title, a.md_author AS author
    FROM ann a
    JOIN documents d ON d.id = a.doc_id
    JOIN params p ON TRUE
    WHERE
      (p.author_approx IS NULL OR (lower(coalesce(a.md_author,'')) % p.author_approx))
      AND (p.title_approx  IS NULL OR (lower(coalesce(a.md_title,''))  % p.title_approx))
      AND (p.source_approx IS NULL OR (lower(d.source) % p.source_approx))
    ORDER BY a.distance ASC
    LIMIT (SELECT top_k FROM params)
    """
    params: Dict[str, Any] = {
        "top_k": int(top_k),
        "type": _type,
        "database": database,
        "jurisdiction": jurisdiction,
        "date_from": date_from,
        "date_to": date_to,
        "country": country,
        "author_approx": author_approx,
        "title_approx": title_approx,
        "source_approx": source_approx,
    }
    params.update(arr_params)
    rows = conn.execute(text(sql), params).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0], "chunk_index": r[1], "distance": float(r[2]),
            "url": r[3], "type": r[4], "court": r[5], "jurisdiction": r[6],
            "date": str(r[7]), "year": r[8], "title": r[9], "author": r[10],
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_title_search_doc_group(conn: Connection, title: str, _type: Optional[str], jurisdiction: Optional[str], database: Optional[str], year: Optional[int], limit: int, trgm_limit: Optional[float], shortlist: int) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_trgm_limit(conn, trgm_limit)
    sql = """
    WITH params AS (
      SELECT LOWER(:q::text) AS q,
             :type::text AS type,
             :jurisdiction::text AS jurisdiction,
             :database::text AS database,
             :year::int AS year,
             :lim::int AS lim,
             :shortlist::int AS shortlist
    ),
    scored AS (
      SELECT
        e.doc_id, d.source AS url,
        e.md_type, e.md_jurisdiction, e.md_database, e.md_year, e.md_date, e.md_title,
        similarity(e.md_title_lc, p.q) AS score
      FROM embeddings e
      JOIN documents d ON d.id = e.doc_id
      CROSS JOIN params p
      WHERE (p.type IS NULL OR e.md_type = p.type)
        AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
        AND (p.database IS NULL OR e.md_database = p.database)
        AND (p.year IS NULL OR e.md_year = p.year)
        AND e.md_title_lc % p.q
      ORDER BY similarity(e.md_title_lc, p.q) DESC, e.md_date DESC
      LIMIT (SELECT shortlist FROM params)
    ),
    ranked AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY score DESC) AS rn
      FROM scored
    )
    SELECT doc_id, url, md_type, md_jurisdiction, md_database AS court, md_year, md_date, md_title AS best_title, score
    FROM ranked
    WHERE rn = 1
    ORDER BY score DESC, md_date DESC
    LIMIT (SELECT lim FROM params)
    """
    rows = conn.execute(text(sql), {"q": title, "type": _type, "jurisdiction": jurisdiction, "database": database, "year": year, "lim": int(limit), "shortlist": int(shortlist)}).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0], "url": r[1], "type": r[2], "jurisdiction": r[3], "court": r[4],
            "year": r[5], "date": str(r[6]), "best_title": r[7], "score": float(r[8])
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_source_approx(conn: Connection, source: str, limit: int, trgm_limit: Optional[float]) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_trgm_limit(conn, trgm_limit)
    sql = """
    SELECT id AS doc_id, source AS url, similarity(lower(source), LOWER(:q)) AS score
    FROM documents
    WHERE lower(source) % LOWER(:q)
    ORDER BY similarity(lower(source), LOWER(:q)) DESC
    LIMIT :lim::int
    """
    rows = conn.execute(text(sql), {"q": source, "lim": int(limit)}).fetchall()
    hits = [{"doc_id": r[0], "url": r[1], "score": float(r[2])} for r in rows]
    t1 = _now_ms()
    return hits, (t1 - t0)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s)-1, int(round((p/100.0) * (len(s)-1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser(description="Benchmark SQL latency for AUSLegalSearch (baseline + optimized scenarios).")
    ap.add_argument("--scenario", default="baseline", choices=[
        "baseline",
        "cases_by_citation",
        "cases_by_name_trgm",
        "cases_by_name_lev",
        "legislation_title_trgm",
        "types_title_trgm",
        "ann_with_filters_doc_group",
        "title_search_doc_group",
        "source_approx",
    ], help="Which scenario to run.")
    # Common/baseline
    ap.add_argument("--query", default=None, help="Search query text (baseline vector/fts/hybrid; or for ANN doc group scenario)")
    ap.add_argument("--top_k", type=int, default=10, help="Top K results")
    ap.add_argument("--probes", type=int, default=10, help="ivfflat.probes for pgvector")
    ap.add_argument("--hnsw_ef", type=int, default=0, help="hnsw.ef_search for HNSW (pgvector >= 0.7). 0 to skip")
    ap.add_argument("--runs", type=int, default=5, help="Number of repetitions per test to compute p50/p95")
    ap.add_argument("--use_jit", action="store_true", help="Enable JIT (off by default for lower tail)")
    ap.add_argument("--trgm_limit", type=float, default=None, help="Set trigram similarity threshold via set_limit(value)")
    ap.add_argument("--explain", action="store_true", help="Print EXPLAIN ANALYZE for the vector query (baseline/ANN)")
    ap.add_argument("--disable_seqscan", action="store_true", help="SET LOCAL enable_seqscan=off (diagnostic)")
    ap.add_argument("--effective_io", type=int, default=0, help="SET LOCAL effective_io_concurrency to this value if >0")

    # Filters (baseline and optimized)
    ap.add_argument("--type", dest="type_", default=None, help="Exact metadata type (case, legislation, journal, treaty, etc.)")
    ap.add_argument("--jurisdiction", default=None, help="Exact metadata jurisdiction (e.g., au, cth, nsw)")
    ap.add_argument("--subjurisdiction", default=None, help="Exact metadata subjurisdiction")
    ap.add_argument("--database", default=None, help="Exact metadata database (e.g., HCA, consol_act, UNSWLawJl, ATS, NSWSC)")
    ap.add_argument("--year", type=int, default=None, help="Exact metadata year")
    ap.add_argument("--date_from", default=None, help="YYYY-MM-DD")
    ap.add_argument("--date_to", default=None, help="YYYY-MM-DD")
    ap.add_argument("--title_eq", default=None, help="Exact title")
    ap.add_argument("--author_eq", default=None, help="Exact author")
    ap.add_argument("--citation", default=None, help="Exact single 'citation' field (journals)")

    # Membership filters
    ap.add_argument("--title_member", default=None, help="Check membership in titles[]")
    ap.add_argument("--citation_member", default=None, help="Check membership in citations[]")
    ap.add_argument("--country", default=None, help="Country membership (countries[])")

    # Approximate (trigram) filters
    ap.add_argument("--source_approx", default=None, help="Approximate match for documents.source (baseline filter)")

    # Optimized scenario params
    ap.add_argument("--citations", default=None, help='JSON array string of lowercase citations for cases_by_citation, e.g., "[\\"[1927] hca 34\\", \\"(1927) 4 clr 12\\"]"')
    ap.add_argument("--name", default=None, help="Name/party string for cases_by_name* scenarios")
    ap.add_argument("--max_dist", type=int, default=3, help="Max Levenshtein distance for cases_by_name_lev")
    ap.add_argument("--title", default=None, help="Title string for legislation_title_trgm/types_title_trgm/title_search_doc_group")
    ap.add_argument("--types", default=None, help="Comma-separated list for types_title_trgm, e.g., 'treaty,journal'")
    ap.add_argument("--limit", type=int, default=20, help="Limit for some optimized scenarios")
    ap.add_argument("--shortlist", type=int, default=1000, help="Shortlist size for trigram scenarios before doc-level ranking")

    # Extra approx filters for ANN doc group scenario
    ap.add_argument("--author", default=None, help="Approximate author for ann_with_filters_doc_group/title_search_doc_group (baseline also uses it)")
    ap.add_argument("--title_approx", default=None, help="Approximate title (ANN doc group)")
    ap.add_argument("--source", default=None, help="Approximate documents.source for source_approx scenario")

    args = ap.parse_args()

    with engine.begin() as conn:
        # Optional GUC warmup
        _set_session_tuning(conn, use_jit=args.use_jit)
        _set_ivf_probes(conn, args.probes)
        _set_hnsw_ef(conn, args.hnsw_ef)
        _set_misc(conn, disable_seqscan=bool(args.disable_seqscan), effective_io=(args.effective_io if args.effective_io and int(args.effective_io) > 0 else None))
        conn.execute(text("SELECT 1"))

        times: List[float] = []
        last_hits: List[Dict[str, Any]] = []

        def _summary_line(name: str, values: List[float]) -> str:
            if not values:
                return f"{name}: n=0"
            return f"{name}: p50={_percentile(values,50):.2f} ms  p95={_percentile(values,95):.2f} ms  n={len(values)}"

        if args.scenario == "baseline":
            # Build embedding for baseline vector/hybrid
            if not args.query:
                raise SystemExit("--query is required for baseline scenario")
            embedder = Embedder()
            query_vec = embedder.embed([args.query])[0].tolist()

            filters = {
                "type": args.type_,
                "jurisdiction": args.jurisdiction,
                "subjurisdiction": args.subjurisdiction,
                "database": args.database,
                "year": args.year,
                "date_from": args.date_from,
                "date_to": args.date_to,
                "title_eq": args.title_eq,
                "author_eq": args.author_eq,
                "citation": args.citation,
                "title_member": args.title_member,
                "citation_member": args.citation_member,
                "country": args.country,
                "source_approx": args.source_approx,
                "author": args.author,
                "title": args.title,
            }

            # Optional: print query plan to ensure the correct vector index is used (e.g., partial IVFFLAT by md_type).
            if args.explain:
                _explain_vector_query(conn, query_vec, args.top_k, filters, probes=args.probes, hnsw_ef=args.hnsw_ef, use_jit=args.use_jit)

            vec_times: List[float] = []
            fts_times: List[float] = []
            md_times: List[float] = []
            last_vec_hits: List[Dict[str, Any]] = []
            last_fts_hits: List[Dict[str, Any]] = []
            last_md_hits: List[Dict[str, Any]] = []

            for _ in range(max(1, args.runs)):
                vec_hits, vec_ms = run_vector_query(conn, query_vec, args.top_k, filters, probes=args.probes, hnsw_ef=args.hnsw_ef, use_jit=args.use_jit)
                vec_times.append(vec_ms); last_vec_hits = vec_hits

            for _ in range(max(1, args.runs)):
                fts_hits, fts_ms = run_fts_query(conn, args.query, args.top_k, use_jit=args.use_jit)
                fts_times.append(fts_ms); last_fts_hits = fts_hits

            for _ in range(max(1, args.runs)):
                md_hits, md_ms = run_metadata_filter_query(conn, args.top_k, filters, use_jit=args.use_jit)
                md_times.append(md_ms); last_md_hits = md_hits

            # Hybrid combine
            t0 = _now_ms()
            hybrid = hybrid_rerank(last_vec_hits, last_fts_hits, alpha=0.5, top_k=args.top_k)
            hybrid_ms = _now_ms() - t0

            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("Vector (pgvector+filters)", vec_times))
            print(_summary_line("FTS (documents_fts)", fts_times))
            print(_summary_line("Metadata-only JSON filters", md_times))
            print(f"Hybrid combine (client): {hybrid_ms:.2f} ms")

            print("\n--- Top Vector Hits ---")
            for h in last_vec_hits[:args.top_k]:
                cm = h.get("chunk_metadata") or {}
                print(f"doc_id={h['doc_id']} chunk={h['chunk_index']} dist={h['distance']:.4f} src={h['source'][:90]} meta_title={(cm.get('title') if isinstance(cm, dict) else None)}")

            print("\n--- Top FTS Hits ---")
            for h in last_fts_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} rank={h['rank']:.4f} src={h['source'][:120]}")

            print("\n--- Top Metadata-only Hits ---")
            for h in last_md_hits[:args.top_k]:
                cm = h.get("chunk_metadata") or {}
                title = cm.get("title") if isinstance(cm, dict) else None
                md_type = cm.get("type") if isinstance(cm, dict) else None
                database = cm.get("database") if isinstance(cm, dict) else None
                print(f"doc_id={h['doc_id']} chunk={h['chunk_index']} title={title} type={md_type} database={database}")

            print("\n--- Top Hybrid (by doc) ---")
            for h in hybrid[:args.top_k]:
                print(f"doc_id={h['doc_id']} score={h['score']:.4f} src={h['source'][:120]}")

            return

        # Optimized scenarios
        if args.scenario == "cases_by_citation":
            if not args.citations:
                raise SystemExit("--citations (JSON array string) is required")
            citations = json.loads(args.citations)
            for _ in range(max(1, args.runs)):
                hits, ms = run_cases_by_citation(conn, citations)
                times.append(ms); last_hits = hits

            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("cases_by_citation", times))
            print("\n--- Top Results ---")
            for h in last_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} date={h['case_date']} court={h['court']} citation={h['citation']} url={h['url'][:120]} name={h['case_name'][:80] if h['case_name'] else ''}")
            return

        if args.scenario == "cases_by_name_trgm":
            if not args.name:
                raise SystemExit("--name is required")
            for _ in range(max(1, args.runs)):
                hits, ms = run_cases_by_name_trgm(conn, args.name, args.jurisdiction, args.year, args.database, args.trgm_limit, args.shortlist)
                times.append(ms); last_hits = hits
            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("cases_by_name_trgm", times))
            print("\n--- Top Results ---")
            for h in last_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} sim={h['name_similarity']:.4f} date={h['case_date']} court={h['court']} url={h['url'][:120]} name={(h['case_name'] or '')[:80]}")
            return

        if args.scenario == "cases_by_name_lev":
            if not args.name:
                raise SystemExit("--name is required")
            for _ in range(max(1, args.runs)):
                hits, ms = run_cases_by_name_lev(conn, args.name, args.max_dist, args.jurisdiction, args.year, args.database)
                times.append(ms); last_hits = hits
            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("cases_by_name_lev", times))
            print("\n--- Top Results ---")
            for h in last_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} edit_distance={h['distance']:.0f} date={h['case_date']} court={h['court']} url={h['url'][:120]} name={(h['case_name'] or '')[:80]}")
            return

        if args.scenario == "legislation_title_trgm":
            if not args.title:
                raise SystemExit("--title is required")
            for _ in range(max(1, args.runs)):
                hits, ms = run_legislation_title_trgm(conn, args.title, args.jurisdiction, args.year, args.database, args.limit, args.trgm_limit)
                times.append(ms); last_hits = hits
            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("legislation_title_trgm", times))
            print("\n--- Top Results ---")
            for h in last_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} score={h['score']:.4f} date={h['enacted_date']} url={h['url'][:120]} name={(h['name'] or '')[:100]}")
            return

        if args.scenario == "types_title_trgm":
            if not args.title or not args.types:
                raise SystemExit("--title and --types are required")
            types = [t.strip() for t in args.types.split(",") if t.strip()]
            for _ in range(max(1, args.runs)):
                hits, ms = run_types_title_trgm(conn, args.title, types, args.limit, args.trgm_limit, args.shortlist)
                times.append(ms); last_hits = hits
            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("types_title_trgm", times))
            print("\n--- Top Results ---")
            for h in last_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} type={h['type']} score={h['score']:.4f} date={h['date']} url={h['url'][:120]} title={(h['title'] or '')[:100]}")
            return

        if args.scenario == "ann_with_filters_doc_group":
            if not args.query:
                raise SystemExit("--query is required (used for embedding)")
            embedder = Embedder()
            qv = embedder.embed([args.query])[0].tolist()
            for _ in range(max(1, args.runs)):
                hits, ms = run_ann_with_filters_doc_group(
                    conn,
                    query_vec=qv,
                    top_k=args.top_k,
                    _type=args.type_,
                    database=args.database,
                    jurisdiction=args.jurisdiction,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    country=args.country,
                    author_approx=args.author,
                    title_approx=args.title_approx,
                    source_approx=args.source_approx,
                    probes=args.probes,
                    hnsw_ef=args.hnsw_ef,
                    use_jit=args.use_jit,
                    trgm_limit=args.trgm_limit,
                )
                times.append(ms); last_hits = hits
            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("ann_with_filters_doc_group", times))
            print("\n--- Top Results ---")
            for h in last_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} dist={h['distance']:.4f} url={h['url'][:120]} type={h['type']} title={(h['title'] or '')[:90]}")
            return

        if args.scenario == "title_search_doc_group":
            if not args.title:
                raise SystemExit("--title is required")
            for _ in range(max(1, args.runs)):
                hits, ms = run_title_search_doc_group(conn, args.title, args.type_, args.jurisdiction, args.database, args.year, args.limit, args.trgm_limit, args.shortlist)
                times.append(ms); last_hits = hits
            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("title_search_doc_group", times))
            print("\n--- Top Results ---")
            for h in last_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} score={h['score']:.4f} url={h['url'][:120]} type={h['type']} best_title={(h['best_title'] or '')[:90]}")
            return

        if args.scenario == "source_approx":
            src = args.source or args.source_approx
            if not src:
                raise SystemExit("--source (or --source_approx) is required")
            for _ in range(max(1, args.runs)):
                hits, ms = run_source_approx(conn, src, args.limit, args.trgm_limit)
                times.append(ms); last_hits = hits
            print("\n--- Latency Summary (ms) ---")
            print(_summary_line("source_approx", times))
            print("\n--- Top Results ---")
            for h in last_hits[:args.top_k]:
                print(f"doc_id={h['doc_id']} score={h['score']:.4f} url={h['url'][:140]}")
            return


if __name__ == "__main__":
    main()
