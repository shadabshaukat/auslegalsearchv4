# AUSLegalSearch v3 — Post-load Indexing & Metadata Strategy (TB-scale)

This repository includes two production-ready SQL scripts for optimizing JSON metadata filtering, trigram search, and vector similarity over very large tables (multi‑TB). Choose the variant that fits your rollout constraints.

Contents
- schema-post-load/create_indexes.sql — Generated columns + simple indexes
- schema-post-load/create_indexes_expression.sql — Expression-index only (no table rewrite)
- tools/bench_sql_latency.py — End-to-end latency benchmark (p50/p95) for vector/FTS/metadata filters

Key metadata fields to surface/index (from ingestion JSON)
- Included: type, jurisdiction, subjurisdiction, database, date, year, title, titles[] (array), author, citations[] (array), citation (single), countries[] (array)
- Excluded (no indexing/columns): data_quality, url

Do I need extra data load?
- No re-ingestion or external data load is required.
- If you use GENERATED ALWAYS AS STORED columns (create_indexes.sql), PostgreSQL will materialize those columns from existing JSONB values. This causes a table rewrite for existing rows; schedule during a maintenance window for 4.5TB-scale tables.
- If you want to avoid a rewrite, use expression indexes only (create_indexes_expression.sql). These build over existing JSONB and do NOT rewrite the table.

Two rollout options

Option A: Generated Columns + Simple Indexes (create_indexes.sql)
- Adds md_* columns via GENERATED ALWAYS AS STORED for hot JSON keys.
- Builds straightforward btree/GIN/trigram indexes on these md_* columns.
- Pros: simpler query predicates, predictable planner behavior, easy future extensions.
- Cons: ALTER TABLE adds STORED columns → table rewrite. Use during maintenance.
- Requirements: PostgreSQL 12+ (for STORED), pg_trgm (for trigram), vector (for pgvector).

Option B: Expression Indexes Only (create_indexes_expression.sql)
- Builds expression indexes directly over JSONB (e.g., (chunk_metadata->>'type'), to_tsvector(...), lower(...)).
- Pros: no table rewrite; safe post-load; no schema changes.
- Cons: predicates must match index expressions; slightly more complex to maintain.
- Requirements: PostgreSQL, pg_trgm. No table rewrite.

Vector index choice (pgvector)
- Prefer HNSW if pgvector >= 0.7: lower tail latency for top‑k
  CREATE INDEX CONCURRENTLY idx_embeddings_vector_hnsw_cosine
    ON public.embeddings USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
  Query-time tuning:
  SET LOCAL hnsw.ef_search = 40..100; -- higher improves recall, increases latency

- If HNSW unavailable, use IVFFLAT with larger lists:
  DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat_cosine;
  CREATE INDEX CONCURRENTLY idx_embeddings_vector_ivfflat_cosine
    ON public.embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 4096);
  Query-time tuning:
  SET LOCAL ivfflat.probes = 8..16;

HNSW vs IVFFLAT at very large scale (pros/cons)

HNSW (Hierarchical Navigable Small World, pgvector >= 0.7)
- Pros:
  - Low-latency top‑k even at very large N; typically better p95 than IVFFLAT for the same recall.
  - Good recall/latency tradeoff via hnsw.ef_search at query time (40–100 common).
  - No need to choose “lists” up front; graph adapts to the dataset.
- Cons:
  - Larger index size than IVFFLAT (graph overhead).
  - Build time is heavier and can be memory-intensive. The build uses maintenance_work_mem; if the graph does not fit, the build becomes significantly slower (spills to disk).
  - Parameters (m, ef_construction) impact build time/size: higher values improve recall but increase build cost.

IVFFLAT (Inverted File)
- Pros:
  - Faster, lighter builds; lower memory consumption.
  - Predictable memory usage; suitable when operational windows are tight.
  - Query-time control with ivfflat.probes (8–16 typical).
- Cons:
  - Must choose lists at build time; too few → poor recall/high tail, too many → larger index/slower build.
  - Typically higher p95 than HNSW for the same recall. To compensate, you increase probes (which increases latency).
  - Lists sizing requires iteration; a rule of thumb is lists ~ sqrt(N).

Choosing between them
- Latency-sensitive, stable recall -> HNSW (if you can accept longer, memory-heavier builds). Use ef_search for per-route tuning.
- Operationally constrained builds or simpler operations at massive scale -> IVFFLAT (increase lists; tune probes). Expect to iterate lists for optimal recall/latency.

Speeding up index builds (maintenance_work_mem and friends)

Why you see:
NOTICE: hnsw graph no longer fits into maintenance_work_mem … building will take significantly more time.
- The HNSW constructor uses maintenance_work_mem to hold the graph during build. If it overflows, it spills to disk, slowing builds substantially.

Options to increase maintenance_work_mem (choose one)
- Session-only (recommended for one-off index builds):
  -- In a dedicated psql session (not inside a transaction block):
  SET maintenance_work_mem = '8GB';
  CREATE INDEX CONCURRENTLY idx_embeddings_vector_hnsw_cosine
    ON public.embeddings USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

  -- Or via PGOPTIONS:
  PGOPTIONS='-c maintenance_work_mem=8GB' psql "$PGURL" -c "CREATE INDEX CONCURRENTLY idx_embeddings_vector_hnsw_cosine ON public.embeddings USING hnsw (vector vector_cosine_ops) WITH (m=16, ef_construction=200);"

- System-wide (requires superuser; remember to reload and later revert):
  ALTER SYSTEM SET maintenance_work_mem = '8GB';
  SELECT pg_reload_conf();

- Role/database scoped (persists for a user or DB):
  ALTER ROLE youruser SET maintenance_work_mem = '8GB';
  -- or:
  ALTER DATABASE yourdb SET maintenance_work_mem = '8GB';

Guidance
- Pick a value that your server can sustain for a single backend (e.g., 4–16 GB for very large HNSW builds). Don’t run multiple huge builds in parallel.
- CREATE INDEX CONCURRENTLY cannot run inside a transaction block; set the GUC in the session (SET …) before running the command.
- For IVFFLAT, maintenance_work_mem impacts build less, but increasing lists raises build time/size. Use ANALYZE afterward and validate recall vs probes.

Tuning HNSW/IVFFLAT parameters
- HNSW:
  - Build: m (graph degree, 16–32 common), ef_construction (200–400 common)
  - Query: hnsw.ef_search (40–100 common; higher → better recall, higher latency)
- IVFFLAT:
  - Build: lists (≈ sqrt(N) starting point; iterate)
  - Query: ivfflat.probes (8–16 common; higher → better recall, higher latency)

Optimized IVFFLAT recipe (near‑HNSW results with smaller memory)
- Goal: Achieve HNSW‑like p95 latency/recall using IVFFLAT with lighter builds and footprint.

1) Aggressive prefilter to shrink N
- Always apply indexed md_* equality/range filters first (type/jurisdiction/subjurisdiction/database/year/date) to reduce the candidate set before ORDER BY vector.
- This improves IVFFLAT recall and reduces required lists/probes.

2) Size lists relative to filtered N
- Rule of thumb: lists ≈ sqrt(N_filtered). Round to a nearby power of two.
- Estimate N by cohort:
  ```sql
  SELECT md_type, COUNT(*) AS n
  FROM embeddings
  GROUP BY md_type
  ORDER BY n DESC;
  ```
- Example: if “case” has ~50M rows, sqrt(50,000,000) ≈ 7071 → lists 8192 (2^13) is a good start.

3) Partial IVFFLAT indexes by cohort (recommended)
- Keep per‑index N small and tuned for your most common filters:
  ```sql
  CREATE INDEX CONCURRENTLY idx_e_vec_ivf_case
    ON public.embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 8192)
    WHERE md_type = 'case';

  CREATE INDEX CONCURRENTLY idx_e_vec_ivf_leg
    ON public.embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 4096)
    WHERE md_type = 'legislation';
  ```
- Ensure the query includes md_type = '...' so the planner chooses the matching partial index.
- You can add partial indexes by jurisdiction/subjurisdiction if a cohort is still very large.

4) Adaptive probes by top_k and lists
- Start small and increase until p95 meets target:
  - top_k ≤ 10 → probes 16–48
  - top_k ≈ 50 → probes 32–96
  - top_k ≈ 100 → probes 64–128
- Heuristic: probes ≈ min(128, max(16, lists / 128)), then refine with the benchmark.

5) Two‑phase refine (approx → exact) for higher accuracy
- Fetch a larger candidate pool with IVFFLAT (e.g., ~5–10× top_k), then re‑rank exactly:
  ```sql
  WITH params AS (
    SELECT ARRAY[:v0, :v1, ...]::vector AS qv
  ),
  approx AS (
    SELECT e.doc_id, e.chunk_index
    FROM embeddings e
    WHERE e.md_type = 'case'  -- prefilter
    ORDER BY e.vector <#> (SELECT qv FROM params)
    LIMIT :approx_k           -- e.g., top_k * 10
  )
  SELECT e.doc_id, e.chunk_index, d.source,
         (e.vector <#> (SELECT qv FROM params)) AS distance
  FROM embeddings e
  JOIN approx a USING (doc_id, chunk_index)
  JOIN documents d ON d.id = e.doc_id
  ORDER BY distance ASC
  LIMIT :top_k;
  ```
- This pattern yields recall close to HNSW while keeping IVFFLAT’s build and footprint modest.

6) Partitioning (optional but effective)
- Partition embeddings by md_type or by year range to keep per‑partition N smaller; build an IVFFLAT per partition with smaller lists and lower probes.
- Use predicates that enable partition pruning.

7) I/O and planner tips
- Set `effective_io_concurrency` high on fast storage (e.g., 200–300).
- Keep ANALYZE current so the planner chooses the right partial index.
- Ensure `work_mem` is reasonable for the final re‑rank sort when using two‑phase refine.

Build example (IVFFLAT)
```sql
-- Case law cohort (~50M rows)
CREATE INDEX CONCURRENTLY idx_e_vec_ivf_case
  ON public.embeddings USING ivfflat (vector vector_cosine_ops)
  WITH (lists = 8192)
  WHERE md_type = 'case';

-- Query example
SET LOCAL ivfflat.probes = 48;
-- Apply md_type/md_jurisdiction filters and ORDER BY vector <#> qv LIMIT :top_k
```

Validate with the benchmark
- Use `tools/bench_sql_latency.py` with your real filters, vary probes, and confirm p50/p95.
- If recall needs a bump, increase `approx_k` in two‑phase refine or raise probes modestly.

Optimized IVFFLAT recipe (near‑HNSW results with smaller memory)

Goal
- Achieve HNSW‑like p95 latency/recall using IVFFLAT with operationally lighter builds.

1) Aggressive prefilter to shrink N
- Use indexed md_* equality/range filters first (type/jurisdiction/subjurisdiction/database/year/date) to reduce the candidate set before vector ORDER BY.
- This improves IVFFLAT recall and reduces required lists/probes.

2) Size lists relative to filtered N
- Rule of thumb: lists ≈ sqrt(N_filtered). Round to a nearby power of two.
- Example to estimate N per cohort:
  SELECT md_type, COUNT(*) AS n FROM embeddings GROUP BY md_type ORDER BY n DESC;
- Pick lists per common cohort (partial index). E.g. if “case” has ~50M rows:
  sqrt(50,000,000) ≈ 7071 → lists 8192 (2^13) is a good start.

3) Partial IVFFLAT indexes by cohort (recommended)
- Build separate IVFFLAT indexes for frequent filters to keep per‑index N small:
  CREATE INDEX CONCURRENTLY idx_e_vec_ivf_case
    ON public.embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 8192)
    WHERE md_type = 'case';

  CREATE INDEX CONCURRENTLY idx_e_vec_ivf_leg
    ON public.embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 4096)
    WHERE md_type = 'legislation';

- Use the matching md_type = '...' predicate in your query so the planner can pick the partial index.
- You can create additional partial indexes by jurisdiction or subjurisdiction if a cohort is still very large.

4) Adaptive probes by top_k and lists
- Start small and grow until p95 meets target:
  - top_k ≤ 10 → probes 16–48
  - top_k 50 → probes 32–96
  - top_k 100 → probes 64–128
- Heuristic: probes ≈ min(128, max(16, lists / 128)) and then refine with the benchmark.

5) Two‑phase refine (approx → exact) for higher accuracy
- Use IVFFLAT to fetch a larger candidate pool (e.g., 5–10× top_k) fast, then re‑rank exactly without the index.

Example (cosine) two‑phase refine:
WITH params AS (
  SELECT ARRAY[:v0,:v1, ... ]::vector AS qv
),
approx AS (
  SELECT e.doc_id, e.chunk_index
  FROM embeddings e
  WHERE e.md_type = 'case'  -- prefilter
  ORDER BY e.vector <#> (SELECT qv FROM params)
  LIMIT :approx_k            -- e.g., top_k * 10
)
SELECT e.doc_id, e.chunk_index, d.source,
       (e.vector <#> (SELECT qv FROM params)) AS distance
FROM embeddings e
JOIN approx a USING (doc_id, chunk_index)
JOIN documents d ON d.id = e.doc_id
ORDER BY distance ASC
LIMIT :top_k;

- This pattern makes recall competitive with HNSW while keeping IVFFLAT build/footprint smaller.

6) Partitioning (optional but effective)
- Partition embeddings by md_type or by year range to keep per‑partition N smaller; build an IVFFLAT index per partition with smaller lists and lower probes.
- Enable partition pruning with your predicates so the planner touches only relevant partitions.

7) I/O and planner tips
- Set effective_io_concurrency high on fast storage (e.g., 200–300) to improve concurrent prefetching.
- Keep ANALYZE current so the planner estimates correctly and chooses the partial index.
- Ensure work_mem is reasonable for final re‑rank sorts when using two‑phase refine.

Build example (IVFFLAT only)
- Case law cohort (~50M rows):
  CREATE INDEX CONCURRENTLY idx_e_vec_ivf_case
    ON public.embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 8192)
    WHERE md_type = 'case';

- Query:
  SET LOCAL ivfflat.probes = 48;
  -- Apply md_type/md_jurisdiction filters and ORDER BY vector <#> qv LIMIT k

Validate with the benchmark
- Use tools/bench_sql_latency.py with your real filters, vary probes, and confirm p50/p95.
- If recall needs a bump, increase approx_k in the two‑phase refine or raise probes modestly.

Troubleshooting HNSW build failures (server closed connection / OOM)

Symptom
- During CREATE INDEX (CONCURRENTLY) USING hnsw you see:
  - NOTICE: hnsw graph no longer fits into maintenance_work_mem ...
  - server closed the connection unexpectedly
  - The connection to the server was lost

Likely cause
- The HNSW index build exceeded available memory and/or spilled heavily to disk, triggering very long runtimes and in some cases an OS OOM-kill of the PostgreSQL backend. HNSW builds are memory-intensive; when the in-memory graph does not fit into maintenance_work_mem, performance degrades sharply.

Immediate checks
- PostgreSQL logs: tail or journalctl for the postgres service to confirm backend termination.
- Kernel logs: dmesg | grep -i oom to verify OOM killer involvement.
- Free memory and swap availability during the build.

Mitigations (choose a combination)
1) Increase maintenance_work_mem for the build session
   -- One-off session (psql), not inside a transaction:
   SET maintenance_work_mem = '8GB';
   CREATE INDEX CONCURRENTLY idx_embeddings_vector_hnsw_cosine
     ON public.embeddings USING hnsw (vector vector_cosine_ops)
     WITH (m = 16, ef_construction = 200);

   -- Or via PGOPTIONS wrapper:
   PGOPTIONS='-c maintenance_work_mem=8GB' psql "$PGURL" -c "CREATE INDEX CONCURRENTLY idx_embeddings_vector_hnsw_cosine ON public.embeddings USING hnsw (vector vector_cosine_ops) WITH (m=16, ef_construction=200);"

   Pick a value your server can sustain (e.g., 4–16 GB). Do not run multiple large builds in parallel.

2) Reduce HNSW build parameters to lower memory
   - Start with smaller m and ef_construction, e.g.:
     WITH (m = 12, ef_construction = 100)
   - This reduces memory and build time at the cost of some recall; you can compensate with a slightly higher hnsw.ef_search at query time.

3) Build without CONCURRENTLY (if a maintenance window is possible)
   - CREATE INDEX (without CONCURRENTLY) can be more stable and sometimes faster; it acquires stronger locks and blocks writes but avoids some concurrent bookkeeping overhead.
   - Run in a dedicated session with increased maintenance_work_mem.

4) Ensure fast temp space for spills
   - If spills are unavoidable, point temporary files to a fast, large storage:
     SET temp_tablespaces = 'fast_tbs';
   - Verify that temp_tablespaces points to an SSD-backed tablespace with ample space.

5) Avoid competing memory load
   - Do not run other heavy memory consumers (autovacuum workers, other index builds) concurrently.
   - Consider temporarily lowering autovacuum_max_workers or scheduling builds off-peak.

6) WAL and durability considerations
   - Large index builds generate significant WAL. Ensure max_wal_size is sufficient and disk space is ample.
   - If building on a replica or a staging copy, you can tolerate longer build times without affecting production traffic.

Monitoring
- Use pg_stat_activity to track the backend and duration.
- pg_stat_progress_create_index can show progress phases for many index types; check it to monitor long-running builds:
  SELECT * FROM pg_stat_progress_create_index WHERE pid = <backend_pid>;

Build playbooks

Playbook A — HNSW (concurrent) with high memory
- Pros: no long table lock on writes; fits 24/7 systems.
- Steps:
  1) SET maintenance_work_mem = '8GB';
  2) CREATE INDEX CONCURRENTLY ... USING hnsw (m=16, ef_construction=200);
  3) ANALYZE public.embeddings;

Playbook B — HNSW (non-concurrent) in a short maintenance window
- Pros: more stable/faster; simpler internal bookkeeping.
- Steps:
  1) Schedule a brief write-free window.
  2) SET maintenance_work_mem = '8GB';
  3) CREATE INDEX ... USING hnsw (m=16, ef_construction=200);
  4) ANALYZE public.embeddings;

Playbook C — Interim IVFFLAT, then HNSW later
- Pros: quick operational win; you can query immediately.
- Steps:
  1) CREATE INDEX CONCURRENTLY ... USING ivfflat WITH (lists=4096);
  2) ANALYZE public.embeddings;
  3) Serve traffic with IVFFLAT (set ivfflat.probes=8–16).
  4) Plan a later HNSW build (A or B) when resources allow, then drop IVFFLAT if no longer needed.

Notes
- tmux is fine; the “server closed the connection” indicates the PostgreSQL backend died (likely OOM). Use the mitigations above to prevent recurrence.
- After any large build, validate with tools/bench_sql_latency.py and tune hnsw.ef_search (or ivfflat.probes) to meet p95 targets.

Session tuning for low latency
- Disable JIT for short queries:
  SET LOCAL jit = off;
- Set HNSW/IVF knobs per-query/route as shown above.

Post-index maintenance
- After building indexes or bulk loads:
  ANALYZE public.embeddings;
  ANALYZE public.documents;

Queries and predicates (write predicates to match the indexes)

If using generated columns (create_indexes.sql), prefer md_* columns where possible:
- Equality/range
  WHERE e.md_type = 'case'
    AND e.md_jurisdiction = 'cth'
    AND e.md_database = 'HCA'
    AND e.md_year = 1927
    AND e.md_date BETWEEN '1927-01-01' AND '1927-12-31'

- Array membership
  WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements_text(COALESCE(e.md_countries, e.chunk_metadata->'countries')) AS c(val)
    WHERE LOWER(c.val) = LOWER('United States')
  )

- Trigram (approximate)
  WHERE e.md_title_lc % LOWER(:title_approx)
  -- or for author
  WHERE e.md_author_lc % LOWER(:author_approx)

- documents.source trigram (approximate)
  WHERE EXISTS (
    SELECT 1
    FROM documents d
    WHERE d.id = e.doc_id
      AND d.source_lc % LOWER(:src_approx)
  )

If using expression indexes only (create_indexes_expression.sql), ensure predicates match expressions:
- Equality/range
  WHERE (e.chunk_metadata->>'type') = 'case'
    AND (e.chunk_metadata->>'jurisdiction') = 'cth'
    AND (e.chunk_metadata->>'database') = 'HCA'
    AND ((e.chunk_metadata->>'year')::int) = 1927
    AND ((e.chunk_metadata->>'date')::date) BETWEEN '1927-01-01' AND '1927-12-31'

- Array membership
  WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements_text(e.chunk_metadata->'countries') AS c(val)
    WHERE LOWER(c.val) = LOWER('United States')
  )

- Trigram (approximate)
  WHERE LOWER(coalesce(e.chunk_metadata->>'title','')) % LOWER(:title_approx)

- documents.source trigram (approximate)
  WHERE LOWER(d.source) % LOWER(:src_approx)

End-to-end latency benchmark (p50/p95)
- Use tools/bench_sql_latency.py. It measures:
  - Vector (pgvector + JSON filters)
  - FTS (documents_fts)
  - Metadata-only filters
  - Hybrid combine client-side
- Examples:
  python3 tools/bench_sql_latency.py \
    --query "Australia Peru investment agreement" \
    --top_k 10 --runs 10 --probes 10 --hnsw_ef 60 \
    --type treaty --subjurisdiction dfat --jurisdiction au --database ATS \
    --country "United States"

  python3 tools/bench_sql_latency.py \
    --query "Angelides v James Stedman Hendersons" \
    --top_k 10 --runs 10 --probes 12 \
    --type case --jurisdiction cth --database HCA \
    --title_member "Angelides v James Stedman Hendersons Sweets Ltd" \
    --citation_member "[1927] HCA 34"

Operational guidance for TB-scale
- If you cannot afford a table rewrite now:
  - Apply expression indexes first (create_indexes_expression.sql).
  - Validate with bench_sql_latency.py to reach <50ms p95.
  - Plan a maintenance window later if you want to migrate to generated columns.

- If you can accept a rewrite window:
  - Apply create_indexes.sql once. Expect table rewrite time proportional to size.
  - Build HNSW (or IVFFLAT lists) in parallel and ANALYZE.
  - Validate with the benchmark; tune probes/ef_search.

About “dbtools” syntax errors in editor
- Some IDE DB linters don’t recognize PostgreSQL JSONB operators (->, ->>) or GENERATED columns, and will flag false errors.
- These scripts are valid for PostgreSQL. Execute them via psql or a Postgres-aware admin tool. If your tool still rejects expressions, use the generated-columns script or psql.

Contact points for further tuning
- Provide EXPLAIN (ANALYZE, BUFFERS) for slow queries and the output of:
  SELECT extversion FROM pg_extension WHERE extname='vector';
- We can adjust partial indexes, lists/ef_search, or query shapes further based on real plans.

Operational updates and notes

Troubleshooting “generation expression is not immutable” (md_date)
- Some date expressions are not considered immutable by PostgreSQL when used in GENERATED ALWAYS AS STORED columns (e.g., (text)::date).
- This repo uses an immutable expression for md_date:
  make_date(
    substr((chunk_metadata->>'date'), 1, 4)::int,
    substr((chunk_metadata->>'date'), 6, 2)::int,
    substr((chunk_metadata->>'date'), 9, 2)::int
  )
- If you previously saw “generation expression is not immutable,” re-apply schema-post-load/create_indexes.sql with this expression.

Do I need extra data load if rows are added/removed?
- No. Generated STORED columns are computed from the row’s JSONB automatically on INSERT/UPDATE; DELETE removes the row.
- Indexes (btree/GIN/trigram) and vector indexes (HNSW/IVFFLAT) are maintained automatically by PostgreSQL on DML.
- The only time a table rewrite happens is when you add/alter a GENERATED STORED column (one-time rewrite to materialize). Routine DML does not require any manual sync.

Post-batch maintenance (TB-scale)
- After large ingests/deletes/updates: ANALYZE public.embeddings; ANALYZE public.documents;
- If heavy deletes caused table bloat, VACUUM (or pg_repack for online compaction).
- Consider autovacuum tuning (scale_factor/threshold/naptime) to keep up with churn at very large sizes.

Bench script coverage and session tuning
- tools/bench_sql_latency.py includes:
  - Vector + JSON filters latency
  - FTS latency
  - Metadata-only filters latency
  - Multi-run p50/p95 summary, hybrid combine
  - Optimized scenarios aligned with optimized_sql.sql:
    * cases_by_citation (exact citation match across md_citation/md_citations[])
    * cases_by_name_trgm (approx case/party name via trigram on md_title_lc)
    * cases_by_name_lev (Levenshtein refinement/alternative)
    * legislation_title_trgm (approx legislation title)
    * types_title_trgm (title search over specified types, e.g., treaty,journal)
    * ann_with_filters_doc_group (ANN with metadata filters + doc-level grouping)
    * title_search_doc_group (approx title search with doc-level grouping)
    * source_approx (approx documents.source)
- Supported filters include:
  type, jurisdiction, subjurisdiction, database, year, date_from/date_to,
  title_eq, author_eq, citation (single),
  title_member (titles[]), citation_member (citations[]), country (countries[]),
  source_approx (trigram on documents.source), author/title trigram
- Per-run tuning:
  --probes (IVFFLAT), --hnsw_ef (HNSW), --use_jit (JIT off by default recommended for tail latency)

About editor/linter errors
- Some DB tools do not parse PostgreSQL JSONB operators (->, ->>) or GENERATED columns and may show false errors.
- These scripts are valid for PostgreSQL 12+. Execute via psql or a Postgres-aware admin tool.
- If your tool blocks STORED columns, use the expression-index-only script and schedule the generated-columns variant later.

Appendix — Verification, examples, and ops tips

Verify pgvector version
- Check features (HNSW available if version ≥ 0.7):
  SELECT extversion FROM pg_extension WHERE extname='vector';

EXPLAIN ANALYZE templates (validate index usage)
- Vector + JSON filters (uses hnsw/ivfflat; shrink candidates via md_* filters):
  EXPLAIN (ANALYZE, BUFFERS)
  WITH params AS (SELECT ARRAY[...]::vector AS qv)
  SELECT e.doc_id, e.chunk_index
  FROM embeddings e
  JOIN documents d ON d.id = e.doc_id
  WHERE e.md_type = 'treaty' AND e.md_jurisdiction = 'au'
  ORDER BY e.vector <#> (SELECT qv FROM params)
  LIMIT 10;

- Metadata-only filters (btree/GIN paths):
  EXPLAIN (ANALYZE, BUFFERS)
  SELECT e.doc_id
  FROM embeddings e
  WHERE e.md_type = 'case' AND e.md_database = 'HCA' AND e.md_year = 1927
  LIMIT 10;

- FTS over documents:
  EXPLAIN (ANALYZE, BUFFERS)
  WITH q AS (SELECT plainto_tsquery('english', 'Peru agreement') AS ts)
  SELECT d.id
  FROM documents d, q
  WHERE d.document_fts @@ (SELECT ts FROM q)
  ORDER BY ts_rank(d.document_fts, (SELECT ts FROM q)) DESC
  LIMIT 10;

Bench script quickstart (p50/p95, repeatable)
- Treaty example:
  python3 tools/bench_sql_latency.py \
    --query "Australia Peru investment agreement" \
    --top_k 10 --runs 10 --probes 10 --hnsw_ef 60 \
    --type treaty --subjurisdiction dfat --jurisdiction au --database ATS \
    --country "United States"

- Case example:
  python3 tools/bench_sql_latency.py \
    --query "Angelides v James Stedman Hendersons" \
    --top_k 10 --runs 10 --probes 12 \
    --type case --jurisdiction cth --database HCA \
    --title_member "Angelides v James Stedman Hendersons Sweets Ltd" \
    --citation_member "[1927] HCA 34"

Expression-index guards (date/year)
- The expression-index file mirrors robust guards for date/year to avoid cast failures on malformed values:
  - Date: CASE + make_date(substr(... )::int, …) with pattern/digit checks
  - Year: digit-only + length checks
- Ensure your SQL predicates use the same shapes to keep queries indexable.

Running CREATE INDEX CONCURRENTLY safely
- CONCURRENTLY cannot run inside a transaction block. In psql, ensure you are not in an explicit BEGIN; issue:
  \echo :AUTOCOMMIT   -- should be on
- Set memory for the session before CREATE INDEX:
  SET maintenance_work_mem = '8GB';
- Avoid accidental interrupts (Ctrl‑C) mid-command; if interrupted, reconnect and restart the build.
- Monitor progress:
  SELECT * FROM pg_stat_progress_create_index WHERE pid = pg_backend_pid();

Temp space for spills (optional)
- If you expect spills, point temporary files to a fast tablespace with ample space:
  -- Create once (as superuser):
  -- CREATE TABLESPACE fast_tbs LOCATION '/mnt/fast_ssd_pgtemp';
  -- Session scope:
  SET temp_tablespaces = 'fast_tbs';

Autovacuum tuning (large tables, optional)
- Ensure autovacuum can keep up with churn:
  ALTER TABLE public.embeddings SET (
    autovacuum_vacuum_scale_factor = 0.02,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_vacuum_cost_limit = 2000
  );
- Tune globally in postgresql.conf if preferred; values depend on hardware and write rates.

Operational playbooks recap
- Playbook A (HNSW concurrent, high memory): SET maintenance_work_mem='8GB'; CREATE INDEX CONCURRENTLY … USING hnsw; ANALYZE.
- Playbook B (HNSW non-concurrent, short window): SET maintenance_work_mem='8GB'; CREATE INDEX … USING hnsw; ANALYZE.
- Playbook C (IVFFLAT interim): CREATE INDEX CONCURRENTLY … USING ivfflat WITH (lists=4096); ANALYZE; switch to HNSW later when resources allow.

After build
- Always ANALYZE public.embeddings; ANALYZE public.documents;
- Validate p95 with tools/bench_sql_latency.py and tune hnsw.ef_search or ivfflat.probes as needed.

Automated IVFFLAT index build (per‑cohort, partial indexes)

Overview
- Build IVFFLAT indexes per common filter cohort (e.g., md_type) to keep per‑index N smaller.
- For each cohort, choose lists ≈ next power of two ≥ ceil(sqrt(N_cohort)).
- Use CREATE INDEX CONCURRENTLY to avoid blocking writes.
- Planner will pick the matching partial index when the predicate includes the cohort filter.

1) Compute cohort sizes (pick cohorts to index)
```sql
SELECT md_type, COUNT(*) AS n
FROM public.embeddings
GROUP BY md_type
ORDER BY n DESC;
```

2) Create per‑md_type partial IVFFLAT indexes (auto‑sized lists)
- This DO block loops md_type cohorts, computes lists ≈ next power of two ≥ ceil(sqrt(n)), and creates an index if it does not exist.
```sql
DO $$
DECLARE
  rec RECORD;
  needed int;
  lists int;
  idx_name text;
BEGIN
  FOR rec IN
    SELECT md_type, COUNT(*)::bigint AS n
    FROM public.embeddings
    WHERE md_type IS NOT NULL
    GROUP BY md_type
    ORDER BY n DESC
  LOOP
    -- target = ceil(sqrt(n))
    needed := CEIL(SQRT(rec.n::numeric))::int;
    -- round up to next power of two (min 1024)
    lists := 1;
    WHILE lists < GREATEST(1024, needed) LOOP
      lists := lists * 2;
    END LOOP;

    idx_name := 'idx_e_vec_ivf_type_' || lower(regexp_replace(rec.md_type, '[^a-zA-Z0-9_]+', '_', 'g'));

    RAISE NOTICE 'Building % with lists=% for md_type=% (n=%)', idx_name, lists, rec.md_type, rec.n;

    EXECUTE format(
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I
         ON public.embeddings USING ivfflat (vector vector_cosine_ops)
         WITH (lists = %s)
         WHERE md_type = %L',
      idx_name, lists, rec.md_type
    );
  END LOOP;
END$$ LANGUAGE plpgsql;
```

3) (Optional) Also split by subjurisdiction within a very large md_type
- Example for md_type='case', further splitting by subjurisdiction:
```sql
DO $$
DECLARE
  rec RECORD;
  needed int;
  lists int;
  idx_name text;
BEGIN
  FOR rec IN
    SELECT subjurisdiction, COUNT(*)::bigint AS n
    FROM public.embeddings
    WHERE md_type = 'case' AND subjurisdiction IS NOT NULL
    GROUP BY subjurisdiction
    ORDER BY n DESC
  LOOP
    needed := CEIL(SQRT(rec.n::numeric))::int;
    lists := 1;
    WHILE lists < GREATEST(1024, needed) LOOP
      lists := lists * 2;
    END LOOP;

    idx_name := 'idx_e_vec_ivf_case_subj_' || lower(regexp_replace(rec.subjurisdiction, '[^a-zA-Z0-9_]+', '_', 'g'));

    RAISE NOTICE 'Building % with lists=% for case subjurisdiction=% (n=%)', idx_name, lists, rec.subjurisdiction, rec.n;

    EXECUTE format(
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I
         ON public.embeddings USING ivfflat (vector vector_cosine_ops)
         WITH (lists = %s)
         WHERE md_type = %L AND subjurisdiction = %L',
      idx_name, lists, 'case', rec.subjurisdiction
    );
  END LOOP;
END$$ LANGUAGE plpgsql;
```

4) Drop generic IVFFLAT if present (optional)
- After partials are in place and validated, drop the generic IVFFLAT to save space:
```sql
DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat_cosine;
```

5) Query usage and probes
- Ensure your query includes the cohort predicate so the partial index is used, then tune probes:
```sql
SET LOCAL ivfflat.probes = 48;  -- start here for top_k ~10; adjust via benchmark

WITH params AS (SELECT ARRAY[:v0, :v1, ...]::vector AS qv)
SELECT e.doc_id, e.chunk_index, d.source,
       e.vector <#> (SELECT qv FROM params) AS distance
FROM embeddings e
JOIN documents d ON d.id = e.doc_id
WHERE e.md_type = 'case'               -- matches the partial index
  AND e.md_jurisdiction = 'cth'        -- additional prefilters
ORDER BY distance ASC
LIMIT :top_k;
```

6) Validate recall with two‑phase refine (if needed)
- Fetch approx_k (5–10× top_k) using the index, then re‑rank exactly without changing lists:
```sql
WITH params AS (
  SELECT ARRAY[:v0, :v1, ...]::vector AS qv
), approx AS (
  SELECT e.doc_id, e.chunk_index
  FROM embeddings e
  WHERE e.md_type = 'case'
  ORDER BY e.vector <#> (SELECT qv FROM params)
  LIMIT (:top_k * 10)
)
SELECT e.doc_id, e.chunk_index, d.source,
       e.vector <#> (SELECT qv FROM params) AS distance
FROM embeddings e
JOIN approx a USING (doc_id, chunk_index)
JOIN documents d ON d.id = e.doc_id
ORDER BY distance ASC
LIMIT :top_k;
```

Notes
- Always ANALYZE after building indexes so the planner learns cohort selectivity.
- Use tools/bench_sql_latency.py to tune probes and confirm p50/p95.
- Consider partitioning embeddings by md_type or year for even smaller per‑index N.
