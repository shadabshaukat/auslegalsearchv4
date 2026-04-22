-- =====================================================================
-- AUSLegalSearch v3 — Optimized SQL Templates (Documented + Snippets)
--
-- Purpose:
--   Curated query templates for high-performance legal search, aligned
--   with the post-load schema/indexing strategy:
--     - Generated STORED columns: md_* derived from embeddings.chunk_metadata
--     - BTree equality/range on: md_type, md_jurisdiction, md_database,
--       md_date, md_year, md_title, md_author, md_citation
--     - Trigram GIN on: md_title_lc, md_author_lc (pg_trgm)
--     - GIN on arrays: md_countries, md_titles, md_citations
--     - Vector ANN (pgvector): IVFFLAT/HNSW with vector_cosine_ops, using
--       cosine operator (<=>) for index use
--     - documents.source trigram: lower(source) GIN (pg_trgm)
--
-- Notes:
--   - Prefer equality/range filters on md_* STORED columns to enable
--     selective btree pruning and partial vector index usage (e.g., md_type).
--   - ANN uses cosine operator (<=>) to match vector_cosine_ops opclass.
--   - For IVFFLAT, tune ivfflat.probes at runtime. For HNSW, tune hnsw.ef_search.
--   - Apply cohort prefilters (md_type/jurisdiction/database/year/date) to shrink
--     candidates before ANN ordering.
--
-- Driver-agnostic vs Python bindings:
--   - The canonical templates below use positional parameters $1,$2,... for drivers
--     and psql/prepared statements.
--   - Below each query we include:
--       • Driver-agnostic usage (psql/positional) and optional alternative forms
--       • Python (SQLAlchemy) snippet using named binds and Python-native types
-- =====================================================================


-- =====================================================================
-- 1) Cases by Citation (Exact)
-- What it does:
--   Finds case documents by exact citation match from either the single
--   md_citation or any member of md_citations[] (normalized lowercase).
-- Why it’s fast (indexes used):
--   - md_type = 'case' leverages btree on md_type
--   - Equality test on md_citation (btree)
--   - Membership test on md_citations via GIN (jsonb array)
--   - No vector search here; purely metadata filters and join
--
-- Driver-agnostic (positional):
--   PARAMS: $1 => text[] (lowercased citations)
--   SELECT ... WHERE e.md_type='case' AND (
--       (e.md_citation IS NOT NULL AND lower(e.md_citation) = ANY($1))
--       OR EXISTS (
--         SELECT 1
--         FROM jsonb_array_elements_text(COALESCE(e.md_citations,'[]'::jsonb)) AS c(val)
--         WHERE lower(c.val) = ANY($1)
--       )
--   )
--
-- Python (SQLAlchemy):
--   sql = '''
--   SELECT d.id AS doc_id, d.source AS url, e.md_jurisdiction AS jurisdiction, e.md_date AS case_date,
--          e.md_database AS court,
--          COALESCE(e.md_citation,(SELECT min(x) FROM jsonb_array_elements_text(e.md_citations) AS x)) AS citation,
--          (SELECT string_agg(DISTINCT t,'; ') FROM (SELECT e2.md_title AS t FROM embeddings e2 WHERE e2.doc_id=e.doc_id AND e2.md_title IS NOT NULL) s) AS case_name
--   FROM embeddings e
--   JOIN documents d ON d.id=e.doc_id
--   WHERE e.md_type='case'
--     AND ((e.md_citation IS NOT NULL AND lower(e.md_citation) = ANY(:citations))
--          OR EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(e.md_citations,'[]'::jsonb)) AS c(val)
--                     WHERE lower(c.val) = ANY(:citations)))
--   GROUP BY d.id, d.source, e.md_jurisdiction, e.md_date, e.md_database, e.md_citation
--   ORDER BY e.md_date DESC
--   '''
--   rows = conn.execute(text(sql), {"citations": citations_list}).fetchall()
-- =====================================================================
WITH params AS (
  SELECT $1::text[] AS citations
),
matched AS (
  SELECT DISTINCT e.doc_id
  FROM embeddings e, params p
  WHERE e.md_type = 'case'
    AND (
      (e.md_citation IS NOT NULL AND lower(e.md_citation) = ANY(p.citations))
      OR EXISTS (
        SELECT 1
        FROM jsonb_array_elements_text(COALESCE(e.md_citations, '[]'::jsonb)) AS c(val)
        WHERE lower(c.val) = ANY(p.citations)
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
        (SELECT min(x) FROM jsonb_array_elements_text(COALESCE(e2.md_citations, '[]'::jsonb)) AS x(x))
      )
    ) AS citation,
    (
      SELECT string_agg(DISTINCT t, '; ')
      FROM (
        SELECT e3.md_title AS t
        FROM embeddings e3
        WHERE e3.doc_id = e2.doc_id AND e3.md_title IS NOT NULL
      ) s
    ) AS case_name
  FROM embeddings e2
  JOIN matched m ON m.doc_id = e2.doc_id
  GROUP BY e2.doc_id
)
SELECT
  n.doc_id,
  d.source AS url,
  n.jurisdiction AS jurisdiction,
  n.case_date AS case_date,
  n.court AS court,
  n.citation AS citation,
  n.case_name AS case_name
FROM names n
JOIN documents d ON d.id = n.doc_id
ORDER BY n.case_date DESC;


-- =====================================================================
-- 2) Cases by Name (Trigram)
-- What it does:
--   Approximate match (trigram) against case/party name on md_title_lc.
--   Optional filters: jurisdiction/year/court(database). Ranks by similarity.
-- Why it’s fast (indexes used):
--   - Trigram GIN on md_title_lc (pg_trgm)
--   - BTree filters on md_jurisdiction/md_year/md_database
--   - md_type = 'case' narrows scope by cohort
--
-- Driver-agnostic (positional):
--   PARAMS: $1 => text (name), $2 => text (jurisdiction), $3 => int (year), $4 => text (court)
--   ... WHERE e.md_type='case' AND e.md_title_lc % LOWER($1)
--       AND ($2 IS NULL OR e.md_jurisdiction = $2)
--       AND ($3 IS NULL OR e.md_year = $3)
--       AND ($4 IS NULL OR e.md_database = $4)
--
-- Python (SQLAlchemy):
--   sql = '''
--   WITH params AS (SELECT LOWER(:q::text) AS q, :jurisdiction::text AS jurisdiction, :year::int AS year, :court::text AS court)
--   SELECT d.id AS doc_id, d.source AS url, e.md_jurisdiction AS jurisdiction, e.md_date AS case_date, e.md_database AS court,
--          (SELECT string_agg(DISTINCT t,'; ') FROM (SELECT e2.md_title AS t FROM embeddings e2 WHERE e2.doc_id=e.doc_id AND e2.md_title IS NOT NULL) s) AS case_name,
--          MAX(similarity(e.md_title_lc, p.q)) AS name_similarity
--   FROM embeddings e JOIN documents d ON d.id=e.doc_id CROSS JOIN params p
--   WHERE e.md_type='case' AND (e.md_title_lc % p.q)
--     AND (p.jurisdiction IS NULL OR e.md_jurisdiction=p.jurisdiction)
--     AND (p.year IS NULL OR e.md_year=p.year)
--     AND (p.court IS NULL OR e.md_database=p.court)
--   GROUP BY d.id,d.source,e.md_jurisdiction,e.md_date,e.md_database
--   ORDER BY name_similarity DESC,e.md_date DESC
--   '''
--   rows = conn.execute(text(sql), {"q": name, "jurisdiction": jur, "year": year, "court": court}).fetchall()
-- =====================================================================
WITH params AS (
  SELECT LOWER($1::text) AS q,
         $2::text AS jurisdiction,
         $3::int  AS year,
         $4::text AS court,
         COALESCE($5::int, 1000) AS shortlist
),
seed AS (
  SELECT
    e.doc_id,
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
ORDER BY name_similarity DESC, case_date DESC;


-- =====================================================================
-- 3) Cases by Name (Levenshtein)
-- What it does:
--   Uses Levenshtein edit distance against md_title_lc with optional
--   jurisdiction/year/court filters. Slower, but precise refinement.
-- Why it’s fast(er) than naive:
--   - Still benefits from equality filters on md_* btree columns
--   - Note: Levenshtein is CPU-intensive; use low max_dist or combine
--     with trigram prefilter upstream for best performance
--
-- Driver-agnostic (positional):
--   PARAMS: $1 => text (name), $2 => int (max_dist), $3 => text (jurisdiction), $4 => int (year), $5 => text (court)
--   ... WHERE e.md_type='case' AND (levenshtein(e.md_title_lc, LOWER($1)) < $2 OR e.md_title_lc ILIKE LOWER($1))
--
-- Python (SQLAlchemy):
--   sql = '''
--   WITH params AS (SELECT LOWER(:q::text) AS q, :maxd::int AS max_dist, :jurisdiction::text AS jurisdiction, :year::int AS year, :court::text AS court)
--   SELECT ... MIN(levenshtein(e.md_title_lc, p.q)) AS distance
--   ...
--   '''
--   rows = conn.execute(text(sql), {"q": name, "maxd": max_dist, "jurisdiction": jur, "year": year, "court": court}).fetchall()
-- =====================================================================
WITH params AS (
  SELECT LOWER($1::text) AS q,
         $2::int AS max_dist,
         $3::text AS jurisdiction,
         $4::int AS year,
         $5::text AS court
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
ORDER BY MIN(levenshtein(e.md_title_lc, p.q)) ASC, e.md_date DESC;


-- =====================================================================
-- 4) Legislation by Title (Trigram)
-- What it does:
--   Finds legislation by approximate title using trigram on md_title_lc,
--   with optional jurisdiction/year/database filters.
-- Why it’s fast (indexes used):
--   - md_type = 'legislation' (btree)
--   - Trigram GIN on md_title_lc
--   - BTree on md_jurisdiction/md_year/md_database
--
-- Driver-agnostic (positional):
--   PARAMS: $1 => text (title), $2 => text (jurisdiction), $3 => int (year), $4 => text (database), $5 => int (limit)
--   ... WHERE e.md_type='legislation' AND e.md_title_lc % LOWER($1)
--
-- Python (SQLAlchemy):
--   sql = '''
--   WITH params AS (SELECT LOWER(:q::text) AS q, :jurisdiction::text AS jurisdiction, :year::int AS year, :database::text AS database, :lim::int AS lim)
--   SELECT ... similarity(e.md_title_lc, p.q) AS score
--   ...
--   '''
--   rows = conn.execute(text(sql), {"q": title, "jurisdiction": jur, "year": year, "database": db, "lim": limit}).fetchall()
-- =====================================================================
WITH params AS (
  SELECT LOWER($1::text) AS q,
         $2::text AS jurisdiction,
         $3::int  AS year,
         $4::text AS database,
         $5::int  AS lim
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
LIMIT (SELECT lim FROM params);


-- =====================================================================
-- 5) Title Search by Types (Trigram)
-- What it does:
--   Title search across a provided list of types (e.g., treaty, journal),
--   using trigram on md_title_lc and returning a similarity score.
-- Why it’s fast (indexes used):
--   - md_type IN (...) (btree on md_type)
--   - Trigram GIN on md_title_lc
--
-- Driver-agnostic (positional):
--   PARAMS: $1 => text (title), $2 => text[] (types), $3 => int (limit)
--   SELECT ... WHERE e.md_type = ANY($2) AND e.md_title_lc % LOWER($1)
--
-- Python (SQLAlchemy):
--   sql = '''
--   SELECT d.id AS doc_id, d.source AS url, e.md_type AS type, e.md_title AS title, e.md_author AS author,
--          e.md_date AS date, similarity(e.md_title_lc, LOWER(:q::text)) AS score
--   FROM embeddings e JOIN documents d ON d.id=e.doc_id
--   WHERE e.md_type = ANY(:types) AND (e.md_title_lc % LOWER(:q::text))
--   GROUP BY d.id,d.source,e.md_type,e.md_title,e.md_author,e.md_date
--   ORDER BY score DESC,e.md_date DESC
--   LIMIT :lim
--   '''
--   rows = conn.execute(text(sql), {"q": title, "types": ["treaty","journal"], "lim": 20}).fetchall()
-- =====================================================================
WITH params AS (
  SELECT LOWER($1::text) AS q, $2::text[] AS types, $3::int AS lim
),
scored AS (
  SELECT
    e.doc_id,
    d.source AS url,
    e.md_type AS type,
    e.md_title AS title,
    e.md_author AS author,
    e.md_date AS date,
    similarity(e.md_title_lc, p.q) AS score
  FROM embeddings e
  JOIN documents d ON d.id = e.doc_id
  CROSS JOIN params p
  WHERE e.md_type = ANY(p.types)
    AND (e.md_title_lc % p.q)
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY score DESC, date DESC) AS rn
  FROM scored
)
SELECT doc_id, url, type, title, author, date, score
FROM ranked
WHERE rn = 1
ORDER BY score DESC, date DESC
LIMIT (SELECT lim FROM params);


-- =====================================================================
-- 6) ANN Vector Search with Filters (+ Doc Grouping Pattern)
-- What it does:
--   Performs cosine ANN over embeddings.vector with metadata equality/range
--   filters, optional country membership, and optional trigram approx filters
--   on author/title/source after ANN shortlist. Results include top-k chunks
--   per the ANN ordering and are doc-joined for source URL.
-- Why it’s fast (indexes used):
--   - Vector ANN index (IVFFLAT or HNSW) with vector_cosine_ops and <=> operator
--     • If cohort filter present (md_type), planner can pick partial vector index
--     • Otherwise planner can use a global vector index
--   - BTree on md_type/md_database/md_jurisdiction/md_date/md_year
--   - GIN on md_countries for membership test
--   - Trigram GIN on (lower(author/title)) post-ANN refinement (low N)
-- Tuning:
--   - Set ivfflat.probes or hnsw.ef_search per session for recall/latency trade-off
--   - Keep candidate set small with cohort/date filters for best p95
--
-- Driver-agnostic (positional):
--   PARAMS: $1 => vector, $2 => int (top_k), $3 => text (type), $4 => text (database), $5 => text (jurisdiction)
--           $6 => date (date_from), $7 => date (date_to), $8 => text (country),
--           $9 => text (author_approx), $10 => text (title_approx), $11 => text (source_approx)
--   -- Set ivfflat/hnsw session GUCs at runtime (not shown in SQL).
--   ORDER BY e.vector <=> $1::vector LIMIT (SELECT top_k FROM params)
--
-- Python (SQLAlchemy):
--   qv = '[' + ','.join(f'{float(v):.6f}' for v in query_vec) + ']'
--   sql = '''
--   WITH params AS (SELECT :top_k::int AS top_k, :type::text AS type, :database::text AS database, :jurisdiction::text AS jurisdiction,
--                          :date_from::date AS date_from, :date_to::date AS date_to, :country::text AS country,
--                          LOWER(:author_approx::text) AS author_approx, LOWER(:title_approx::text) AS title_approx, LOWER(:source_approx::text) AS source_approx),
--        ann AS (
--          SELECT e.doc_id, e.chunk_index, e.vector <=> (:qv)::vector AS distance,
--                 e.md_type, e.md_database, e.md_jurisdiction, e.md_date, e.md_year, e.md_title, e.md_author, e.md_countries
--          FROM embeddings e, params p
--          WHERE (p.type IS NULL OR e.md_type=p.type)
--            AND (p.database IS NULL OR e.md_database=p.database)
--            AND (p.jurisdiction IS NULL OR e.md_jurisdiction=p.jurisdiction)
--            AND (p.date_from IS NULL OR e.md_date>=p.date_from)
--            AND (p.date_to IS NULL OR e.md_date<=p.date_to)
--            AND (p.country IS NULL OR EXISTS (
--                 SELECT 1 FROM jsonb_array_elements_text(COALESCE(e.md_countries,'[]'::jsonb)) AS c(val)
--                 WHERE LOWER(c.val)=LOWER(p.country)))
--          ORDER BY e.vector <=> (:qv)::vector
--          LIMIT (SELECT top_k FROM params)
--        )
--   SELECT a.doc_id, a.chunk_index, a.distance, d.source AS url, a.md_type, a.md_database AS court, a.md_jurisdiction AS jurisdiction,
--          a.md_date AS date, a.md_year AS year, a.md_title AS title, a.md_author AS author
--   FROM ann a JOIN documents d ON d.id=a.doc_id JOIN params p ON TRUE
--   WHERE (p.author_approx IS NULL OR (lower(coalesce(a.md_author,'')) % p.author_approx))
--     AND (p.title_approx IS NULL OR (lower(coalesce(a.md_title,'')) % p.title_approx))
--     AND (p.source_approx IS NULL OR (lower(d.source) % p.source_approx))
--   ORDER BY a.distance ASC
--   LIMIT (SELECT top_k FROM params)
--   '''
--   params = {"qv": qv, "top_k": top_k, "type": t, "database": db, "jurisdiction": jur, "date_from": df, "date_to": dt,
--             "country": country, "author_approx": author, "title_approx": title_approx, "source_approx": source_approx}
--   rows = conn.execute(text(sql), params).fetchall()
-- =====================================================================
WITH params AS (
  SELECT
    $2::int AS top_k,
    $3::text AS type,
    $4::text AS database,
    $5::text AS jurisdiction,
    $6::date AS date_from,
    $7::date AS date_to,
    $8::text AS country,
    LOWER($9::text)  AS author_approx,
    LOWER($10::text) AS title_approx,
    LOWER($11::text) AS source_approx
),
ann AS (
  SELECT
    e.doc_id,
    e.chunk_index,
    e.vector <=> $1::vector AS distance,
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
  ORDER BY e.vector <=> $1::vector
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
LIMIT (SELECT top_k FROM params);


-- =====================================================================
-- 7) Approximate Title Search with Doc-level Grouping
-- What it does:
--   Computes trigram similarity on md_title_lc with optional filters,
--   then picks the best (highest-similarity) title per doc_id using ROW_NUMBER.
-- Why it’s fast (indexes used):
--   - Trigram GIN on md_title_lc
--   - BTree on md_type/md_jurisdiction/md_database/md_year
--   - Doc-level grouping reduces duplicates for display
--
-- Driver-agnostic (positional):
--   PARAMS: $1 => text (title), $2 => text (type), $3 => text (jurisdiction), $4 => text (database), $5 => int (year), $6 => int (limit)
--
-- Python (SQLAlchemy):
--   sql = '''
--   WITH params AS (SELECT LOWER(:q::text) AS q, :type::text AS type, :jurisdiction::text AS jurisdiction, :database::text AS database, :year::int AS year, :lim::int AS lim),
--        scored AS (
--          SELECT e.doc_id,d.source AS url,e.md_type,e.md_jurisdiction,e.md_database,e.md_year,e.md_date,e.md_title,
--                 similarity(e.md_title_lc, p.q) AS score
--          FROM embeddings e JOIN documents d ON d.id=e.doc_id CROSS JOIN params p
--          WHERE (p.type IS NULL OR e.md_type=p.type)
--            AND (p.jurisdiction IS NULL OR e.md_jurisdiction=p.jurisdiction)
--            AND (p.database IS NULL OR e.md_database=p.database)
--            AND (p.year IS NULL OR e.md_year=p.year)
--            AND e.md_title_lc % p.q
--        ),
--        ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY score DESC) AS rn FROM scored)
--   SELECT doc_id,url,md_type,md_jurisdiction,md_database AS court,md_year,md_date,md_title AS best_title,score
--   FROM ranked WHERE rn=1 ORDER BY score DESC, md_date DESC LIMIT (SELECT lim FROM params)
--   '''
--   rows = conn.execute(text(sql), {"q": title, "type": t, "jurisdiction": jur, "database": db, "year": year, "lim": limit}).fetchall()
-- =====================================================================
WITH params AS (
  SELECT LOWER($1::text) AS q, $2::text AS type, $3::text AS jurisdiction, $4::text AS database, $5::int AS year, $6::int AS lim
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
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY score DESC) AS rn
  FROM scored
)
SELECT doc_id, url, md_type, md_jurisdiction, md_database AS court, md_year, md_date, md_title AS best_title, score
FROM ranked
WHERE rn = 1
ORDER BY score DESC, md_date DESC
LIMIT (SELECT lim FROM params);


-- =====================================================================
-- 8) Approximate documents.source (Trigram)
-- What it does:
--   Fuzzy matches input against documents.source (lowercased), useful for
--   filtering by site/collection labels (e.g., 'nswcaselaw', 'legislation').
-- Why it’s fast (indexes used):
--   - Trigram GIN on lower(source) for approximate matching
--
-- Driver-agnostic (positional):
--   PARAMS: $1 => text (query), $2 => int (limit)
--
-- Python (SQLAlchemy):
--   sql = '''
--   SELECT id AS doc_id, source AS url, similarity(lower(source), LOWER(:q::text)) AS score
--   FROM documents
--   WHERE lower(source) % LOWER(:q::text)
--   ORDER BY similarity(lower(source), LOWER(:q::text)) DESC
--   LIMIT :lim::int
--   '''
--   rows = conn.execute(text(sql), {"q": query, "lim": limit}).fetchall()
-- =====================================================================
SELECT id AS doc_id, source AS url
FROM documents
WHERE lower(source) % LOWER($1::text)
ORDER BY similarity(lower(source), LOWER($1::text)) DESC
LIMIT $2::int;
