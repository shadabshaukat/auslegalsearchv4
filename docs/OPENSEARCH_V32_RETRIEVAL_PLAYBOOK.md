# OpenSearch 3.2 Retrieval Playbook for AustLII Use-Cases

This playbook maps the use-cases from `austlii-enhanced-capabilities-20250519-1.pdf` to practical OpenSearch 3.2 retrieval patterns.

## 1) Recommended retrieval architecture (optimal at AustLII scale)

Use a **two-stage retrieval design**:

1. **Stage A — candidate retrieval**
   - lexical candidate set (BM25 / query_string / phrase)
   - semantic candidate set (kNN on embeddings)
2. **Stage B — fusion + rerank**
   - fuse with RRF or weighted normalized score
   - optional cross-encoder rerank for top 100→20

Why:
- lexical is best for citations/case names/statutory identifiers
- semantic is best for NL/fact pattern/argument similarity
- hybrid gives strongest recall+precision under mixed legal queries

---

## 2) Use-case to technique mapping

1. Keyword search → `multi_match` (`operator: and`)
2. Citation search → exact/near exact phrase + keyword fields (`query_string` + boosts)
3. Case name search → phrase-heavy match (`multi_match` type `phrase`)
4. Legislation search → phrase + section token handling (`query_string`, boosts for act/year/section)
5. Boolean/connector search → `query_string` (AND/OR/NOT/NEAR equivalent via phrase/slop/proximity)
6. Conceptual search → hybrid (BM25 + kNN)
7. Semantic search → kNN primary + lexical fallback
8. Natural language questions → hybrid + query rewrite/classification
9. Related materials → kNN on seed case chunk(s) + metadata constraints
10. Query segmentation → split into sub-queries then union+rerank
11. Metadata filtering → bool `filter` clauses on jurisdiction/court/date
12. Precedent-based search → kNN + citation graph/metadata constraints
13-20 (LLM-enriched tasks) → retrieval first (hybrid), then grounded generation with citations

---

## 3) DSL patterns (examples)

### A. Keyword / case / legislation lexical base

```json
{
  "size": 10,
  "query": {
    "multi_match": {
      "query": "Fair Work Act 2009 s 351",
      "fields": ["content^4", "text^3", "text_preview", "chunk_metadata.*", "source^1.2"],
      "type": "best_fields",
      "operator": "and"
    }
  }
}
```

### B. Boolean/connector

```json
{
  "size": 10,
  "query": {
    "query_string": {
      "query": "discrimination AND (age OR gender)",
      "fields": ["content^3", "text^2", "text_preview", "chunk_metadata.*"],
      "default_operator": "AND"
    }
  }
}
```

### C. Metadata filtered retrieval

```json
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "duty of care",
            "fields": ["content^2", "text^2", "chunk_metadata.*"]
          }
        }
      ],
      "filter": [
        {"term": {"chunk_metadata.jurisdiction.keyword": "Cth"}},
        {"term": {"chunk_metadata.court.keyword": "High Court"}}
      ]
    }
  }
}
```

### D. kNN semantic retrieval

```json
{
  "size": 20,
  "query": {
    "knn": {
      "vector": {
        "vector": [0.01, -0.02, 0.13],
        "k": 20
      }
    }
  }
}
```

### E. Hybrid candidate retrieval (practical pattern)

Run BM25 + kNN separately, then fuse in app with RRF/weighted normalization.

---

## 4) Test harness in this repo

Use:

```bash
python -m tools.opensearch_dsl_search_harness \
  --usecases tools/opensearch_search_usecases.json \
  --top-k 10
```

Outputs:
- summary metrics
- per-use-case metrics
- report JSON with generated DSL per case (for audit/replay)

Accuracy proxies tracked:
- `hits_returned`
- `hit_rate`
- `topk_term_match_rate`
- `expected_term_coverage`

---

## 5) Better efficiency options beyond baseline DSL

1. **RRF fusion** for BM25 + kNN (recommended default)
2. **Query classification** first (citation/case/legislation/NL) and route to specialized retriever
3. **Field-aware analyzers** for citations and section references (custom analyzers)
4. **Two-level retrieval**: source case retrieval, then section/chunk drill-down
5. **Post-filter rerank** (cross-encoder) on top 100 candidates
6. **Alias + rollover + index tuning** during ingest windows (already present in this repo)
