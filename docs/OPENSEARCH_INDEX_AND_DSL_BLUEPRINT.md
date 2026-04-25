# OpenSearch Blueprint: 3-Index Mappings + DSL Templates (20 Capabilities)

This blueprint is tailored for AustLII’s legal assistant platform using:
- **Lexical retrieval (BM25 / query_string / phrase)**
- **Semantic retrieval (kNN)**
- **Query routing + hybrid fusion (RRF)**

---

## 1) Concrete mappings for 3 indexes

> Use one write alias and one read alias per index family in production (rollover-friendly), e.g. `austlii_authorities-read`, `austlii_authorities-write`.

### 1.1 `austlii_authorities_v1` (document-level authority index)

```json
PUT austlii_authorities_v1
{
  "settings": {
    "number_of_shards": 12,
    "number_of_replicas": 1,
    "analysis": {
      "normalizer": {
        "lc_norm": {
          "type": "custom",
          "char_filter": [],
          "filter": ["lowercase", "asciifolding"]
        }
      }
    }
  },
  "mappings": {
    "dynamic": false,
    "properties": {
      "authority_id": {"type": "keyword"},
      "type": {"type": "keyword"},

      "title": {
        "type": "text",
        "fields": {
          "kw": {"type": "keyword", "normalizer": "lc_norm"}
        }
      },
      "title_normalized": {"type": "keyword", "normalizer": "lc_norm"},

      "citations": {"type": "keyword", "normalizer": "lc_norm"},
      "neutral_citation": {"type": "keyword", "normalizer": "lc_norm"},
      "citation_tokens": {"type": "text"},

      "parties": {"type": "text"},
      "case_name_tokens": {"type": "text"},

      "act_name": {
        "type": "text",
        "fields": {
          "kw": {"type": "keyword", "normalizer": "lc_norm"}
        }
      },
      "section_refs": {"type": "keyword", "normalizer": "lc_norm"},

      "authors": {"type": "keyword", "normalizer": "lc_norm"},
      "countries": {"type": "keyword", "normalizer": "lc_norm"},

      "jurisdiction": {"type": "keyword", "normalizer": "lc_norm"},
      "subjurisdiction": {"type": "keyword", "normalizer": "lc_norm"},
      "database": {"type": "keyword", "normalizer": "lc_norm"},
      "court": {"type": "keyword", "normalizer": "lc_norm"},

      "date": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd HH:mm:ss||dd-MM-yyyy HH:mm:ss||epoch_millis"},
      "year": {"type": "integer"},

      "url": {"type": "keyword"},
      "data_quality": {"type": "keyword"},

      "updated_at": {"type": "date"}
    }
  }
}
```

---

### 1.2 `austlii_chunks_lex_v1` (chunk lexical index)

```json
PUT austlii_chunks_lex_v1
{
  "settings": {
    "number_of_shards": 20,
    "number_of_replicas": 1,
    "analysis": {
      "normalizer": {
        "lc_norm": {
          "type": "custom",
          "char_filter": [],
          "filter": ["lowercase", "asciifolding"]
        }
      }
    }
  },
  "mappings": {
    "dynamic": false,
    "properties": {
      "chunk_id": {"type": "keyword"},
      "authority_id": {"type": "keyword"},
      "chunk_index": {"type": "integer"},

      "text": {"type": "text"},
      "text_preview": {"type": "text"},

      "title": {
        "type": "text",
        "fields": {
          "kw": {"type": "keyword", "normalizer": "lc_norm"}
        }
      },
      "citations": {"type": "keyword", "normalizer": "lc_norm"},
      "citation_tokens": {"type": "text"},
      "case_name_tokens": {"type": "text"},
      "section_refs": {"type": "keyword", "normalizer": "lc_norm"},

      "type": {"type": "keyword", "normalizer": "lc_norm"},
      "jurisdiction": {"type": "keyword", "normalizer": "lc_norm"},
      "subjurisdiction": {"type": "keyword", "normalizer": "lc_norm"},
      "database": {"type": "keyword", "normalizer": "lc_norm"},
      "court": {"type": "keyword", "normalizer": "lc_norm"},
      "date": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd HH:mm:ss||dd-MM-yyyy HH:mm:ss||epoch_millis"},
      "year": {"type": "integer"},
      "url": {"type": "keyword"},

      "chunk_metadata": {"type": "object", "enabled": true}
    }
  }
}
```

---

### 1.3 `austlii_chunks_vec_v1` (chunk vector index)

```json
PUT austlii_chunks_vec_v1
{
  "settings": {
    "index": {
      "knn": true,
      "number_of_shards": 20,
      "number_of_replicas": 1
    }
  },
  "mappings": {
    "dynamic": false,
    "properties": {
      "chunk_id": {"type": "keyword"},
      "authority_id": {"type": "keyword"},
      "chunk_index": {"type": "integer"},

      "vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "engine": "lucene",
          "space_type": "cosinesimil"
        }
      },

      "text_preview": {"type": "text"},

      "type": {"type": "keyword"},
      "jurisdiction": {"type": "keyword"},
      "subjurisdiction": {"type": "keyword"},
      "database": {"type": "keyword"},
      "court": {"type": "keyword"},
      "date": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd HH:mm:ss||dd-MM-yyyy HH:mm:ss||epoch_millis"},
      "year": {"type": "integer"},

      "citations": {"type": "keyword"},
      "section_refs": {"type": "keyword"},
      "url": {"type": "keyword"}
    }
  }
}
```

---

## 2) Retrieval strategy baseline

For most complex tasks:
1. Retrieve top-N lexical candidates from `austlii_chunks_lex_v1`
2. Retrieve top-N semantic candidates from `austlii_chunks_vec_v1`
3. Fuse with RRF (or normalized weighted score)
4. Optional rerank top 100 -> top 20 (cross-encoder)
5. Fetch authority metadata from `austlii_authorities_v1` by `authority_id`

---

## 3) Exact DSL templates for 20 capabilities

> Replace placeholders like `${QUERY}`, `${JURISDICTION}`, `${DATE_FROM}`.

### 1) Keyword Search
```json
{
  "size": 20,
  "query": {
    "multi_match": {
      "query": "${QUERY}",
      "fields": ["text^4", "title^2", "text_preview"],
      "operator": "and"
    }
  }
}
```

### 2) Case Name Search
```json
{
  "size": 20,
  "query": {
    "bool": {
      "should": [
        {"match_phrase": {"title": {"query": "${QUERY}", "boost": 8}}},
        {"match_phrase": {"case_name_tokens": {"query": "${QUERY}", "boost": 5}}},
        {"match": {"title": {"query": "${QUERY}", "operator": "and", "boost": 3}}}
      ],
      "minimum_should_match": 1
    }
  }
}
```

### 3) Citation Search
```json
{
  "size": 20,
  "query": {
    "bool": {
      "should": [
        {"term": {"citations": {"value": "${QUERY}", "boost": 12}}},
        {"term": {"neutral_citation": {"value": "${QUERY}", "boost": 12}}},
        {"match_phrase": {"citation_tokens": {"query": "${QUERY}", "boost": 6}}},
        {"query_string": {"query": "\"${QUERY}\"", "fields": ["title^2", "text"]}}
      ],
      "minimum_should_match": 1
    }
  }
}
```

### 4) Legislation Search (Act + section)
```json
{
  "size": 20,
  "query": {
    "bool": {
      "must": [
        {"term": {"type": "legislation"}},
        {
          "multi_match": {
            "query": "${ACT_QUERY}",
            "fields": ["title^5", "text^2"],
            "operator": "and"
          }
        }
      ],
      "should": [
        {"term": {"section_refs": {"value": "${SECTION_REF}", "boost": 8}}},
        {"query_string": {"query": "${SECTION_REF}", "fields": ["text^2", "text_preview"]}}
      ]
    }
  }
}
```

### 5) Boolean/Connector Search
```json
{
  "size": 20,
  "query": {
    "query_string": {
      "query": "${BOOLEAN_QUERY}",
      "fields": ["text^3", "title^2", "text_preview"],
      "default_operator": "AND"
    }
  }
}
```

### 6) Conceptual Search (hybrid lexical request shown)
```json
{
  "size": 50,
  "query": {
    "multi_match": {
      "query": "${QUERY}",
      "fields": ["text^2", "title^1.5", "text_preview"],
      "operator": "or"
    }
  }
}
```

### 7) Semantic Search (kNN)
```json
{
  "size": 50,
  "query": {
    "knn": {
      "vector": {
        "vector": ${QUERY_VECTOR},
        "k": 50
      }
    }
  }
}
```

### 8) Natural Language Questions (hybrid lexical side)
```json
{
  "size": 50,
  "query": {
    "multi_match": {
      "query": "${NL_QUERY}",
      "fields": ["text^2", "title^2", "text_preview"],
      "type": "best_fields"
    }
  }
}
```

### 9) Related Materials Search (seed authority)
```json
{
  "size": 50,
  "query": {
    "bool": {
      "must_not": [{"term": {"authority_id": "${SEED_AUTHORITY_ID}"}}],
      "filter": [{"term": {"type": "case"}}],
      "should": [
        {"terms": {"citations": ${SEED_CITATIONS}}},
        {"terms": {"section_refs": ${SEED_SECTION_REFS}}}
      ]
    }
  }
}
```

### 10) Query Processing / Segmentation (template per segment)
```json
{
  "size": 20,
  "query": {
    "multi_match": {
      "query": "${SEGMENT}",
      "fields": ["text^3", "title^2", "text_preview"]
    }
  }
}
```

### 11) Metadata Filtering
```json
{
  "size": 30,
  "query": {
    "bool": {
      "must": [
        {"multi_match": {"query": "${QUERY}", "fields": ["text^2", "title^2"]}}
      ],
      "filter": [
        {"term": {"jurisdiction": "${JURISDICTION}"}},
        {"term": {"database": "${DATABASE}"}},
        {"range": {"date": {"gte": "${DATE_FROM}", "lte": "${DATE_TO}"}}}
      ]
    }
  }
}
```

### 12) Precedent-Based Search
```json
{
  "size": 50,
  "query": {
    "bool": {
      "filter": [
        {"term": {"type": "case"}},
        {"term": {"jurisdiction": "${JURISDICTION}"}}
      ],
      "must": {
        "knn": {
          "vector": {
            "vector": ${FACT_PATTERN_VECTOR},
            "k": 50
          }
        }
      }
    }
  }
}
```

### 13) Prompt-Based Tasks (context retrieval)
```json
{
  "size": 25,
  "query": {
    "multi_match": {
      "query": "${TASK_QUERY}",
      "fields": ["text^2", "title^2", "text_preview"]
    }
  }
}
```

### 14) Statutory Analysis
```json
{
  "size": 30,
  "query": {
    "bool": {
      "must": [
        {"term": {"type": "legislation"}},
        {"multi_match": {"query": "${QUERY}", "fields": ["title^4", "text^3", "section_refs^3"]}}
      ]
    }
  }
}
```

### 15) Citation Tracing (base retrieval)
```json
{
  "size": 50,
  "query": {
    "bool": {
      "should": [
        {"term": {"citations": "${SEED_CITATION}"}},
        {"match_phrase": {"citation_tokens": "${SEED_CITATION}"}},
        {"multi_match": {"query": "${SEED_CASE_NAME}", "fields": ["title^3", "text"]}}
      ],
      "minimum_should_match": 1
    }
  },
  "sort": [{"date": "asc"}]
}
```

### 16) Argument-Based Retrieval
```json
{
  "size": 50,
  "query": {
    "bool": {
      "must": [
        {"knn": {"vector": {"vector": ${ARGUMENT_VECTOR}, "k": 50}}}
      ],
      "should": [
        {"multi_match": {"query": "${ARGUMENT_QUERY}", "fields": ["text^2", "title"]}}
      ]
    }
  }
}
```

### 17) Fact Pattern Search
```json
{
  "size": 50,
  "query": {
    "bool": {
      "filter": [{"term": {"type": "case"}}],
      "must": {
        "knn": {
          "vector": {
            "vector": ${FACT_VECTOR},
            "k": 50
          }
        }
      }
    }
  }
}
```

### 18) Argument Drafting (evidence retrieval)
```json
{
  "size": 30,
  "query": {
    "bool": {
      "should": [
        {"multi_match": {"query": "${THESIS}", "fields": ["text^2", "title^2"]}},
        {"knn": {"vector": {"vector": ${THESIS_VECTOR}, "k": 30}}}
      ],
      "minimum_should_match": 1
    }
  }
}
```

### 19) Counterargument Generation (retrieve contrary authorities)
```json
{
  "size": 30,
  "query": {
    "bool": {
      "must": [
        {"multi_match": {"query": "${POSITION_QUERY}", "fields": ["text^2", "title"]}}
      ],
      "should": [
        {"match": {"text": {"query": "however OR distinguished OR not followed", "boost": 1.2}}}
      ]
    }
  }
}
```

### 20) Multi-step Legal Research
```json
{
  "size": 40,
  "query": {
    "bool": {
      "must": [
        {"multi_match": {"query": "${STEP_QUERY}", "fields": ["text^2", "title^2"]}}
      ],
      "filter": [
        {"range": {"date": {"gte": "${DATE_FROM}"}}}
      ]
    }
  }
}
```

---

## 4) Hybrid fusion template (application layer)

- Run lexical DSL and vector DSL in parallel.
- Fuse by **RRF**:

```
score(doc) = Σ retriever in {lex, vec} 1 / (k + rank_retriever(doc))
```

Recommended:
- `k = 60`
- lexical top 200, vector top 200, fused top 100, rerank top 30.

---

## 5) Practical routing rules (first-pass)

1. If regex matches citation forms (`\[\d{4}\]\s+[A-Z]+\s+\d+`, `\d+\s+CLR\s+\d+`) -> Capability 3
2. If query contains `v`/`vs` with proper nouns -> Capability 2
3. If query contains `Act`, `s`, `section` -> Capability 4
4. If query contains boolean operators -> Capability 5
5. Else if query length > 7 tokens or question-like -> Capability 8/7 hybrid
6. Add metadata filters when jurisdiction/date/court entities are detected.

---

## 6) Notes for your current v4 repo

- Keep `austlii_documents` and `austlii_embeddings` during transition.
- Introduce `authority_id` in ingestion pipeline and backfill it.
- Add new indexes + aliases, dual-write, then switch reads via routing layer.
