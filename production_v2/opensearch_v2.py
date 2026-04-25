from __future__ import annotations

from typing import Dict, Any

from production_v2.config import settings


def get_client():
    from opensearchpy import OpenSearch  # type: ignore
    from urllib.parse import urlparse

    p = urlparse(settings.os_host)
    host = p.hostname or "localhost"
    port = int(p.port or (443 if p.scheme == "https" else 9200))
    use_ssl = (p.scheme == "https")
    auth = (settings.os_user, settings.os_pass) if settings.os_user else None

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=auth,
        use_ssl=use_ssl,
        verify_certs=settings.os_verify_certs,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=settings.os_timeout,
        max_retries=5,
        retry_on_timeout=True,
    )


def authorities_mapping() -> Dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 12,
            "number_of_replicas": 1,
            "analysis": {
                "normalizer": {
                    "lc_norm": {
                        "type": "custom",
                        "char_filter": [],
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "authority_id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "title": {"type": "text", "fields": {"kw": {"type": "keyword", "normalizer": "lc_norm"}}},
                "title_normalized": {"type": "keyword", "normalizer": "lc_norm"},
                "citations": {"type": "keyword", "normalizer": "lc_norm"},
                "neutral_citation": {"type": "keyword", "normalizer": "lc_norm"},
                "citation_tokens": {"type": "text"},
                "parties": {"type": "text"},
                "case_name_tokens": {"type": "text"},
                "act_name": {"type": "text", "fields": {"kw": {"type": "keyword", "normalizer": "lc_norm"}}},
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
                "updated_at": {"type": "date"},
            },
        },
    }


def chunks_lex_mapping() -> Dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 20,
            "number_of_replicas": 1,
            "analysis": {
                "normalizer": {
                    "lc_norm": {
                        "type": "custom",
                        "char_filter": [],
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "chunk_id": {"type": "keyword"},
                "authority_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "source": {"type": "keyword"},
                "text": {"type": "text"},
                "text_preview": {"type": "text"},
                "title": {"type": "text", "fields": {"kw": {"type": "keyword", "normalizer": "lc_norm"}}},
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
                "chunk_metadata": {"type": "object", "enabled": True},
            },
        },
    }


def chunks_vec_mapping() -> Dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 20,
                "number_of_replicas": 1,
            }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "chunk_id": {"type": "keyword"},
                "authority_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "source": {"type": "keyword"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": int(settings.embed_dim),
                    "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
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
                "url": {"type": "keyword"},
            },
        },
    }


def citation_graph_mapping() -> Dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 8,
            "number_of_replicas": 1,
            "analysis": {
                "normalizer": {
                    "lc_norm": {
                        "type": "custom",
                        "char_filter": [],
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "edge_id": {"type": "keyword"},
                "from_authority_id": {"type": "keyword"},
                "to_authority_id": {"type": "keyword"},
                "from_citation": {"type": "keyword", "normalizer": "lc_norm"},
                "to_citation": {"type": "keyword", "normalizer": "lc_norm"},
                "from_title": {"type": "text", "fields": {"kw": {"type": "keyword", "normalizer": "lc_norm"}}},
                "to_title": {"type": "text", "fields": {"kw": {"type": "keyword", "normalizer": "lc_norm"}}},
                "context": {"type": "text"},
                "source_chunk_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "jurisdiction": {"type": "keyword", "normalizer": "lc_norm"},
                "database": {"type": "keyword", "normalizer": "lc_norm"},
                "date": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd HH:mm:ss||dd-MM-yyyy HH:mm:ss||epoch_millis"},
                "weight": {"type": "float"},
            },
        },
    }


def ensure_indexes() -> Dict[str, str]:
    client = get_client()
    specs = {
        settings.index_authorities: authorities_mapping(),
        settings.index_chunks_lex: chunks_lex_mapping(),
        settings.index_chunks_vec: chunks_vec_mapping(),
        settings.index_citation_graph: citation_graph_mapping(),
    }
    out: Dict[str, str] = {}
    for idx, body in specs.items():
        if client.indices.exists(index=idx):
            out[idx] = "exists"
        else:
            client.indices.create(index=idx, body=body)
            out[idx] = "created"
    return out
