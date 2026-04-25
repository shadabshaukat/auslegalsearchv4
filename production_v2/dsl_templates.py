from __future__ import annotations

from typing import Dict, Any, List, Optional


def _filter_clauses(filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filters = filters or {}
    out: List[Dict[str, Any]] = []
    for key in ["type", "jurisdiction", "subjurisdiction", "database", "court"]:
        val = filters.get(key)
        if val:
            out.append({"term": {key: str(val).lower()}})
    gte = filters.get("date_from")
    lte = filters.get("date_to")
    if gte or lte:
        out.append({"range": {"date": {k: v for k, v in {"gte": gte, "lte": lte}.items() if v}}})
    return out


def keyword(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": ["text^4", "title^2", "text_preview"], "operator": "and"}}],
                "filter": _filter_clauses(filters),
            }
        },
    }


def case_name(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"match_phrase": {"title": {"query": query, "boost": 8}}},
                    {"match_phrase": {"case_name_tokens": {"query": query, "boost": 5}}},
                    {"match": {"title": {"query": query, "operator": "and", "boost": 3}}},
                ],
                "minimum_should_match": 1,
                "filter": _filter_clauses(filters),
            }
        },
    }


def citation(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"term": {"citations": {"value": query, "boost": 12}}},
                    {"match_phrase": {"citation_tokens": {"query": query, "boost": 8}}},
                    {"query_string": {"query": f'"{query}"', "fields": ["title^2", "text"]}},
                ],
                "minimum_should_match": 1,
                "filter": _filter_clauses(filters),
            }
        },
    }


def legislation(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    f = dict(filters or {})
    f.setdefault("type", "legislation")
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": ["title^5", "text^2", "section_refs^3"], "operator": "and"}}],
                "filter": _filter_clauses(f),
            }
        },
    }


def boolean_query(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": query,
                            "fields": ["text^3", "title^2", "text_preview"],
                            "default_operator": "AND",
                        }
                    }
                ],
                "filter": _filter_clauses(filters),
            }
        },
    }


def conceptual(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": ["text^2", "title^1.5", "text_preview"], "operator": "or"}}],
                "filter": _filter_clauses(filters),
            }
        },
    }


def metadata_filter(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": ["text^2", "title^2"]}}],
                "filter": _filter_clauses(filters),
            }
        },
    }


def related(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": ["title^2", "text", "citations"]}}],
                "filter": _filter_clauses(filters),
            }
        },
    }


def generic_template(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return conceptual(query=query, top_k=top_k, filters=filters)


SCENARIO_TO_BUILDER = {
    "keyword": keyword,
    "case_name": case_name,
    "citation": citation,
    "legislation": legislation,
    "boolean": boolean_query,
    "conceptual": conceptual,
    "semantic": generic_template,
    "natural_language": generic_template,
    "related": related,
    "query_segmentation": generic_template,
    "metadata_filter": metadata_filter,
    "precedent": generic_template,
    "prompt_task": generic_template,
    "statutory_analysis": legislation,
    "citation_tracing": citation,
    "argument_based": generic_template,
    "fact_pattern": generic_template,
    "argument_drafting": generic_template,
    "counterargument": generic_template,
    "multi_step": generic_template,
}


SCENARIOS = [
    {"id": "keyword", "label": "1. Keyword Search"},
    {"id": "case_name", "label": "2. Case Name Search"},
    {"id": "citation", "label": "3. Citation Search"},
    {"id": "legislation", "label": "4. Legislation Search"},
    {"id": "boolean", "label": "5. Boolean/Connector Search"},
    {"id": "conceptual", "label": "6. Conceptual Search"},
    {"id": "semantic", "label": "7. Semantic Search"},
    {"id": "natural_language", "label": "8. Natural Language Questions"},
    {"id": "related", "label": "9. Related Materials Search"},
    {"id": "query_segmentation", "label": "10. Query Segmentation"},
    {"id": "metadata_filter", "label": "11. Metadata Filtering"},
    {"id": "precedent", "label": "12. Precedent-Based Search"},
    {"id": "prompt_task", "label": "13. Prompt-Based Tasks"},
    {"id": "statutory_analysis", "label": "14. Statutory Analysis"},
    {"id": "citation_tracing", "label": "15. Citation Tracing"},
    {"id": "argument_based", "label": "16. Argument-Based Retrieval"},
    {"id": "fact_pattern", "label": "17. Fact Pattern Search"},
    {"id": "argument_drafting", "label": "18. Argument Drafting"},
    {"id": "counterargument", "label": "19. Counterargument Generation"},
    {"id": "multi_step", "label": "20. Multi-step Legal Research"},
]
