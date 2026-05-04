from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable


BUCKETS = ("cases", "treaties", "journals", "legislation", "hca")


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def resolve_bucket(source_path: str, chunk_metadata_items: Iterable[Dict[str, Any]]) -> str:
    """
    Resolve target index bucket for a file/chunk set.
    Precedence:
      1) database == HCA -> hca
      2) type mapping case/treaty/journal/legislation
      3) path hint contains '/HCA/'
      4) default -> cases
    """
    for md in chunk_metadata_items:
        db = _norm((md or {}).get("database"))
        if db == "hca":
            return "hca"

    for md in chunk_metadata_items:
        t = _norm((md or {}).get("type"))
        if t == "case":
            return "cases"
        if t == "treaty":
            return "treaties"
        if t in {"journal", "article"}:
            return "journals"
        if t in {"legislation", "act", "regulation"}:
            return "legislation"

    p = str(source_path or "").lower()
    if "/hca/" in p:
        return "hca"

    return "cases"


def bucket_index_name(bucket: str) -> str:
    """
    5 production indexes (one per requested type family).
    Env override supported.
    """
    b = _norm(bucket)
    if b not in BUCKETS:
        b = "cases"
    env_key = f"OPENSEARCH_PROD_INDEX_{b.upper()}"
    default = f"auslegalsearch_{b}"
    return os.environ.get(env_key, default).strip() or default


def infer_filetype_from_path(path: str) -> str:
    p = str(Path(path or "").as_posix()).lower()
    if "/hca/" in p:
        return "hca"
    if "/treat" in p:
        return "treaties"
    if "/journal" in p:
        return "journals"
    if "/legis" in p or "/legislation" in p:
        return "legislation"
    return "cases"
