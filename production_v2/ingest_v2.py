from __future__ import annotations

import hashlib
import os
import re
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from opensearchpy.helpers import bulk  # type: ignore

from embedding.embedder import Embedder
from ingest.loader import parse_html, parse_txt
from ingest.semantic_chunker import (
    ChunkingConfig,
    chunk_document_semantic,
    chunk_generic_rcts,
    chunk_legislation_dashed_semantic,
    detect_doc_type,
)
from production_v2.config import settings
from production_v2.opensearch_v2 import ensure_indexes, get_client

SUPPORTED_EXTS = {".txt", ".html"}

_CIT_RE_BRACKET = re.compile(r"\[(19|20)\d{2}\]\s*[A-Z]{2,}\s*\d+", re.IGNORECASE)
_CIT_RE_CLR = re.compile(r"\b\d+\s+CLR\s+\d+\b", re.IGNORECASE)
_CIT_RE_HCA = re.compile(r"\bHCA\s+\d+\b", re.IGNORECASE)


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _natural_sort_key(s: str):
    import re as _re

    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s or "")]


def _parse_file(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return parse_txt(path)
    if ext == ".html":
        return parse_html(path)
    return {}


def _find_all_supported_files(root_dir: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = sorted(dirnames, key=_natural_sort_key)
        fs = [f for f in filenames if Path(f).suffix.lower() in SUPPORTED_EXTS]
        for f in sorted(fs, key=_natural_sort_key):
            out.append(os.path.abspath(os.path.join(dirpath, f)))
    return sorted(list(dict.fromkeys(out)), key=_natural_sort_key)


def _sort_by_size_zigzag(paths: List[str]) -> List[str]:
    def _sz(p: str) -> int:
        try:
            return int(os.path.getsize(p))
        except Exception:
            return 0

    sized = sorted(paths, key=_sz, reverse=True)
    out: List[str] = []
    i, j = 0, len(sized) - 1
    while i <= j:
        out.append(sized[i])
        i += 1
        if i <= j:
            out.append(sized[j])
            j -= 1
    return out


def _partition_by_size(items: List[str], n: int) -> List[List[str]]:
    if n <= 1:
        return [items]

    def _sz(p: str) -> int:
        try:
            return int(os.path.getsize(p))
        except Exception:
            return 0

    sized = sorted(items, key=_sz, reverse=True)
    bins: List[List[str]] = [[] for _ in range(n)]
    bin_sizes = [0] * n
    for p in sized:
        j = min(range(n), key=lambda i: bin_sizes[i])
        bins[j].append(p)
        bin_sizes[j] += _sz(p)
    return [b for b in bins if b]


def _write_partition_manifest(session_tag: str, parts: List[List[str]]) -> None:
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "session": session_tag,
            "total_files": sum(len(p) for p in parts),
            "shards": [{"index": i, "count": len(p)} for i, p in enumerate(parts)],
        }
        out = logs_dir / f"{session_tag}.partition.manifest.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        pass


def _normalize_case_tokens(title: str) -> str:
    t = (title or "").replace(" v ", " versus ").replace(" vs ", " versus ")
    return " ".join(t.split()).strip().lower()


def _derive_path_metadata(file_path: str, root_dir: str) -> Dict[str, Any]:
    root_dir = os.path.abspath(root_dir) if root_dir else ""
    file_path = os.path.abspath(file_path)
    rel_path = os.path.relpath(file_path, root_dir) if root_dir and file_path.startswith(root_dir) else file_path
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    return {
        "dataset_root": root_dir,
        "rel_path": rel_path,
        "path_parts": parts,
        "filename": os.path.basename(file_path),
        "ext": Path(file_path).suffix.lower(),
    }


def _extract_authority(path: str, base_doc: Dict[str, Any], root_dir: str) -> Tuple[str, Dict[str, Any]]:
    md = dict(base_doc.get("chunk_metadata") or {})
    md.update(_derive_path_metadata(path, root_dir))

    title = str(md.get("title") or Path(path).stem)
    citations = md.get("citations") or md.get("citation") or []
    if isinstance(citations, str):
        citations = [citations]
    citations = [str(c).strip() for c in citations if str(c).strip()]
    neutral = citations[0] if citations else ""
    authority_type = str(md.get("type") or detect_doc_type(md, base_doc.get("text", "")) or "unknown").lower()
    url = str(md.get("url") or "")
    jurisdiction = str(md.get("jurisdiction") or "").lower()
    subjurisdiction = str(md.get("subjurisdiction") or "").lower()
    database = str(md.get("database") or "").lower()
    court = str(md.get("court") or database or "").lower()
    year = md.get("year")
    try:
        year = int(year) if year is not None and str(year).strip() != "" else None
    except Exception:
        year = None

    authority_id = _sha1(f"{url}|{title}|{neutral}|{authority_type}")
    doc = {
        "authority_id": authority_id,
        "type": authority_type,
        "title": title,
        "title_normalized": title.lower(),
        "citations": citations,
        "neutral_citation": neutral,
        "citation_tokens": " ".join(citations),
        "parties": title if authority_type == "case" else "",
        "case_name_tokens": _normalize_case_tokens(title),
        "act_name": title if authority_type == "legislation" else "",
        "section_refs": [str(md.get("section_identifier") or "").strip()] if md.get("section_identifier") else [],
        "authors": [str(md.get("author"))] if md.get("author") else [],
        "countries": [str(c).lower() for c in (md.get("countries") or [])] if isinstance(md.get("countries"), list) else [],
        "jurisdiction": jurisdiction,
        "subjurisdiction": subjurisdiction,
        "database": database,
        "court": court,
        "date": md.get("date"),
        "year": year,
        "url": url,
        "data_quality": str(md.get("data_quality") or ""),
    }
    return authority_id, doc


def _fallback_chunk_text(text: str, base_meta: Dict[str, Any], chars_per_chunk: int = 4000, overlap_chars: int = 200) -> List[Dict[str, Any]]:
    if chars_per_chunk <= 0:
        chars_per_chunk = 4000
    overlap_chars = max(0, overlap_chars)
    step = max(1, chars_per_chunk - overlap_chars)
    chunks: List[Dict[str, Any]] = []
    n = len(text or "")
    i = 0
    idx = 0
    while i < n:
        j = min(n, i + chars_per_chunk)
        chunk_text = text[i:j]
        md = dict(base_meta or {})
        md["fallback"] = True
        md["strategy"] = "fallback-naive"
        md["chunk_idx"] = idx
        md["start_char"] = i
        md["end_char"] = j
        chunks.append({"text": chunk_text, "chunk_metadata": md})
        i += step
        idx += 1
    return chunks


def _cpu_prepare_file(payload: Dict[str, Any]) -> Dict[str, Any]:
    filepath = str(payload.get("filepath") or "")
    root_dir = str(payload.get("root_dir") or "")
    include_rcts = bool(payload.get("include_rcts") or False)
    cfg = ChunkingConfig(
        target_tokens=int(payload.get("target_tokens") or 512),
        overlap_tokens=int(payload.get("overlap_tokens") or 64),
        max_tokens=int(payload.get("max_tokens") or 640),
        min_sentence_tokens=int(payload.get("min_sentence_tokens") or 8),
        min_chunk_tokens=int(payload.get("min_chunk_tokens") or 60),
    )

    try:
        t0 = time.time()
        base_doc = _parse_file(filepath)
        parse_ms = int((time.time() - t0) * 1000)
        if not base_doc or not (base_doc.get("text") or "").strip():
            return {"filepath": filepath, "status": "empty", "parse_ms": parse_ms}

        authority_id, authority_doc = _extract_authority(filepath, base_doc, root_dir)

        base_meta = dict(base_doc.get("chunk_metadata") or {})
        base_meta.update(_derive_path_metadata(filepath, root_dir))
        text = base_doc.get("text") or ""

        t1 = time.time()
        chunks = chunk_legislation_dashed_semantic(text, base_meta=base_meta, cfg=cfg)
        strategy = "dashed-semantic" if chunks else ""
        if not chunks:
            chunks = chunk_document_semantic(text, base_meta=base_meta, cfg=cfg)
            strategy = "semantic" if chunks else ""
        if not chunks and include_rcts:
            chunks = chunk_generic_rcts(text, base_meta=base_meta, cfg=cfg)
            strategy = "rcts-generic" if chunks else ""
        if not chunks:
            chunks = _fallback_chunk_text(text, base_meta)
            strategy = "fallback-naive" if chunks else "no-chunks"
        chunk_ms = int((time.time() - t1) * 1000)

        return {
            "filepath": filepath,
            "status": "ok" if chunks else "zero_chunks",
            "authority_id": authority_id,
            "authority_doc": authority_doc,
            "chunks": chunks,
            "parse_ms": parse_ms,
            "chunk_ms": chunk_ms,
            "chunk_strategy": strategy,
        }
    except Exception as e:
        return {"filepath": filepath, "status": "error", "error": str(e)}


def _extract_citation_mentions(text: str) -> List[str]:
    out: List[str] = []
    for pat in (_CIT_RE_BRACKET, _CIT_RE_CLR, _CIT_RE_HCA):
        for m in pat.finditer(text or ""):
            v = " ".join(str(m.group(0)).split())
            if v:
                out.append(v)
    seen = set()
    uniq = []
    for c in out:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            uniq.append(c)
    return uniq


def _embed_shard_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    gpu_id = str(payload.get("gpu_id", "")).strip()
    model = str(payload.get("model") or settings.embed_model)
    items: List[Tuple[int, str]] = payload.get("items") or []

    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["AUSLEGALSEARCH_EMBED_USE_CUDA"] = "1"

    emb = Embedder(model)
    vecs = emb.embed([t for _, t in items])
    out = []
    for i, (idx, _) in enumerate(items):
        v = vecs[i]
        out.append((int(idx), v.tolist() if hasattr(v, "tolist") else list(v)))
    return {"vectors": out}


def _embed_texts(
    texts: List[str],
    embedder: Embedder,
    should_stop: Callable[[], bool],
    embed_batch: int,
    gpu_ids_csv: str,
    multigpu_min_texts: int,
) -> List[Any]:
    if not texts:
        return []

    gpu_ids = [x.strip() for x in str(gpu_ids_csv or "").split(",") if x.strip()]
    if gpu_ids and len(texts) >= int(multigpu_min_texts):
        window_size = max(int(multigpu_min_texts), int(embed_batch) * max(1, len(gpu_ids)) * 8)
        merged_all: Dict[int, Any] = {}
        for ws in range(0, len(texts), window_size):
            if should_stop():
                raise RuntimeError("Ingestion stopped by user request")
            sub = texts[ws : ws + window_size]

            shards: List[List[Tuple[int, str]]] = [[] for _ in gpu_ids]
            loads = [0 for _ in gpu_ids]
            for local_idx, txt in enumerate(sub):
                j = min(range(len(loads)), key=lambda k: loads[k])
                shards[j].append((ws + local_idx, txt))
                loads[j] += max(1, len(txt))

            payloads = [
                {"gpu_id": gpu_ids[i], "model": settings.embed_model, "items": shards[i]}
                for i in range(len(gpu_ids))
                if shards[i]
            ]

            merged: Dict[int, Any] = {}
            with ProcessPoolExecutor(max_workers=len(payloads)) as ex:
                futs = [ex.submit(_embed_shard_worker, p) for p in payloads]
                for fut in futs:
                    part = fut.result()
                    for idx, vec in part.get("vectors") or []:
                        merged[int(idx)] = vec

            missing = [i for i in range(ws, ws + len(sub)) if i not in merged]
            if missing:
                fb_vecs = embedder.embed([texts[i] for i in missing])
                for j, mi in enumerate(missing):
                    v = fb_vecs[j]
                    merged[mi] = v.tolist() if hasattr(v, "tolist") else list(v)

            merged_all.update(merged)

        return [merged_all[i] for i in range(len(texts))]

    out: List[Any] = []
    bs = max(1, int(embed_batch))
    for i in range(0, len(texts), bs):
        if should_stop():
            raise RuntimeError("Ingestion stopped by user request")
        part = texts[i : i + bs]
        vecs = embedder.embed(part)
        for v in vecs:
            out.append(v.tolist() if hasattr(v, "tolist") else list(v))
    return out


def _embed_and_index_file(
    *,
    client,
    authority_id: str,
    authority_doc: Dict[str, Any],
    filepath: str,
    chunks: List[Dict[str, Any]],
    embedder: Embedder,
    should_stop: Callable[[], bool],
    citation_to_authority: Dict[str, Dict[str, str]],
    gpu_ids_override: Optional[str] = None,
    multigpu_min_texts_override: Optional[int] = None,
) -> Tuple[int, int]:
    if not chunks:
        return 0, 0

    stream_chunk_size = int(os.environ.get("V2_INGEST_STREAM_CHUNK_FLUSH_SIZE", "0") or 0)
    if stream_chunk_size <= 0:
        stream_chunk_size = max(800, int(settings.ingest_bulk_chunk_size))

    bulk_chunk_size = max(100, int(settings.ingest_bulk_chunk_size))

    indexed_chunks = 0
    citation_edges = 0

    total = len(chunks)
    for ws in range(0, total, stream_chunk_size):
        if should_stop():
            raise RuntimeError("Ingestion stopped by user request")

        sub = chunks[ws : ws + stream_chunk_size]
        texts = [str(c.get("text") or "") for c in sub]
        vecs = _embed_texts(
            texts=texts,
            embedder=embedder,
            should_stop=should_stop,
            embed_batch=int(settings.ingest_embed_batch),
            gpu_ids_csv=gpu_ids_override if gpu_ids_override is not None else settings.ingest_gpu_ids,
            multigpu_min_texts=(
                int(multigpu_min_texts_override)
                if multigpu_min_texts_override is not None
                else int(settings.ingest_multigpu_min_texts)
            ),
        )

        actions: List[Dict[str, Any]] = []
        own_cits = [str(c).strip().lower() for c in (authority_doc.get("citations") or []) if str(c).strip()]

        for i, c in enumerate(sub):
            if should_stop():
                raise RuntimeError("Ingestion stopped by user request")
            ctext = str(c.get("text") or "")
            if not ctext.strip():
                continue

            global_i = ws + i
            md = dict(c.get("chunk_metadata") or {})
            chunk_id = _sha1(f"{authority_id}:{global_i}:{filepath}")
            title = str(md.get("title") or authority_doc.get("title") or "")
            citations = md.get("citations") or authority_doc.get("citations") or []
            if isinstance(citations, str):
                citations = [citations]
            section_ref = str(md.get("section_identifier") or md.get("section") or "").strip()

            common = {
                "chunk_id": chunk_id,
                "authority_id": authority_id,
                "chunk_index": global_i,
                "source": filepath,
                "title": title,
                "citations": citations,
                "citation_tokens": " ".join([str(x) for x in citations]),
                "case_name_tokens": _normalize_case_tokens(title),
                "section_refs": [section_ref] if section_ref else [],
                "type": str(md.get("type") or authority_doc.get("type") or "").lower(),
                "jurisdiction": str(md.get("jurisdiction") or authority_doc.get("jurisdiction") or "").lower(),
                "subjurisdiction": str(md.get("subjurisdiction") or authority_doc.get("subjurisdiction") or "").lower(),
                "database": str(md.get("database") or authority_doc.get("database") or "").lower(),
                "court": str(md.get("court") or authority_doc.get("court") or "").lower(),
                "date": md.get("date") or authority_doc.get("date"),
                "year": md.get("year") or authority_doc.get("year"),
                "url": str(md.get("url") or authority_doc.get("url") or ""),
            }

            actions.append(
                {
                    "_op_type": "index",
                    "_index": settings.index_chunks_lex,
                    "_id": chunk_id,
                    "_source": {**common, "text": ctext, "text_preview": ctext[:500], "chunk_metadata": md},
                }
            )
            actions.append(
                {
                    "_op_type": "index",
                    "_index": settings.index_chunks_vec,
                    "_id": chunk_id,
                    "_source": {**common, "vector": vecs[i], "text_preview": ctext[:500]},
                }
            )

            mentioned = _extract_citation_mentions(ctext)
            for mc in mentioned:
                if mc.lower() in own_cits:
                    continue
                tgt = citation_to_authority.get(mc.lower(), {})
                edge_id = _sha1(f"{authority_id}:{chunk_id}:{mc}")
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": settings.index_citation_graph,
                        "_id": edge_id,
                        "_source": {
                            "edge_id": edge_id,
                            "from_authority_id": authority_id,
                            "to_authority_id": tgt.get("authority_id", ""),
                            "from_citation": (authority_doc.get("neutral_citation") or ""),
                            "to_citation": mc,
                            "from_title": authority_doc.get("title") or "",
                            "to_title": tgt.get("title", ""),
                            "context": ctext[:1200],
                            "source_chunk_id": chunk_id,
                            "source": filepath,
                            "jurisdiction": authority_doc.get("jurisdiction") or "",
                            "database": authority_doc.get("database") or "",
                            "date": authority_doc.get("date"),
                            "weight": 1.0,
                        },
                    }
                )
                citation_edges += 1

            indexed_chunks += 1

        if actions:
            bulk(client, actions, chunk_size=bulk_chunk_size, request_timeout=settings.os_timeout)

    return indexed_chunks, citation_edges


def _run_shard(
    *,
    shard_id: int,
    gpu_id: Optional[str],
    files: List[str],
    root_dir: str,
    client,
    should_stop: Callable[[], bool],
    progress: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    embedder = Embedder(settings.embed_model)
    citation_to_authority: Dict[str, Dict[str, str]] = {}

    ok_files = 0
    failed_files = 0
    indexed_chunks = 0
    citation_edges = 0
    errors: List[str] = []

    include_rcts = os.environ.get("AUSLEGALSEARCH_USE_RCTS_GENERIC", "0") == "1"

    for fp in files:
        if should_stop():
            raise RuntimeError("Ingestion stopped by user request")

        res = _cpu_prepare_file(
            {
                "filepath": fp,
                "root_dir": root_dir,
                "include_rcts": include_rcts,
                "target_tokens": int(settings.chunk_target_tokens),
                "overlap_tokens": int(settings.chunk_overlap_tokens),
                "max_tokens": int(settings.chunk_max_tokens),
                "min_sentence_tokens": int(settings.chunk_min_sentence_tokens),
                "min_chunk_tokens": int(settings.chunk_min_chunk_tokens),
            }
        )

        status_val = str(res.get("status") or "error")
        if status_val in {"error", "empty"}:
            failed_files += 1
            errors.append(f"{fp}: {res.get('error') or status_val}")
            progress(
                {
                    "phase": "prepare",
                    "shard_id": shard_id,
                    "gpu_id": gpu_id,
                    "ok_files": ok_files,
                    "failed_files": failed_files,
                    "indexed_chunks": indexed_chunks,
                    "citation_edges": citation_edges,
                    "parse_ms": res.get("parse_ms"),
                }
            )
            continue

        authority_id = str(res.get("authority_id") or "")
        authority_doc = dict(res.get("authority_doc") or {})
        chunks = list(res.get("chunks") or [])

        if authority_id:
            client.index(index=settings.index_authorities, id=authority_id, body=authority_doc, refresh=False)
            for cit in authority_doc.get("citations") or []:
                c = str(cit or "").strip().lower()
                if c:
                    citation_to_authority[c] = {
                        "authority_id": authority_id,
                        "title": str(authority_doc.get("title") or ""),
                    }

        try:
            inc_chunks, inc_edges = _embed_and_index_file(
                client=client,
                authority_id=authority_id,
                authority_doc=authority_doc,
                filepath=fp,
                chunks=chunks,
                embedder=embedder,
                should_stop=should_stop,
                citation_to_authority=citation_to_authority,
                gpu_ids_override=(gpu_id or ""),
                multigpu_min_texts_override=1 if gpu_id else int(settings.ingest_multigpu_min_texts),
            )
            indexed_chunks += int(inc_chunks)
            citation_edges += int(inc_edges)
            ok_files += 1
        except Exception as e:
            failed_files += 1
            errors.append(f"{fp}: {e}")

        progress(
            {
                "phase": "index",
                "shard_id": shard_id,
                "gpu_id": gpu_id,
                "ok_files": ok_files,
                "failed_files": failed_files,
                "indexed_chunks": indexed_chunks,
                "citation_edges": citation_edges,
                "parse_ms": res.get("parse_ms"),
                "chunk_ms": res.get("chunk_ms"),
                "chunk_strategy": res.get("chunk_strategy"),
            }
        )

    return {
        "ok_files": ok_files,
        "failed_files": failed_files,
        "indexed_chunks": indexed_chunks,
        "citation_edges": citation_edges,
        "errors": errors,
    }


def run_ingestion(
    root_dir: str,
    limit_files: Optional[int] = None,
    include_html: bool = True,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop_cb: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    ensure_indexes()
    client = get_client()

    root_path = Path(root_dir)
    if not root_path.exists() or not root_path.is_dir():
        raise RuntimeError(
            f"Ingestion root_dir not found inside runtime: '{root_dir}'. "
            "Provide a valid absolute path on this VM (e.g. /home/ubuntu/auslegalsearchv4/sample-data-austlii-all-file-types)."
        )

    files = _find_all_supported_files(root_dir)
    if not include_html:
        files = [f for f in files if Path(f).suffix.lower() == ".txt"]
    if limit_files and limit_files > 0:
        files = files[: int(limit_files)]
    if not files:
        raise RuntimeError(
            f"No supported files (.txt/.html) found under '{root_dir}'. "
            "Check folder path and file extensions."
        )

    files = _sort_by_size_zigzag(files)

    def _progress(payload: Dict[str, Any]) -> None:
        if progress_cb:
            try:
                progress_cb(payload)
            except Exception:
                pass

    def _should_stop() -> bool:
        if should_stop_cb is None:
            return False
        try:
            return bool(should_stop_cb())
        except Exception:
            return False

    _progress({"phase": "scan", "total_files": len(files)})

    embedder = Embedder(settings.embed_model)
    if int(settings.embed_dim) != int(embedder.dimension):
        raise RuntimeError(
            f"Embedding dimension mismatch: V2_EMBED_DIM={settings.embed_dim} but model '{settings.embed_model}' -> {embedder.dimension}."
        )

    ok_files = 0
    failed_files = 0
    indexed_chunks = 0
    citation_edges = 0
    errors: List[str] = []

    dynamic_sharding = str(os.environ.get("V2_INGEST_DYNAMIC_SHARDING", "1")).strip().lower() in {"1", "true", "yes", "on"}
    gpu_ids = [x.strip() for x in str(settings.ingest_gpu_ids or "").split(",") if x.strip()]

    if dynamic_sharding and gpu_ids:
        shard_count = int(os.environ.get("V2_INGEST_SHARDS", "0") or 0)
        if shard_count <= 0:
            shard_count = max(len(gpu_ids) * 4, len(gpu_ids))
        shard_count = min(max(1, shard_count), len(files))

        parts = _partition_by_size(files, shard_count)
        session_tag = f"v2-ingest-{int(time.time())}"
        _write_partition_manifest(session_tag, parts)

        aggregates = {"ok_files": 0, "failed_files": 0, "indexed_chunks": 0, "citation_edges": 0}

        next_shard = 0
        active: Dict[Any, Tuple[int, str]] = {}

        def _launch(executor: ThreadPoolExecutor, gpu_id: str, shard_idx: int) -> None:
            fut = executor.submit(
                _run_shard,
                shard_id=shard_idx,
                gpu_id=gpu_id,
                files=parts[shard_idx],
                root_dir=root_dir,
                client=client,
                should_stop=_should_stop,
                progress=_progress,
            )
            active[fut] = (shard_idx, gpu_id)

        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
            for gpu in gpu_ids:
                if next_shard >= len(parts):
                    break
                _launch(ex, gpu, next_shard)
                next_shard += 1

            while active:
                if _should_stop():
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError("Ingestion stopped by user request")

                done, _ = wait(list(active.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    shard_idx, gpu_id = active.pop(fut)
                    try:
                        sr = fut.result()
                    except Exception as e:
                        failed_files += len(parts[shard_idx])
                        errors.append(f"shard {shard_idx} gpu {gpu_id}: {e}")
                        sr = {"ok_files": 0, "failed_files": len(parts[shard_idx]), "indexed_chunks": 0, "citation_edges": 0, "errors": []}

                    for k in ("ok_files", "failed_files", "indexed_chunks", "citation_edges"):
                        aggregates[k] += int(sr.get(k) or 0)
                    errors.extend(list(sr.get("errors") or []))

                    _progress(
                        {
                            "phase": "index",
                            "files_completed": aggregates["ok_files"] + aggregates["failed_files"],
                            "ok_files": aggregates["ok_files"],
                            "failed_files": aggregates["failed_files"],
                            "total_files": len(files),
                            "indexed_chunks": aggregates["indexed_chunks"],
                            "citation_edges": aggregates["citation_edges"],
                            "completed_shard": shard_idx,
                            "gpu_id": gpu_id,
                        }
                    )

                    if next_shard < len(parts):
                        _launch(ex, gpu_id, next_shard)
                        next_shard += 1

        ok_files = aggregates["ok_files"]
        failed_files = aggregates["failed_files"]
        indexed_chunks = aggregates["indexed_chunks"]
        citation_edges = aggregates["citation_edges"]
    else:
        # Fallback: single-shard path (still robust and supports stop semantics)
        sr = _run_shard(
            shard_id=0,
            gpu_id=(gpu_ids[0] if gpu_ids else None),
            files=files,
            root_dir=root_dir,
            client=client,
            should_stop=_should_stop,
            progress=_progress,
        )
        ok_files = int(sr.get("ok_files") or 0)
        failed_files = int(sr.get("failed_files") or 0)
        indexed_chunks = int(sr.get("indexed_chunks") or 0)
        citation_edges = int(sr.get("citation_edges") or 0)
        errors.extend(list(sr.get("errors") or []))

    return {
        "root_dir": root_dir,
        "total_files": len(files),
        "ok_files": ok_files,
        "failed_files": failed_files,
        "indexed_chunks": indexed_chunks,
        "citation_edges": citation_edges,
        "errors": errors[:200],
        "indexes": {
            "authorities": settings.index_authorities,
            "chunks_lex": settings.index_chunks_lex,
            "chunks_vec": settings.index_chunks_vec,
            "citation_graph": settings.index_citation_graph,
        },
        "runtime": {
            "dynamic_sharding": dynamic_sharding,
            "gpu_ids": settings.ingest_gpu_ids,
            "embed_batch": int(settings.ingest_embed_batch),
            "bulk_chunk_size": int(settings.ingest_bulk_chunk_size),
            "embed_model": settings.embed_model,
            "multigpu_min_texts": int(settings.ingest_multigpu_min_texts),
        },
    }
