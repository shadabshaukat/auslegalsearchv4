from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from opensearchpy.helpers import bulk  # type: ignore

from embedding.embedder import Embedder
from ingest.loader import parse_txt, parse_html
from ingest.semantic_chunker import (
    ChunkingConfig,
    chunk_document_semantic,
    chunk_generic_rcts,
    chunk_legislation_dashed_semantic,
    detect_doc_type,
)
from production_v2.config import settings
from production_v2.opensearch_v2 import ensure_indexes, get_client


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _parse_file(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return parse_txt(path)
    if ext == ".html":
        return parse_html(path)
    return {}


def _find_all_supported_files(root_dir: str) -> List[str]:
    out: List[str] = []
    for p in Path(root_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".html"}:
            out.append(str(p.resolve()))
    return sorted(out)


def _normalize_case_tokens(title: str) -> str:
    t = (title or "").replace(" v ", " versus ").replace(" vs ", " versus ")
    return " ".join(t.split()).strip().lower()


def _extract_authority(path: str, base_doc: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    md = dict(base_doc.get("chunk_metadata") or {})
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


def _chunk_doc(path: str, base_doc: Dict[str, Any], cfg: ChunkingConfig) -> List[Dict[str, Any]]:
    text = base_doc.get("text", "") or ""
    if not text.strip():
        return []
    base_meta = dict(base_doc.get("chunk_metadata") or {})
    chunks = chunk_legislation_dashed_semantic(text, base_meta=base_meta, cfg=cfg)
    if not chunks:
        chunks = chunk_document_semantic(text, base_meta=base_meta, cfg=cfg)
    if not chunks:
        chunks = chunk_generic_rcts(text, base_meta=base_meta, cfg=cfg)
    return chunks


_CIT_RE_BRACKET = re.compile(r"\[(19|20)\d{2}\]\s*[A-Z]{2,}\s*\d+", re.IGNORECASE)
_CIT_RE_CLR = re.compile(r"\b\d+\s+CLR\s+\d+\b", re.IGNORECASE)
_CIT_RE_HCA = re.compile(r"\bHCA\s+\d+\b", re.IGNORECASE)


def _extract_citation_mentions(text: str) -> List[str]:
    out: List[str] = []
    for pat in (_CIT_RE_BRACKET, _CIT_RE_CLR, _CIT_RE_HCA):
        for m in pat.finditer(text or ""):
            v = " ".join(str(m.group(0)).split())
            if v:
                out.append(v)
    # de-dupe preserve order
    seen = set()
    uniq = []
    for c in out:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            uniq.append(c)
    return uniq


def _embed_shard_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process worker for true multi-GPU sharded embedding."""
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


def run_ingestion(
    root_dir: str,
    limit_files: Optional[int] = None,
    include_html: bool = True,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop_cb: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    ensure_indexes()
    client = get_client()
    embedder = Embedder(settings.embed_model)
    if int(settings.embed_dim) != int(embedder.dimension):
        raise RuntimeError(
            f"Embedding dimension mismatch: config V2_EMBED_DIM={settings.embed_dim} "
            f"but model '{settings.embed_model}' produced dimension={embedder.dimension}. "
            "Update V2_EMBED_DIM or model selection before ingest/bootstrap."
        )
    cfg = ChunkingConfig(
        target_tokens=int(settings.chunk_target_tokens),
        overlap_tokens=int(settings.chunk_overlap_tokens),
        max_tokens=int(settings.chunk_max_tokens),
        min_sentence_tokens=int(settings.chunk_min_sentence_tokens),
        min_chunk_tokens=int(settings.chunk_min_chunk_tokens),
    )

    root_path = Path(root_dir)
    if not root_path.exists() or not root_path.is_dir():
        raise RuntimeError(
            f"Ingestion root_dir not found inside runtime: '{root_dir}'. "
            "If running in Docker, use container-visible path (e.g. /app/data) "
            "and ensure host folder is mounted into container."
        )

    files = _find_all_supported_files(root_dir)
    if not include_html:
        files = [f for f in files if Path(f).suffix.lower() == ".txt"]
    if limit_files and limit_files > 0:
        files = files[: int(limit_files)]

    if len(files) == 0:
        raise RuntimeError(
            f"No supported files (.txt/.html) found under '{root_dir}'. "
            "Check folder path, file extensions, and Docker volume mounts."
        )

    def _progress(payload: Dict[str, Any]) -> None:
        if progress_cb is not None:
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

    ok_files = 0
    failed_files = 0
    indexed_chunks = 0
    citation_edges = 0
    errors: List[str] = []

    def _prep_file(fp: str) -> Dict[str, Any]:
        base_doc = _parse_file(fp)
        if not base_doc or not (base_doc.get("text") or "").strip():
            return {"fp": fp, "skip": True}
        authority_id, authority_doc = _extract_authority(fp, base_doc)
        chunks = _chunk_doc(fp, base_doc, cfg)
        return {
            "fp": fp,
            "skip": False,
            "authority_id": authority_id,
            "authority_doc": authority_doc,
            "chunks": chunks,
        }

    prepared: List[Dict[str, Any]] = []
    configured_workers = int(settings.ingest_file_workers)
    workers = max(1, (os.cpu_count() or 4) if configured_workers <= 0 else configured_workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_prep_file, fp): fp for fp in files}
        prep_done = 0
        for fut in as_completed(fut_map):
            if _should_stop():
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError("Ingestion stopped by user request")
            fp = fut_map[fut]
            try:
                rec = fut.result()
                prepared.append(rec)
                prep_done += 1
                _progress({"phase": "prepare", "prepared_files": prep_done, "total_files": len(files)})
            except Exception as e:
                failed_files += 1
                errors.append(f"{fp}: {e}")
                prep_done += 1
                _progress({"phase": "prepare", "prepared_files": prep_done, "total_files": len(files), "failed_files": failed_files})

    citation_to_authority: Dict[str, Dict[str, str]] = {}
    for rec in prepared:
        if rec.get("skip"):
            ok_files += 1
            continue
        authority_doc = rec["authority_doc"]
        for cit in authority_doc.get("citations") or []:
            if cit:
                citation_to_authority[str(cit).lower()] = {
                    "authority_id": rec["authority_id"],
                    "title": str(authority_doc.get("title") or ""),
                }

    def _embed_texts(texts: List[str]) -> List[Any]:
        gpu_ids = [x.strip() for x in str(settings.ingest_gpu_ids or "").split(",") if x.strip()]
        if gpu_ids and len(texts) >= int(settings.ingest_multigpu_min_texts):
            window_size = max(
                int(settings.ingest_multigpu_min_texts),
                int(settings.ingest_embed_batch) * max(1, len(gpu_ids)) * 8,
            )
            merged_all: Dict[int, Any] = {}

            for ws in range(0, len(texts), window_size):
                if _should_stop():
                    raise RuntimeError("Ingestion stopped by user request")

                sub = texts[ws : ws + window_size]
                # Greedy load-balancing by text length to reduce GPU idling.
                shards: List[List[Tuple[int, str]]] = [[] for _ in gpu_ids]
                loads = [0 for _ in gpu_ids]
                for local_idx, txt in enumerate(sub):
                    j = min(range(len(loads)), key=lambda k: loads[k])
                    shards[j].append((ws + local_idx, txt))
                    loads[j] += max(1, len(txt))

                payloads = [
                    {
                        "gpu_id": gpu_ids[i],
                        "model": settings.embed_model,
                        "items": shards[i],
                    }
                    for i in range(len(gpu_ids))
                    if shards[i]
                ]

                merged: Dict[int, Any] = {}
                with ProcessPoolExecutor(max_workers=len(payloads)) as ex:
                    futs = [ex.submit(_embed_shard_worker, p) for p in payloads]
                    for fut in as_completed(futs):
                        part = fut.result()
                        for idx, vec in part.get("vectors") or []:
                            merged[int(idx)] = vec

                missing = [i for i in range(ws, ws + len(sub)) if i not in merged]
                if missing:
                    # Fallback to in-process embed for any missing shards to preserve alignment.
                    fb_vecs = embedder.embed([texts[i] for i in missing])
                    for j, mi in enumerate(missing):
                        v = fb_vecs[j]
                        merged[mi] = v.tolist() if hasattr(v, "tolist") else list(v)

                merged_all.update(merged)

            return [merged_all[i] for i in range(len(texts))]

        bs = max(1, int(settings.ingest_embed_batch))
        out: List[Any] = []
        for i in range(0, len(texts), bs):
            if _should_stop():
                raise RuntimeError("Ingestion stopped by user request")
            part = texts[i : i + bs]
            vec = embedder.embed(part)
            for v in vec:
                out.append(v)
        return out

    bulk_chunk_size = max(100, int(settings.ingest_bulk_chunk_size))

    for rec in prepared:
        if _should_stop():
            raise RuntimeError("Ingestion stopped by user request")
        fp = rec.get("fp", "")
        if rec.get("skip"):
            continue
        try:
            authority_id = rec["authority_id"]
            authority_doc = rec["authority_doc"]
            chunks = rec.get("chunks") or []

            client.index(index=settings.index_authorities, id=authority_id, body=authority_doc, refresh=False)

            if not chunks:
                ok_files += 1
                continue

            texts = [str(c.get("text") or "") for c in chunks]
            vecs = _embed_texts(texts)

            actions_lex: List[Dict[str, Any]] = []
            actions_vec: List[Dict[str, Any]] = []
            actions_cg: List[Dict[str, Any]] = []
            own_cits = [str(c).strip().lower() for c in (authority_doc.get("citations") or []) if str(c).strip()]

            for i, c in enumerate(chunks):
                if _should_stop():
                    raise RuntimeError("Ingestion stopped by user request")
                ctext = str(c.get("text") or "")
                if not ctext.strip():
                    continue
                md = dict(c.get("chunk_metadata") or {})
                chunk_id = _sha1(f"{authority_id}:{i}:{fp}")
                title = str(md.get("title") or authority_doc.get("title") or "")
                citations = md.get("citations") or authority_doc.get("citations") or []
                if isinstance(citations, str):
                    citations = [citations]
                section_ref = str(md.get("section_identifier") or md.get("section") or "").strip()

                common = {
                    "chunk_id": chunk_id,
                    "authority_id": authority_id,
                    "chunk_index": i,
                    "source": fp,
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

                actions_lex.append(
                    {
                        "_op_type": "index",
                        "_index": settings.index_chunks_lex,
                        "_id": chunk_id,
                        "_source": {
                            **common,
                            "text": ctext,
                            "text_preview": ctext[:500],
                            "chunk_metadata": md,
                        },
                    }
                )
                v = vecs[i] if i < len(vecs) else embedder.embed([ctext])[0]
                actions_vec.append(
                    {
                        "_op_type": "index",
                        "_index": settings.index_chunks_vec,
                        "_id": chunk_id,
                        "_source": {
                            **common,
                            "vector": v.tolist() if hasattr(v, "tolist") else list(v),
                            "text_preview": ctext[:500],
                        },
                    }
                )

                mentioned = _extract_citation_mentions(ctext)
                for mc in mentioned:
                    if mc.lower() in own_cits:
                        continue
                    tgt = citation_to_authority.get(mc.lower(), {})
                    edge_id = _sha1(f"{authority_id}:{chunk_id}:{mc}")
                    actions_cg.append(
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
                                "source": fp,
                                "jurisdiction": authority_doc.get("jurisdiction") or "",
                                "database": authority_doc.get("database") or "",
                                "date": authority_doc.get("date"),
                                "weight": 1.0,
                            },
                        }
                    )

            if actions_lex:
                bulk(client, actions_lex, chunk_size=bulk_chunk_size, request_timeout=settings.os_timeout)
            if actions_vec:
                bulk(client, actions_vec, chunk_size=bulk_chunk_size, request_timeout=settings.os_timeout)
            if actions_cg:
                bulk(client, actions_cg, chunk_size=bulk_chunk_size, request_timeout=settings.os_timeout)

            indexed_chunks += len(actions_lex)
            citation_edges += len(actions_cg)
            ok_files += 1
            _progress(
                {
                    "phase": "index",
                    "files_completed": ok_files + failed_files,
                    "ok_files": ok_files,
                    "failed_files": failed_files,
                    "total_files": len(files),
                    "indexed_chunks": indexed_chunks,
                    "citation_edges": citation_edges,
                }
            )
        except Exception as e:
            failed_files += 1
            errors.append(f"{fp}: {e}")
            _progress(
                {
                    "phase": "index",
                    "files_completed": ok_files + failed_files,
                    "ok_files": ok_files,
                    "failed_files": failed_files,
                    "total_files": len(files),
                    "indexed_chunks": indexed_chunks,
                    "citation_edges": citation_edges,
                }
            )

    return {
        "root_dir": root_dir,
        "total_files": len(files),
        "ok_files": ok_files,
        "failed_files": failed_files,
        "indexed_chunks": indexed_chunks,
        "citation_edges": citation_edges,
        "errors": errors[:100],
        "indexes": {
            "authorities": settings.index_authorities,
            "chunks_lex": settings.index_chunks_lex,
            "chunks_vec": settings.index_chunks_vec,
            "citation_graph": settings.index_citation_graph,
        },
        "runtime": {
            "file_workers": workers,
            "embed_batch": int(settings.ingest_embed_batch),
            "bulk_chunk_size": bulk_chunk_size,
            "embed_model": settings.embed_model,
        },
    }
