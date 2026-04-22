"""
Beta dataset ingestion pipeline for AUSLegalSearch v3.

Requirements (kept):
- Do NOT modify existing ingestion/embedding code; provide a parallel path specific to the beta dataset.
- 'Scanning' preview may select one representative file per folder and skip year directories (e.g., 2001, 2002, …).
- ACTUAL CHUNKING/INGESTION: must traverse ALL folders recursively (no skipping by year) and ingest all supported files.
- Perform modern, semantic, token-aware chunking using ingest/semantic_chunker.py.
- Insert data into the existing Postgres+pgvector schema using db.store models.
- Enrich chunk_metadata with path-derived hints for improved retrieval and FTS over JSONB.
- Track ingestion progress via EmbeddingSession and EmbeddingSessionFile without altering existing logic.
- Write command-line logs listing successful and failed files for the session.

Modes:
1) Full ingest (default): recursively find ALL .txt/.html files (no year-dir skipping), chunk, embed, and store.
2) Preview/sample scan (optional): one file per folder, skipping year directories — intended only to preview structure.

How it works:
1) File listing:
   - Full ingest: find_all_supported_files(root_dir) — all files.
   - Preview scan: ingest.beta_scanner.find_sample_files(root_dir) — one per folder, skip year dirs.
2) Parse: reuse ingest.loader.parse_txt / parse_html for base text + metadata block extraction.
3) Chunk: semantic, token-aware chunking (semantic_chunker.chunk_document_semantic).
4) Embed: batch embeddings via embedding.embedder.Embedder.
5) Store: for each chunk, create a Document row (content = chunk text), then an Embedding row with vector and rich chunk_metadata.
6) Logging: per-session success and error logs saved under --log_dir.

Usage (Full Ingest - recommended):
    python -m ingest.beta_ingest \\
      --root "/Users/shadab/Downloads/OracleContent/CoE/CoE-Projects/Austlii/Data_for_Beta_Launch" \\
      --session "beta-$(date +%Y%m%d-%H%M%S)" \\
      --model "nomic-ai/nomic-embed-text-v1.5" \\
      --log_dir "./logs"

Optional Preview (scanning only semantics):
    python -m ingest.beta_ingest --root ".../Data_for_Beta_Launch" --session "beta-sample-..." --sample_per_folder --log_dir "./logs"

Notes:
- FTS over documents.content is managed by triggers in db.store.create_all_tables().
- Metadata FTS is handled by /search/fts over embeddings.chunk_metadata::text.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from tqdm import tqdm

# Reuse existing parser for .txt and .html; it already understands optional metadata header blocks.
from ingest.loader import parse_txt, parse_html
from ingest.beta_scanner import find_sample_files
from ingest.semantic_chunker import chunk_document_semantic, ChunkingConfig, detect_doc_type

# DB models and helpers (no schema changes)
from db.store import (
    start_session, update_session_progress, complete_session, fail_session,
    create_all_tables, add_document, add_embedding,
    get_session_file, upsert_session_file_status
)
from embedding.embedder import Embedder


SUPPORTED_EXTS = {".txt", ".html"}
_YEAR_DIR_RE = re.compile(r"^(19|20)\d{2}$")  # 1900-2099


def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s or "")]  # type: ignore[name-defined]


def find_all_supported_files(root_dir: str) -> List[str]:
    """
    Full-ingest file lister: recursively lists ALL supported files under root_dir.
    Does NOT skip year directories. Deterministic natural ordering.
    """
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = sorted(dirnames, key=_natural_sort_key)
        files = [f for f in filenames if Path(f).suffix.lower() in SUPPORTED_EXTS]
        for f in sorted(files, key=_natural_sort_key):
            out.append(os.path.abspath(os.path.join(dirpath, f)))
    # dedupe + sort
    return sorted(list(dict.fromkeys(out)), key=_natural_sort_key)


def derive_path_metadata(file_path: str, root_dir: str) -> Dict[str, Any]:
    """
    Derive helpful metadata from the folder structure (non-destructive).
    Example:
        /root/au/cases/cth/HCA/2001/CaseA.txt -> {
            "dataset_root": "/root",
            "rel_path": "au/cases/cth/HCA/2001/CaseA.txt",
            "rel_path_no_years": "au/cases/cth/HCA/CaseA.txt",
            "path_parts": ["au","cases","cth","HCA","2001"],
            "path_parts_no_years": ["au","cases","cth","HCA"],
            "jurisdiction_guess": "au",
            "court_guess": "HCA",
            "series_guess": "cases/cth",
            "filename": "CaseA.txt",
            "ext": ".txt"
        }
    """
    root_dir = os.path.abspath(root_dir)
    file_path = os.path.abspath(file_path)
    rel_path = os.path.relpath(file_path, root_dir)
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]

    # For retrieval hints, also provide a "no-years" variant
    parts_no_years = [p for p in parts if not _YEAR_DIR_RE.match(p or "")]
    jurisdiction_guess = parts_no_years[0].lower() if parts_no_years else None

    # Heuristic: court_guess is the last non-year folder name (excluding filename)
    court_guess = None
    if parts_no_years:
        if len(parts_no_years) >= 2:
            # choose last folder component (before filename) as court guess if possible
            last_non_year = parts_no_years[-2] if "." in parts_no_years[-1] else parts_no_years[-1]
            court_guess = last_non_year

    series_guess = None
    if len(parts_no_years) >= 3:
        series_guess = "/".join(parts_no_years[1:3]).lower()

    return {
        "dataset_root": root_dir,
        "rel_path": rel_path,
        "rel_path_no_years": "/".join(parts_no_years),
        "path_parts": parts,
        "path_parts_no_years": parts_no_years,
        "jurisdiction_guess": jurisdiction_guess,
        "court_guess": court_guess,
        "series_guess": series_guess,
        "filename": os.path.basename(file_path),
        "ext": Path(file_path).suffix.lower(),
    }


def parse_file(filepath: str) -> Dict[str, Any]:
    """
    Dispatch to appropriate parser based on extension. Only supports .txt and .html.
    """
    ext = Path(filepath).suffix.lower()
    if ext == ".txt":
        return parse_txt(filepath)
    if ext == ".html":
        return parse_html(filepath)
    return {}


def _batch_insert_chunks(
    chunks: List[Dict[str, Any]],
    vectors,
    source_path: str,
    fmt: str
) -> int:
    """
    Insert a batch of chunks and matching vectors into the database as
    Document and Embedding rows. Returns inserted count.
    """
    if not chunks:
        return 0
    if vectors is None or len(vectors) != len(chunks):
        raise ValueError("Vector batch does not match number of chunks")

    inserted = 0
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        cm = chunk.get("chunk_metadata") or {}
        doc_id = add_document({
            "source": source_path,
            "content": text,
            "format": fmt.strip(".") if fmt.startswith(".") else fmt,
        })
        add_embedding(
            doc_id=doc_id,
            chunk_index=idx,
            vector=vectors[idx],
            chunk_metadata=cm,
        )
        inserted += 1
    return inserted


def _write_logs(log_dir: str, session_name: str, successes: List[str], failures: List[str]) -> Dict[str, str]:
    os.makedirs(log_dir, exist_ok=True)
    succ_path = os.path.join(log_dir, f"{session_name}.success.log")
    fail_path = os.path.join(log_dir, f"{session_name}.error.log")
    with open(succ_path, "w", encoding="utf-8") as f:
        for p in successes:
            f.write(p + "\n")
    with open(fail_path, "w", encoding="utf-8") as f:
        for p in failures:
            f.write(p + "\n")
    return {"success_log": succ_path, "error_log": fail_path}


def ingest_beta_dataset(
    root_dir: str,
    session_name: str,
    embedding_model: Optional[str] = None,
    token_target: int = 512,
    token_overlap: int = 64,
    token_max: int = 640,
    sample_per_folder: bool = False,
    skip_year_dirs_in_sample: bool = True,
    log_dir: str = "./logs"
) -> None:
    """
    End-to-end ingestion for the beta dataset root.

    Full mode (default):
      - Lists ALL supported files recursively (no skipping), then chunks/embeds/stores.

    Preview/sample mode (sample_per_folder=True):
      - Uses find_sample_files() to pick at most one file per folder, optionally skipping year directories.
      - Intended for previewing structure; ingestion still occurs on the sampled set if you run with this flag.

    Writes per-session success and error log files to log_dir.
    """
    create_all_tables()  # ensure extensions, indexes, triggers
    os.makedirs(log_dir, exist_ok=True)

    if sample_per_folder:
        files = find_sample_files(root_dir, skip_year_dirs=skip_year_dirs_in_sample)
    else:
        files = find_all_supported_files(root_dir)

    total_files = len(files)
    sess = start_session(session_name=session_name, directory=os.path.abspath(root_dir),
                         total_files=total_files, total_chunks=None)

    embedder = Embedder(embedding_model) if embedding_model else Embedder()
    cfg = ChunkingConfig(
        target_tokens=int(token_target),
        overlap_tokens=int(token_overlap),
        max_tokens=int(token_max)
    )

    processed_chunks = 0
    successes: List[str] = []
    failures: List[str] = []

    for filepath in tqdm(files, desc="Beta ingest", unit="file"):
        try:
            # Track per-file status row (create if missing; will update on completion)
            esf = get_session_file(session_name, filepath)
            if not esf:
                upsert_session_file_status(session_name, filepath, "pending")

            # Parse
            base_doc = parse_file(filepath)
            if not base_doc or not base_doc.get("text"):
                upsert_session_file_status(session_name, filepath, "error")
                failures.append(filepath)
                continue

            # Enrich metadata
            path_meta = derive_path_metadata(filepath, root_dir)
            base_meta = dict(base_doc.get("chunk_metadata") or {})
            base_meta.update(path_meta)

            # Detect content type for analytics
            detected_type = detect_doc_type(base_meta, base_doc.get("text", ""))
            if detected_type and not base_meta.get("type"):
                base_meta["type"] = detected_type

            # Semantic chunking
            file_chunks = chunk_document_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
            if not file_chunks:
                upsert_session_file_status(session_name, filepath, "complete")
                successes.append(filepath)  # processed but produced zero chunks
                continue

            # Batch embed
            texts = [c["text"] for c in file_chunks]
            vecs = embedder.embed(texts)

            # Insert
            inserted = _batch_insert_chunks(
                chunks=file_chunks,
                vectors=vecs,
                source_path=filepath,
                fmt=base_doc.get("format", Path(filepath).suffix.lower().strip(".")),
            )
            processed_chunks += inserted

            # Update progress
            upsert_session_file_status(session_name, filepath, "complete")
            update_session_progress(session_name, last_file=filepath, last_chunk=inserted - 1, processed_chunks=processed_chunks)
            successes.append(filepath)

        except Exception:
            # Mark session/file error and continue to next file
            try:
                upsert_session_file_status(session_name, filepath, "error")
            except Exception:
                pass
            failures.append(filepath)
            continue

    # Write logs and print summary
    paths = _write_logs(log_dir, session_name, successes, failures)
    print(f"[beta_ingest] Session {session_name} complete. Files OK: {len(successes)}, failed: {len(failures)}")
    print(f"[beta_ingest] Success log: {paths['success_log']}")
    print(f"[beta_ingest] Error log:   {paths['error_log']}")

    # Mark session complete (do not hard-fail on partial errors; logs capture details)
    complete_session(session_name)


def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    import argparse
    ap = argparse.ArgumentParser(description="Ingest beta dataset with semantic chunking + pgvector storage.")
    ap.add_argument("--root", required=True, help="Root directory of the beta dataset")
    ap.add_argument("--session", required=True, help="Embedding session name (unique)")
    ap.add_argument("--model", default=None, help="Embedding model name (optional; defaults to env or module default)")
    ap.add_argument("--target_tokens", type=int, default=512, help="Chunking target tokens (default 512)")
    ap.add_argument("--overlap_tokens", type=int, default=64, help="Chunk overlap tokens (default 64)")
    ap.add_argument("--max_tokens", type=int, default=640, help="Hard max per chunk (default 640)")
    ap.add_argument("--sample_per_folder", action="store_true", help="Preview: ingest only one file per folder (scanning mode); skips year dirs by default")
    ap.add_argument("--no_skip_years_in_sample", action="store_true", help="In sample mode, do not skip year directories")
    ap.add_argument("--log_dir", default="./logs", help="Directory to write success/error logs")
    return vars(ap.parse_args(argv))


if __name__ == "__main__":
    args = _parse_cli_args(sys.argv[1:])
    ingest_beta_dataset(
        root_dir=args["root"],
        session_name=args["session"],
        embedding_model=args.get("model"),
        token_target=args.get("target_tokens") or 512,
        token_overlap=args.get("overlap_tokens") or 64,
        token_max=args.get("max_tokens") or 640,
        sample_per_folder=bool(args.get("sample_per_folder")),
        skip_year_dirs_in_sample=not bool(args.get("no_skip_years_in_sample")),
        log_dir=args.get("log_dir") or "./logs",
    )
