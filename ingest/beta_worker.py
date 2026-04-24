"""
Parallel beta ingestion worker for AUSLegalSearch v4.

This worker:
- Processes a provided list of files (via --partition_file) OR discovers all files under --root.
- Parses (.txt/.html), performs semantic, token-aware chunking, embeds (batched), and writes to DB.
- Tracks progress in EmbeddingSession and EmbeddingSessionFile for the given session_name.
- Intended to be launched by a multi-GPU orchestrator, one process per GPU with CUDA_VISIBLE_DEVICES set.
- Writes per-worker log files listing successfully ingested files and failures.
- NEW: Incremental logging (append to logs per file) and batched embedding to avoid memory stalls.

Usage (typically invoked by orchestrator):
    python -m ingest.beta_worker SESSION_NAME \\
        --root "/abs/path/to/Data_for_Beta_Launch" \\
        --partition_file ".beta-gpu-partition-SESSION_NAME-gpu0.txt" \\
        --model "nomic-ai/nomic-embed-text-v1.5" \\
        --target_tokens 512 --overlap_tokens 64 --max_tokens 640 \\
        --log_dir "./logs"
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# SQLAlchemy is required for postgres/oracle paths, but OpenSearch-only runs should
# still be importable even if sqlalchemy is not installed in the environment.
_SQLA_AVAILABLE = True
try:
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError, DBAPIError
except Exception:
    _SQLA_AVAILABLE = False

    def text(x):
        return x

    class OperationalError(Exception):
        pass

    class DBAPIError(Exception):
        pass

import signal
import contextlib
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
import multiprocessing as mp
import glob

from ingest.loader import parse_txt, parse_html
from ingest.semantic_chunker import chunk_document_semantic, ChunkingConfig, detect_doc_type, chunk_legislation_dashed_semantic, chunk_generic_rcts
from embedding.embedder import Embedder

from db.store import (
    create_all_tables,
    start_session, update_session_progress, complete_session, fail_session,
    EmbeddingSessionFile, SessionLocal, Document, Embedding,
    add_document, add_embedding, get_session_file, upsert_session_file_status,
    bulk_upsert_file_chunks_opensearch,
)
from db.connector import DB_URL, engine
from db.opensearch_connector import get_opensearch_client

STORAGE_BACKEND = os.environ.get("AUSLEGALSEARCH_STORAGE_BACKEND", "postgres").strip().lower()

if STORAGE_BACKEND != "opensearch" and not _SQLA_AVAILABLE:
    raise ModuleNotFoundError(
        "sqlalchemy is required when AUSLEGALSEARCH_STORAGE_BACKEND is not 'opensearch'"
    )

# Timeouts (seconds). Tunable via env.
PARSE_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_TIMEOUT_PARSE", "60"))
CHUNK_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_TIMEOUT_CHUNK", "90"))
EMBED_BATCH_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_TIMEOUT_EMBED_BATCH", "180"))
INSERT_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_TIMEOUT_INSERT", "120"))
SELECT_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_TIMEOUT_SELECT", "30"))
DB_MAX_RETRIES = int(os.environ.get("AUSLEGALSEARCH_DB_MAX_RETRIES", "3"))

# CPU parallelism for parse+chunk stage and prefetch buffer
_CORES = os.cpu_count() or 2
# 0 or unset -> auto: min(cores-1, 8), at least 1
CPU_WORKERS = int(os.environ.get("AUSLEGALSEARCH_CPU_WORKERS", "0"))
if CPU_WORKERS <= 0:
    CPU_WORKERS = max(1, min(8, _CORES - 1))
PIPELINE_PREFETCH = int(os.environ.get("AUSLEGALSEARCH_PIPELINE_PREFETCH", "0"))
if PIPELINE_PREFETCH <= 0:
    PIPELINE_PREFETCH = 64

class _Timeout(Exception):
    pass

@contextlib.contextmanager
def _deadline(seconds: int):
    if seconds is None or seconds <= 0:
        yield
        return
    def _handler(signum, frame):
        raise _Timeout(f"operation exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

SUPPORTED_EXTS = {".txt", ".html"}
_YEAR_DIR_RE = re.compile(r"^(19|20)\d{2}$")  # 1900-2099


def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s or "")]


def find_all_supported_files(root_dir: str) -> List[str]:
    """Recursively list ALL supported files under root_dir. No skipping of year directories."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = sorted(dirnames, key=_natural_sort_key)
        files = [f for f in filenames if Path(f).suffix.lower() in SUPPORTED_EXTS]
        for f in sorted(files, key=_natural_sort_key):
            out.append(os.path.abspath(os.path.join(dirpath, f)))
    # dedupe + sort
    return sorted(list(dict.fromkeys(out)), key=_natural_sort_key)


def read_partition_file(fname: str) -> List[str]:
    with open(fname, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def _sort_by_size_desc(paths: List[str]) -> List[str]:
    def _sz(p: str) -> int:
        try:
            return int(os.path.getsize(p))
        except Exception:
            return 0
    return sorted(paths, key=_sz, reverse=True)


def _sort_by_size_zigzag(paths: List[str]) -> List[str]:
    """
    Smooth file-size progression instead of strict largest-first.
    Order alternates large/small/large/small to reduce long head-of-line tails
    while still keeping large files flowing early.
    """
    sized = _sort_by_size_desc(paths)
    out: List[str] = []
    i, j = 0, len(sized) - 1
    while i <= j:
        out.append(sized[i])
        i += 1
        if i <= j:
            out.append(sized[j])
            j -= 1
    return out


def _order_worker_files(paths: List[str]) -> (List[str], str):
    """
    Worker-local file order strategy.
    Env:
      AUSLEGALSEARCH_WORKER_FILE_ORDER =
        - natural|none|off        : keep input order
        - size_desc|largest_first : strict largest-first
        - size_zigzag|smooth      : alternate large/small (default)

    Backward compatibility:
      AUSLEGALSEARCH_SORT_WORKER_FILES=0 forces natural order.
    """
    # Legacy hard disable takes precedence.
    if os.environ.get("AUSLEGALSEARCH_SORT_WORKER_FILES", "1") == "0":
        return paths, "natural(legacy-disable)"

    mode = os.environ.get("AUSLEGALSEARCH_WORKER_FILE_ORDER", "size_zigzag").strip().lower()
    if mode in {"natural", "none", "off"}:
        return paths, "natural"
    if mode in {"size_desc", "largest_first", "largest"}:
        return _sort_by_size_desc(paths), "size_desc"
    # Default and recommended for smoother long runs.
    if mode in {"size_zigzag", "smooth", "balanced", "size_mix"}:
        return _sort_by_size_zigzag(paths), "size_zigzag"
    return _sort_by_size_zigzag(paths), f"size_zigzag(fallback:{mode})"


def _opensearch_pipeline_enabled() -> bool:
    """
    Dedicated OpenSearch pipeline toggle.
    - AUSLEGALSEARCH_OPENSEARCH_PIPELINED=1|0 explicitly enables/disables.
    - If unset, fallback to existing auto behavior based on CPU workers/prefetch.
    """
    v = os.environ.get("AUSLEGALSEARCH_OPENSEARCH_PIPELINED")
    if v is not None:
        return _truthy(v)
    return CPU_WORKERS > 1 or int(os.environ.get("AUSLEGALSEARCH_PIPELINE_PREFETCH", str(PIPELINE_PREFETCH))) > 0


def derive_path_metadata(file_path: str, root_dir: str) -> Dict[str, Any]:
    """
    Derive helpful metadata from the folder structure.
    """
    root_dir = os.path.abspath(root_dir) if root_dir else ""
    file_path = os.path.abspath(file_path)
    rel_path = os.path.relpath(file_path, root_dir) if root_dir and file_path.startswith(root_dir) else file_path
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]

    parts_no_years = [p for p in parts if not _YEAR_DIR_RE.match(p or "")]
    jurisdiction_guess = parts_no_years[0].lower() if parts_no_years else None

    court_guess = None
    if parts_no_years:
        if len(parts_no_years) >= 2:
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
    ext = Path(filepath).suffix.lower()
    if ext == ".txt":
        return parse_txt(filepath)
    if ext == ".html":
        return parse_html(filepath)
    return {}


def _batch_insert_chunks(
    session,
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
        doc = Document(
            source=source_path,
            content=text,
            format=fmt.strip(".") if fmt.startswith(".") else fmt
        )
        session.add(doc)
        session.flush()  # assign id
        emb = Embedding(
            doc_id=doc.id,
            chunk_index=idx,
            vector=vectors[idx],
            chunk_metadata=cm
        )
        session.add(emb)
        inserted += 1
    session.commit()
    return inserted


def _append_log_line(log_dir: str, session_name: str, file_path: str, success: bool) -> None:
    os.makedirs(log_dir, exist_ok=True)
    fname = f"{session_name}.success.log" if success else f"{session_name}.error.log"
    fpath = os.path.join(log_dir, fname)
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")


def _metrics_enabled() -> bool:
    return os.environ.get("AUSLEGALSEARCH_LOG_METRICS", "1") != "0"


def _os_metrics_ndjson_enabled() -> bool:
    if _throughput_mode_enabled():
        return False
    return os.environ.get("OS_METRICS_NDJSON", "0") == "1"


def _append_metrics_ndjson(log_dir: str, session_name: str, record: Dict[str, Any]) -> None:
    if not _os_metrics_ndjson_enabled():
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{session_name}.metrics.ndjson")
        rec = dict(record or {})
        rec.setdefault("session", session_name)
        rec.setdefault("ts", int(time.time() * 1000))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _throughput_mode_enabled() -> bool:
    """High-throughput ingest mode toggle for OpenSearch workers."""
    return _truthy(os.environ.get("AUSLEGALSEARCH_MAX_THROUGHPUT_MODE", "0"))


class _RuntimeGovernor:
    """
    Conservative runtime governor for OpenSearch bulk ingest parameters.
    Dynamically tunes with bounded, gradual changes.
    """
    def __init__(self):
        self.enabled = _truthy(os.environ.get("AUSLEGALSEARCH_GOVERNOR_ENABLE", "1"))
        self.cooldown_windows = max(1, int(os.environ.get("AUSLEGALSEARCH_GOVERNOR_COOLDOWN_WINDOWS", "3")))
        self.slow_index_ms = max(1000, int(os.environ.get("AUSLEGALSEARCH_GOVERNOR_SLOW_INDEX_MS", "45000")))

        env_conc = int(os.environ.get("OPENSEARCH_BULK_CONCURRENCY", "4"))
        env_chunk = int(os.environ.get("OPENSEARCH_BULK_CHUNK_SIZE", "1000"))
        env_window = int(os.environ.get("AUSLEGALSEARCH_OS_STREAM_CHUNK_FLUSH_SIZE", "0"))

        self.min_conc = max(1, int(os.environ.get("AUSLEGALSEARCH_GOVERNOR_MIN_CONCURRENCY", "2")))
        self.max_conc = max(self.min_conc, int(os.environ.get("AUSLEGALSEARCH_GOVERNOR_MAX_CONCURRENCY", str(max(env_conc, 8)))))

        self.min_chunk = max(100, int(os.environ.get("AUSLEGALSEARCH_GOVERNOR_MIN_BULK_CHUNK_SIZE", "500")))
        self.max_chunk = max(self.min_chunk, int(os.environ.get("AUSLEGALSEARCH_GOVERNOR_MAX_BULK_CHUNK_SIZE", str(max(env_chunk, 2000)))))

        self.min_window = max(200, int(os.environ.get("AUSLEGALSEARCH_GOVERNOR_MIN_WINDOW_SIZE", "800")))
        self.max_window = max(self.min_window, int(os.environ.get("AUSLEGALSEARCH_GOVERNOR_MAX_WINDOW_SIZE", "4000")))

        self.cur_conc = max(self.min_conc, min(self.max_conc, env_conc))
        self.cur_chunk = max(self.min_chunk, min(self.max_chunk, env_chunk))
        if env_window > 0:
            self.cur_window = max(self.min_window, min(self.max_window, env_window))
        else:
            self.cur_window = 0  # 0 = auto adaptive resolver

        self._windows_seen = 0
        self._last_adjust_at = -10_000
        self._good_streak = 0
        self._bad_streak = 0

    def _apply_env(self) -> None:
        os.environ["OPENSEARCH_BULK_CONCURRENCY"] = str(self.cur_conc)
        os.environ["OPENSEARCH_BULK_CHUNK_SIZE"] = str(self.cur_chunk)
        if self.cur_window > 0:
            os.environ["AUSLEGALSEARCH_OS_STREAM_CHUNK_FLUSH_SIZE"] = str(self.cur_window)

    def initialize(self, session_name: str) -> None:
        if not self.enabled:
            return
        self._apply_env()
        print(
            f"[beta_worker/opensearch] {session_name}: governor enabled conc={self.cur_conc} "
            f"bulk_chunk={self.cur_chunk} window={'auto' if self.cur_window == 0 else self.cur_window}",
            flush=True,
        )

    def _can_adjust(self) -> bool:
        return (self._windows_seen - self._last_adjust_at) >= self.cooldown_windows

    def _step_down(self, severe: bool = False) -> bool:
        f = 0.70 if severe else 0.85
        new_conc = max(self.min_conc, int(self.cur_conc * f))
        new_chunk = max(self.min_chunk, int(self.cur_chunk * f))
        if self.cur_window <= 0:
            # enter explicit conservative window control
            new_window = max(self.min_window, 2500)
        else:
            new_window = max(self.min_window, int(self.cur_window * f))

        changed = (new_conc != self.cur_conc) or (new_chunk != self.cur_chunk) or (new_window != self.cur_window)
        self.cur_conc, self.cur_chunk, self.cur_window = new_conc, new_chunk, new_window
        return changed

    def _step_up(self) -> bool:
        f = 1.10
        new_conc = min(self.max_conc, max(self.min_conc, int(round(self.cur_conc * f))))
        new_chunk = min(self.max_chunk, max(self.min_chunk, int(round(self.cur_chunk * f))))
        if self.cur_window <= 0:
            new_window = 0
        else:
            new_window = min(self.max_window, max(self.min_window, int(round(self.cur_window * f))))

        changed = (new_conc != self.cur_conc) or (new_chunk != self.cur_chunk) or (new_window != self.cur_window)
        self.cur_conc, self.cur_chunk, self.cur_window = new_conc, new_chunk, new_window
        return changed

    def observe_window(self, *, session_name: str, success: bool, index_ms: int, error_text: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self._windows_seen += 1

        et = (error_text or "").lower()
        severe = any(k in et for k in ["429", "too many requests", "rejected_execution_exception", "timed out", "timeout"])
        bad = (not success) or (index_ms >= self.slow_index_ms)

        if bad:
            self._bad_streak += 1
            self._good_streak = 0
            if self._can_adjust() and (severe or self._bad_streak >= 2):
                if self._step_down(severe=severe):
                    self._apply_env()
                    self._last_adjust_at = self._windows_seen
                    print(
                        f"[beta_worker/opensearch] {session_name}: governor step-down "
                        f"conc={self.cur_conc} bulk_chunk={self.cur_chunk} window={self.cur_window}",
                        flush=True,
                    )
            return

        # Good window
        self._good_streak += 1
        self._bad_streak = 0
        if index_ms <= int(self.slow_index_ms * 0.5) and self._good_streak >= 8 and self._can_adjust():
            if self._step_up():
                self._apply_env()
                self._last_adjust_at = self._windows_seen
                self._good_streak = 0
                print(
                    f"[beta_worker/opensearch] {session_name}: governor step-up "
                    f"conc={self.cur_conc} bulk_chunk={self.cur_chunk} window={self.cur_window}",
                    flush=True,
                )


def _completed_from_success_logs(log_dir: str) -> set:
    completed = set()
    try:
        for p in glob.glob(os.path.join(log_dir or ".", "*.success.log")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for ln in f:
                        s = (ln or "").strip()
                        if not s or s.startswith("#"):
                            continue
                        fp = s.split("\t", 1)[0].strip()
                        if fp:
                            completed.add(fp)
            except Exception:
                continue
    except Exception:
        pass
    return completed


class _OpenSearchIngestState:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.client = None
        self.index = os.environ.get("OPENSEARCH_INGEST_STATE_INDEX", "auslegalsearch_ingest_state")
        if not self.enabled:
            return
        try:
            self.client = get_opensearch_client()
            if not self.client.indices.exists(index=self.index):
                self.client.indices.create(
                    index=self.index,
                    body={
                        "mappings": {
                            "properties": {
                                "type": {"type": "keyword"},
                                "session": {"type": "keyword"},
                                "file": {"type": "keyword"},
                                "status": {"type": "keyword"},
                                "ts": {"type": "date"},
                            }
                        }
                    },
                )
        except Exception as e:
            self.enabled = False
            self.client = None
            print(f"[beta_worker/opensearch] WARN: could not ensure state index '{self.index}': {e}", flush=True)

    def upsert(self, doc_id: str, body: Dict[str, Any]) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            payload = dict(body or {})
            payload.setdefault("ts", int(time.time() * 1000))
            self.client.index(index=self.index, id=doc_id, body=payload, refresh=False)
        except Exception:
            pass


def _append_success_metrics_line(
    log_dir: str,
    session_name: str,
    file_path: str,
    chunks_count: int,
    text_len: int,
    cfg: ChunkingConfig,
    strategy: str,
    detected_type: Optional[str] = None,
    section_count: Optional[int] = None,
    tokens_est_total: Optional[int] = None,
    tokens_est_mean: Optional[int] = None,
    parse_ms: Optional[int] = None,
    chunk_ms: Optional[int] = None,
    embed_ms: Optional[int] = None,
    insert_ms: Optional[int] = None,
) -> None:
    """
    Append a single TSV-style line to the child success log with per-file metrics.
    Keeps logging overhead minimal for performance.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        fpath = os.path.join(log_dir, f"{session_name}.success.log")
        if not _metrics_enabled():
            # Write only the file path (legacy format) and avoid computing metrics
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(str(file_path) + "\n")
            return
        parts = [
            str(file_path),
            f"chunks={int(chunks_count)}",
            f"text_len={int(text_len)}",
            f"strategy={strategy or ''}",
            f"target_tokens={int(cfg.target_tokens)}",
            f"overlap_tokens={int(cfg.overlap_tokens)}",
            f"max_tokens={int(cfg.max_tokens)}",
        ]
        if detected_type:
            parts.append(f"type={detected_type}")
        if section_count is not None:
            parts.append(f"section_count={int(section_count)}")
        if tokens_est_total is not None:
            parts.append(f"tokens_est_total={int(tokens_est_total)}")
        if tokens_est_mean is not None:
            parts.append(f"tokens_est_mean={int(tokens_est_mean)}")
        if parse_ms is not None:
            parts.append(f"parse_ms={int(parse_ms)}")
        if chunk_ms is not None:
            parts.append(f"chunk_ms={int(chunk_ms)}")
        if embed_ms is not None:
            parts.append(f"embed_ms={int(embed_ms)}")
        if insert_ms is not None:
            parts.append(f"insert_ms={int(insert_ms)}")
        line = "\t".join(parts)
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Never fail pipeline due to logging
        pass


def _write_logs(log_dir: str, session_name: str, successes: List[str], failures: List[str]) -> Dict[str, str]:
    """
    Finalize logs. We DO NOT overwrite the child success log to preserve
    the per-file metrics lines emitted during processing. Instead, append a compact
    summary footer to each log. This keeps overhead negligible.
    """
    os.makedirs(log_dir, exist_ok=True)
    succ_path = os.path.join(log_dir, f"{session_name}.success.log")
    fail_path = os.path.join(log_dir, f"{session_name}.error.log")
    failed_paths_path = os.path.join(log_dir, f"{session_name}.failed.paths.txt")
    failed_ndjson_path = os.path.join(log_dir, f"{session_name}.failed.ndjson")

    # Ensure files exist
    try:
        open(succ_path, "a", encoding="utf-8").close()
    except Exception:
        pass
    try:
        open(fail_path, "a", encoding="utf-8").close()
    except Exception:
        pass

    # Deduplicate file path lists (for counts only; we don't rewrite logs)
    successes = list(dict.fromkeys(successes))
    failures = list(dict.fromkeys(failures))

    # Append summary footers (minimal I/O)
    try:
        with open(succ_path, "a", encoding="utf-8") as f:
            f.write(f"# summary files_ok={len(successes)}\n")
    except Exception:
        pass
    try:
        with open(fail_path, "a", encoding="utf-8") as f:
            f.write(f"# summary files_failed={len(failures)}\n")
    except Exception:
        pass

    # Write a de-duplicated failed-file list that can be reused as --partition_file.
    try:
        with open(failed_paths_path, "w", encoding="utf-8") as f:
            for p in failures:
                f.write(p + "\n")
    except Exception:
        pass

    # Ensure failed NDJSON exists even on all-success runs.
    try:
        open(failed_ndjson_path, "a", encoding="utf-8").close()
    except Exception:
        pass

    return {
        "success_log": succ_path,
        "error_log": fail_path,
        "failed_paths_file": failed_paths_path,
        "failed_ndjson": failed_ndjson_path,
    }

def _error_details_enabled() -> bool:
    return os.environ.get("AUSLEGALSEARCH_ERROR_DETAILS", "1") == "1"

def _append_error_detail(
    log_dir: str,
    session_name: str,
    file_path: str,
    stage: str,
    error_type: str,
    message: str,
    duration_ms: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    tb: Optional[str] = None
) -> None:
    if not _error_details_enabled():
        return
    try:
        rec = {
            "session": session_name,
            "file": file_path,
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "duration_ms": duration_ms,
            "meta": meta or {},
            "ts": int(time.time() * 1000),
        }
        if os.environ.get("AUSLEGALSEARCH_ERROR_TRACE", "0") == "1":
            rec["traceback"] = tb or ""
        path = os.path.join(log_dir, f"{session_name}.errors.ndjson")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Never fail pipeline due to logging
        pass


def _append_failed_reingest_entry(
    log_dir: str,
    session_name: str,
    file_path: str,
    stage: str,
    error_type: str,
    message: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Real-time failed-file tracking for targeted re-ingestion.
    Writes:
      - {session}.failed.ndjson   (structured failure events)
      - {session}.failed.paths.txt (one filepath per line; usable as --partition_file)
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        rec = {
            "ts": int(time.time() * 1000),
            "session": session_name,
            "file": file_path,
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "meta": meta or {},
        }
        ndjson_path = os.path.join(log_dir, f"{session_name}.failed.ndjson")
        with open(ndjson_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        paths_path = os.path.join(log_dir, f"{session_name}.failed.paths.txt")
        with open(paths_path, "a", encoding="utf-8") as f:
            f.write(file_path + "\n")
    except Exception:
        # Never fail ingest because of failure tracking I/O.
        pass


def _maybe_print_counts(dbs, session_name: str, label: str) -> None:
    """
    If AUSLEGALSEARCH_DEBUG_COUNTS=1, print current documents/embeddings counts with a label.
    """
    try:
        if os.environ.get("AUSLEGALSEARCH_DEBUG_COUNTS", "0") != "1":
            return
        docs = dbs.execute(text("SELECT count(*) FROM documents")).scalar()
        embs = dbs.execute(text("SELECT count(*) FROM embeddings")).scalar()
        print(f"[beta_worker] {session_name}: DB counts {label}: documents={docs}, embeddings={embs}", flush=True)
    except Exception as e:
        print(f"[beta_worker] {session_name}: DB counts {label}: error: {e}", flush=True)


def _embed_in_batches(embedder: Embedder, texts: List[str], batch_size: int) -> List:
    """
    Embed texts in smaller batches with adaptive backoff to avoid GPU OOM.
    Returns a list-like of vectors aligned with texts.
    """
    if batch_size <= 0:
        batch_size = 64
    all_vecs = []
    i = 0
    cur_bs = int(batch_size)
    n = len(texts)
    while i < n:
        sub = texts[i:i + cur_bs]
        try:
            with _deadline(EMBED_BATCH_TIMEOUT):
                vecs = embedder.embed(sub)  # ndarray [batch, dim]
            # Faster extend path for ndarray/list outputs.
            if hasattr(vecs, "tolist"):
                all_vecs.extend(vecs.tolist())
            else:
                all_vecs.extend(list(vecs))
            i += cur_bs
            # Optional: try to grow batch again if it was reduced before (conservative)
            if cur_bs < batch_size:
                cur_bs = min(batch_size, max(1, cur_bs * 2))
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                # Adaptive backoff on OOM
                next_bs = max(1, cur_bs // 2)
                if next_bs == cur_bs:
                    # Already at 1; re-raise
                    raise
                print(f"[beta_worker] embed OOM/backoff: reducing batch_size {cur_bs} -> {next_bs}", flush=True)
                cur_bs = next_bs
                # do not advance i; retry with smaller batch
                time.sleep(0.5)
            else:
                # Non-OOM error: propagate
                raise
    return all_vecs


def _resolve_os_stream_flush_size(total_chunks: int) -> int:
    """
    Per-file streaming sub-batch size for OpenSearch embed+index stage.
    - AUSLEGALSEARCH_OS_STREAM_CHUNK_FLUSH_SIZE > 0: explicit value
    - throughput mode: default to 1200
    - otherwise: process whole file in one window (legacy behavior)
    """
    try:
        cfg = int(os.environ.get("AUSLEGALSEARCH_OS_STREAM_CHUNK_FLUSH_SIZE", "0"))
    except Exception:
        cfg = 0
    if cfg > 0:
        return max(1, cfg)
    if _throughput_mode_enabled():
        # Adaptive windows for fairness + throughput:
        # - very large files use smaller windows to avoid long head-of-line stalls
        # - medium/normal files use larger windows for better bulk throughput
        tc = int(total_chunks or 0)
        if tc >= 30000:
            return 800
        if tc >= 10000:
            return 1600
        return 4000
    return max(1, int(total_chunks or 1))


def _embed_and_index_file_chunks_opensearch(
    *,
    embedder: Embedder,
    source_path: str,
    fmt: str,
    file_chunks: List[Dict[str, Any]],
    batch_size: int,
    insert_retries: int,
    governor: Optional[_RuntimeGovernor] = None,
    session_name: str = "",
) -> (int, int, int):
    """
    Stream file chunks through embed->index windows to reduce memory spikes and
    head-of-line blocking for very large files.
    Returns: (inserted_total, embed_ms_total, index_ms_total)
    """
    if not file_chunks:
        return 0, 0, 0

    window_size = _resolve_os_stream_flush_size(len(file_chunks))
    inserted_total = 0
    embed_ms_total = 0
    index_ms_total = 0

    total = len(file_chunks)
    total_windows = (total + window_size - 1) // window_size
    for widx, start in enumerate(range(0, total, window_size), start=1):
        sub_chunks = file_chunks[start:start + window_size]
        texts = [c.get("text", "") for c in sub_chunks]

        t2 = time.time()
        with _deadline(EMBED_BATCH_TIMEOUT):
            sub_vecs = _embed_in_batches(embedder, texts, batch_size=batch_size)
        embed_ms_total += int((time.time() - t2) * 1000)

        # IMPORTANT: avoid retry multiplication.
        # bulk_upsert_file_chunks_opensearch already performs internal retries,
        # so do not wrap it in another external retry loop.
        t3 = time.time()
        per_window_deadline = max(INSERT_TIMEOUT, INSERT_TIMEOUT * max(1, insert_retries))
        try:
            with _deadline(per_window_deadline):
                inserted_win = bulk_upsert_file_chunks_opensearch(
                    source_path=source_path,
                    fmt=fmt,
                    chunks=sub_chunks,
                    vectors=sub_vecs,
                    max_retries=max(1, insert_retries),
                    chunk_start_index=start,
                )
            win_index_ms = int((time.time() - t3) * 1000)
            index_ms_total += win_index_ms
            if governor is not None:
                governor.observe_window(session_name=session_name, success=True, index_ms=win_index_ms)
        except Exception as e:
            win_index_ms = int((time.time() - t3) * 1000)
            if governor is not None:
                governor.observe_window(session_name=session_name, success=False, index_ms=win_index_ms, error_text=str(e))
            raise

        inserted_total += inserted_win

        # Progress heartbeat for large files/windows (helps distinguish stall vs work).
        if total_windows > 1 and (widx == 1 or widx == total_windows or widx % 5 == 0):
            print(
                f"[beta_worker/opensearch] embed+index progress: {source_path} "
                f"window={widx}/{total_windows} chunks_done~{min(total, start + len(sub_chunks))}/{total}",
                flush=True,
            )

    return inserted_total, embed_ms_total, index_ms_total


def _db_insert_with_retry(
    session,
    chunks: List[Dict[str, Any]],
    vectors,
    source_path: str,
    fmt: str,
    max_retries: int = DB_MAX_RETRIES
) -> int:
    """
    Insert with per-call deadline and retry on transient DB errors.
    Rolls back and retries with exponential backoff.
    """
    attempt = 0
    while True:
        try:
            with _deadline(INSERT_TIMEOUT):
                return _batch_insert_chunks(
                    session=session,
                    chunks=chunks,
                    vectors=vectors,
                    source_path=source_path,
                    fmt=fmt,
                )
        except (OperationalError, DBAPIError) as e:
            try:
                session.rollback()
            except Exception:
                pass
            if attempt >= max_retries:
                print(f"[beta_worker] DB insert failed after {max_retries} retries: {e}", flush=True)
                raise
            sleep_s = min(8, 2 ** attempt)
            print(f"[beta_worker] DB insert error, retrying in {sleep_s}s (attempt {attempt+1}/{max_retries}): {e}", flush=True)
            time.sleep(sleep_s)
            attempt += 1


def _db_get_esf_with_retry(
    session,
    session_name: str,
    filepath: str,
    select_timeout: int = SELECT_TIMEOUT,
    max_retries: int = DB_MAX_RETRIES
) -> Optional[EmbeddingSessionFile]:
    """
    Fetch EmbeddingSessionFile row with a bounded deadline and retries on transient DB errors.
    """
    attempt = 0
    while True:
        try:
            with _deadline(select_timeout):
                return session.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
        except (OperationalError, DBAPIError, _Timeout) as e:
            try:
                session.rollback()
            except Exception:
                pass
            if attempt >= max_retries:
                print(f"[beta_worker] DB select failed after {max_retries} retries for file: {filepath} :: {e}", flush=True)
                raise
            sleep_s = min(8, 2 ** attempt)
            print(f"[beta_worker] DB select error, retrying in {sleep_s}s (attempt {attempt+1}/{max_retries}): {e}", flush=True)
            time.sleep(sleep_s)
            attempt += 1


def _db_ensure_pending_esf(
    session,
    session_name: str,
    filepath: str,
    max_retries: int = DB_MAX_RETRIES
) -> EmbeddingSessionFile:
    """
    Ensure there is a row for (session_name, filepath). If missing, insert with status='pending'.
    Handles unique constraint races by re-reading the row.
    """
    attempt = 0
    while True:
        esf = _db_get_esf_with_retry(session, session_name, filepath)
        if esf:
            return esf
        try:
            esf = EmbeddingSessionFile(session_name=session_name, filepath=filepath, status="pending")
            session.add(esf)
            session.commit()
            return esf
        except (OperationalError, DBAPIError) as e:
            try:
                session.rollback()
            except Exception:
                pass
            # Unique violation or transient error; re-read on next loop
            if attempt >= max_retries:
                print(f"[beta_worker] DB insert (pending) failed after {max_retries} retries for file: {filepath} :: {e}", flush=True)
                raise
            sleep_s = min(8, 2 ** attempt)
            print(f"[beta_worker] DB insert (pending) error, retrying in {sleep_s}s (attempt {attempt+1}/{max_retries}): {e}", flush=True)
            time.sleep(sleep_s)
            attempt += 1

def _fallback_chunk_text(text: str, base_meta: Dict[str, Any], cfg: ChunkingConfig) -> List[Dict[str, Any]]:
    """
    Fallback chunker that slices text by characters when semantic chunker times out or fails.
    Uses configurable window/overlap to create chunk dicts compatible with downstream pipeline.
    """
    chars_per_chunk = int(os.environ.get("AUSLEGALSEARCH_FALLBACK_CHARS_PER_CHUNK", "4000"))
    overlap_chars = int(os.environ.get("AUSLEGALSEARCH_FALLBACK_OVERLAP_CHARS", "200"))
    if chars_per_chunk <= 0:
        chars_per_chunk = 4000
    if overlap_chars < 0:
        overlap_chars = 0
    step = max(1, chars_per_chunk - overlap_chars)
    chunks: List[Dict[str, Any]] = []
    n = len(text or "")
    i = 0
    idx = 0
    while i < n:
        j = min(n, i + chars_per_chunk)
        slice_text = text[i:j]
        md = dict(base_meta or {})
        md.setdefault("strategy", "fallback-naive")
        md["fallback"] = True
        md["start_char"] = i
        md["end_char"] = j
        chunks.append({"text": slice_text, "chunk_metadata": md})
        idx += 1
        i += step
    return chunks


def _cpu_prepare_file(
    filepath: str,
    root_dir: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
) -> Dict[str, Any]:
    """
    CPU-bound task: parse file and produce chunk list + per-file metrics.
    Runs in a separate process. Never touches DB or GPU.
    Returns:
      {
        "filepath": str,
        "status": "ok"|"empty"|"zero_chunks"|"fallback_ok"|"error",
        "error": str (optional),
        "chunks": List[{"text":..., "chunk_metadata": {...}}] (optional),
        "text_len": int,
        "chunk_count": int,
        "detected_type": Optional[str],
        "chunk_strategy": str ("dashed-semantic"|"semantic"|"rcts-generic"|"fallback-naive"|"no-chunks"),
        "section_count": Optional[int],
        "tokens_est_total": Optional[int],
        "tokens_est_mean": Optional[int],
        "parse_ms": Optional[int],
        "chunk_ms": Optional[int],
      }
    """
    try:
        t0 = time.time()
        # Parse
        with _deadline(PARSE_TIMEOUT):
            base_doc = parse_file(filepath)
        text_len = len(base_doc.get("text", "") or "")
        parse_ms = int((time.time() - t0) * 1000)
        file_fmt = (base_doc.get("format") or Path(filepath).suffix.lower().strip(".") or "txt")
        if not base_doc or not base_doc.get("text"):
            return {
                "filepath": filepath,
                "status": "empty",
                "format": file_fmt,
                "text_len": 0,
                "chunk_count": 0,
                "chunk_strategy": "no-chunks",
                "parse_ms": parse_ms,
                "chunk_ms": None,
            }

        # Build base metadata
        path_meta = derive_path_metadata(filepath, root_dir or "")
        base_meta = dict(base_doc.get("chunk_metadata") or {})
        base_meta.update(path_meta)

        detected_type = detect_doc_type(base_meta, base_doc.get("text", ""))

        # Chunk
        cfg = ChunkingConfig(
            target_tokens=int(token_target),
            overlap_tokens=int(token_overlap),
            max_tokens=int(token_max),
        )
        chunk_strategy = None
        t1 = time.time()
        with _deadline(CHUNK_TIMEOUT):
            # Try dashed first
            file_chunks = chunk_legislation_dashed_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
            if file_chunks:
                chunk_strategy = "dashed-semantic"
            else:
                # Then semantic
                file_chunks = chunk_document_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
                if file_chunks:
                    chunk_strategy = "semantic"
            # Optional RCTS fallback
            if (not file_chunks) and os.environ.get("AUSLEGALSEARCH_USE_RCTS_GENERIC", "0") == "1":
                rcts_chunks = chunk_generic_rcts(base_doc["text"], base_meta=base_meta, cfg=cfg)
                if rcts_chunks:
                    file_chunks = rcts_chunks
                    chunk_strategy = "rcts-generic"
        chunk_ms = int((time.time() - t1) * 1000)

        if not file_chunks:
            return {
                "filepath": filepath,
                "status": "zero_chunks",
                "format": file_fmt,
                "text_len": text_len,
                "chunk_count": 0,
                "detected_type": detected_type,
                "chunk_strategy": chunk_strategy or "no-chunks",
                "parse_ms": parse_ms,
                "chunk_ms": chunk_ms,
            }

        # Compute light metrics
        section_idxs = set()
        token_vals = []
        for c in file_chunks:
            md = c.get("chunk_metadata") or {}
            if isinstance(md, dict):
                si = md.get("section_idx")
                if si is not None:
                    section_idxs.add(si)
                tv = md.get("tokens_est")
                if isinstance(tv, int):
                    token_vals.append(tv)
        section_count = len(section_idxs) if section_idxs else 0
        tokens_est_total = sum(token_vals) if token_vals else None
        tokens_est_mean = int(round(tokens_est_total / len(token_vals))) if token_vals else None

        return {
            "filepath": filepath,
            "status": "ok",
            "format": file_fmt,
            "chunks": file_chunks,
            "text_len": text_len,
            "chunk_count": len(file_chunks),
            "detected_type": detected_type,
            "chunk_strategy": chunk_strategy or "semantic",
            "section_count": section_count,
            "tokens_est_total": tokens_est_total,
            "tokens_est_mean": tokens_est_mean,
            "parse_ms": parse_ms,
            "chunk_ms": chunk_ms,
        }
    except _Timeout:
        # Fallback-naive on chunking timeout (parse succeeded)
        try:
            if "base_doc" in locals() and isinstance(base_doc, dict) and base_doc.get("text"):
                cfg = ChunkingConfig(
                    target_tokens=int(token_target),
                    overlap_tokens=int(token_overlap),
                    max_tokens=int(token_max),
                )
                fb_chunks = _fallback_chunk_text(base_doc.get("text", ""), base_meta, cfg)  # type: ignore
                section_idxs = set()
                token_vals = []
                for c in fb_chunks:
                    md = c.get("chunk_metadata") or {}
                    if isinstance(md, dict):
                        si = md.get("section_idx")
                        if si is not None:
                            section_idxs.add(si)
                        tv = md.get("tokens_est")
                        if isinstance(tv, int):
                            token_vals.append(tv)
                section_count = len(section_idxs) if section_idxs else 0
                tokens_est_total = sum(token_vals) if token_vals else None
                tokens_est_mean = int(round(tokens_est_total / len(token_vals))) if token_vals else None
                return {
                    "filepath": filepath,
                    "status": "fallback_ok",
                    "format": (base_doc.get("format") or Path(filepath).suffix.lower().strip(".") or "txt"),
                    "chunks": fb_chunks,
                    "text_len": len(base_doc.get("text","")),  # type: ignore
                    "chunk_count": len(fb_chunks),
                    "detected_type": detect_doc_type({}, base_doc.get("text","")),  # type: ignore
                    "chunk_strategy": "fallback-naive",
                    "section_count": section_count,
                    "tokens_est_total": tokens_est_total,
                    "tokens_est_mean": tokens_est_mean,
                    "parse_ms": None,
                    "chunk_ms": None,
                }
        except Exception as e2:
            return {"filepath": filepath, "status": "error", "error": f"Timeout + fallback error: {e2}"}
        return {"filepath": filepath, "status": "error", "error": "Timeout in chunking"}
    except Exception as e:
        return {"filepath": filepath, "status": "error", "error": str(e)}


# Avoid CUDA-for-fork deadlocks: prefer 'spawn' so worker pool processes do NOT inherit CUDA context
try:
    mp.set_start_method("spawn", force=False)
except RuntimeError:
    # Already set elsewhere; ignore
    pass


def run_worker_pipelined(
    session_name: str,
    root_dir: Optional[str],
    partition_file: Optional[str],
    embedding_model: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
    log_dir: str,
    resume: bool = False,
) -> None:
    """
    Pipelined worker:
      - Stage A in ProcessPool: parse+chunk (CPU parallel)
      - Stage B in main: GPU embed
      - Stage C in main: DB insert
    """
    # Early diagnostics to surface stalls on new instances
    try:
        from urllib.parse import urlparse as _urlparse
        _p = _urlparse(DB_URL)
        _safe_db = f"{_p.scheme}://{_p.hostname}:{_p.port}/{_p.path.lstrip('/')}"
    except Exception:
        _safe_db = DB_URL
    print(f"[beta_worker] start session={session_name} cwd={os.getcwd()} DB={_safe_db}", flush=True)
    try:
        with engine.connect() as _conn:
            _conn.execute(text("SELECT 1"))
        print("[beta_worker] DB ping OK", flush=True)
    except Exception as _e:
        print(f"[beta_worker] DB ping FAILED: {_e}", flush=True)
        raise
    print("[beta_worker] Ensuring schema...", flush=True)
    create_all_tables()
    print("[beta_worker] Schema OK", flush=True)

    # Resolve file list
    if partition_file:
        files = read_partition_file(partition_file)
    else:
        if not root_dir:
            raise ValueError("Either --partition_file or --root must be provided")
        files = find_all_supported_files(root_dir)
    if resume or _truthy(os.environ.get("OS_RESUME_FROM_LOGS")):
        before = len(files)
        done = _completed_from_success_logs(log_dir)
        if done:
            files = [fp for fp in files if fp not in done]
        print(f"[beta_worker] {session_name}: resume enabled -> skipping {before - len(files)} already-completed files", flush=True)
    # Worker-local ordering strategy (default: smooth size-zigzag, not strict largest-first)
    try:
        files, order_mode = _order_worker_files(files)
        print(f"[beta_worker] {session_name}: ordered {len(files)} files using {order_mode}", flush=True)
    except Exception as e:
        print(f"[beta_worker] {session_name}: note: could not apply worker file order: {e}", flush=True)

    # Defer CUDA model initialization until AFTER the worker pool is created to avoid CUDA context fork
    embedder = None  # type: ignore

    processed_chunks = 0
    successes: List[str] = []
    failures: List[str] = []

    total_files = len(files)
    os.makedirs(log_dir, exist_ok=True)
    try:
        from urllib.parse import urlparse as _urlparse
        _p = _urlparse(DB_URL)
        _safe_db = f"{_p.scheme}://{_p.hostname}:{_p.port}/{_p.path.lstrip('/')}"
    except Exception:
        _safe_db = DB_URL
    _succ_path = os.path.join(log_dir, f"{session_name}.success.log")
    failed_paths_path = os.path.join(log_dir, f"{session_name}.failed.paths.txt")
    failed_ndjson_path = os.path.join(log_dir, f"{session_name}.failed.ndjson")
    _err_path = os.path.join(log_dir, f"{session_name}.error.log")
    print(f"[beta_worker] {session_name}: DB = {_safe_db}", flush=True)
    print(f"[beta_worker] {session_name}: logs: success->{_succ_path}, error->{_err_path}", flush=True)
    print(f"[beta_worker] {session_name}: starting. files={total_files}, batch_size={int(os.environ.get('AUSLEGALSEARCH_EMBED_BATCH','64'))}, cpu_workers={CPU_WORKERS}, prefetch={PIPELINE_PREFETCH}", flush=True)

    # Producer-consumer pipeline
    with SessionLocal() as dbs:
        _maybe_print_counts(dbs, session_name, "baseline")

        file_iter = iter(files)
        inflight = {}
        submitted = 0
        done_count = 0

        with ProcessPoolExecutor(max_workers=CPU_WORKERS) as pool:
            # Initialize embedder AFTER pool is created so forked processes don't inherit CUDA context
            nonlocal_embedder = Embedder(embedding_model) if embedding_model else Embedder()
            embedder = nonlocal_embedder  # type: ignore

            # Helper to submit up to prefetch
            def _submit_next():
                nonlocal submitted
                while len(inflight) < PIPELINE_PREFETCH:
                    try:
                        fp = next(file_iter)
                    except StopIteration:
                        return
                    # Check session-file status; skip completed
                    esf = _db_get_esf_with_retry(dbs, session_name, fp)
                    if esf and getattr(esf, "status", None) == "complete":
                        done_count_local = submitted + done_count
                        if done_count_local % 200 == 0:
                            print(f"[beta_worker] {session_name}: SKIP already complete {done_count_local}/{total_files} -> {fp}", flush=True)
                        continue
                    if not esf:
                        esf = _db_ensure_pending_esf(dbs, session_name, fp)
                    submitted += 1
                    print(f"[beta_worker] {session_name}: start {submitted}/{total_files} -> {fp}", flush=True)
                    fut = pool.submit(
                        _cpu_prepare_file,
                        fp, root_dir, int(token_target), int(token_overlap), int(token_max)
                    )
                    inflight[fut] = fp

            _submit_next()

            while inflight:
                # take first completed
                for fut in as_completed(list(inflight.keys()), timeout=None):
                    fp = inflight.pop(fut)
                    done_count += 1
                    try:
                        res = fut.result()
                    except Exception as e:
                        # Child crashed
                        try:
                            esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=fp).first()
                            if esf:
                                esf.status = "error"
                                dbs.commit()
                        except Exception:
                            pass
                        failures.append(fp)
                        _append_log_line(log_dir, session_name, fp, success=False)
                        print(f"[beta_worker] {session_name}: FAILED (prep) {done_count}/{total_files} -> {fp} :: {e}", flush=True)
                        _append_error_detail(
                            log_dir, session_name, fp,
                            "prep", type(e).__name__, str(e),
                            None, None, None
                        )
                        break  # process next completion

                    status = res.get("status")
                    if status in ("empty", "zero_chunks"):
                        # mark complete with 0 chunks
                        try:
                            esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=fp).first()
                            if esf:
                                esf.status = "complete"
                                dbs.commit()
                        except Exception:
                            pass
                        successes.append(fp)
                        _append_success_metrics_line(
                            log_dir=log_dir,
                            session_name=session_name,
                            file_path=fp,
                            chunks_count=0,
                            text_len=res.get("text_len") or 0,
                            cfg=ChunkingConfig(target_tokens=int(token_target), overlap_tokens=int(token_overlap), max_tokens=int(token_max)),
                            strategy=res.get("chunk_strategy") or "no-chunks",
                            detected_type=res.get("detected_type"),
                            section_count=0,
                            tokens_est_total=0,
                            tokens_est_mean=0,
                            parse_ms=res.get("parse_ms"),
                            chunk_ms=res.get("chunk_ms"),
                            embed_ms=None,
                            insert_ms=None,
                        )
                        print(f"[beta_worker] {session_name}: OK (0 chunks) {done_count}/{total_files} -> {fp}", flush=True)
                        break

                    if status == "error":
                        try:
                            esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=fp).first()
                            if esf:
                                esf.status = "error"
                                dbs.commit()
                        except Exception:
                            pass
                        failures.append(fp)
                        _append_log_line(log_dir, session_name, fp, success=False)
                        print(f"[beta_worker] {session_name}: FAILED (prep) {done_count}/{total_files} -> {fp} :: {res.get('error')}", flush=True)
                        _append_error_detail(
                            log_dir, session_name, fp,
                            "prep", "Error", str(res.get("error") or ""),
                            None, None, None
                        )
                        break

                    # Embed + Insert (main process)
                    file_chunks = res.get("chunks") or []
                    current_chunk_count = int(res.get("chunk_count") or len(file_chunks))
                    current_text_len = int(res.get("text_len") or 0)
                    chunk_strategy = res.get("chunk_strategy") or "semantic"
                    detected_type = res.get("detected_type")

                    texts = [c["text"] for c in file_chunks]
                    print(f"[beta_worker] {session_name}: parse done {res.get('parse_ms')}ms len={current_text_len}", flush=True)
                    print(f"[beta_worker] {session_name}: chunk done {res.get('chunk_ms')}ms chunks={current_chunk_count}", flush=True)

                    # embed
                    print(f"[beta_worker] {session_name}: embed start batch={int(os.environ.get('AUSLEGALSEARCH_EMBED_BATCH','64'))} texts={len(texts)}", flush=True)
                    tE = time.time()
                    vecs = _embed_in_batches(embedder, texts, batch_size=int(os.environ.get("AUSLEGALSEARCH_EMBED_BATCH","64")))
                    embed_ms = int((time.time() - tE) * 1000)
                    print(f"[beta_worker] {session_name}: embed done {embed_ms}ms", flush=True)

                    # insert
                    print(f"[beta_worker] {session_name}: insert start", flush=True)
                    tI = time.time()
                    fmt = "txt"
                    inserted = _db_insert_with_retry(
                        session=dbs,
                        chunks= file_chunks,
                        vectors=vecs,
                        source_path=fp,
                        fmt=fmt,
                    )
                    insert_ms = int((time.time() - tI) * 1000)
                    print(f"[beta_worker] {session_name}: insert done {insert_ms}ms rows={inserted}", flush=True)
                    processed_chunks += inserted

                    # mark complete + progress
                    try:
                        esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=fp).first()
                        if esf:
                            esf.status = "complete"
                            dbs.commit()
                    except Exception:
                        pass
                    try:
                        with _deadline(SELECT_TIMEOUT):
                            update_session_progress(session_name, last_file=fp, last_chunk=inserted - 1, processed_chunks=processed_chunks)
                    except _Timeout:
                        print(f"[beta_worker] {session_name}: progress update timeout for {fp}", flush=True)

                    successes.append(fp)
                    _append_success_metrics_line(
                        log_dir=log_dir,
                        session_name=session_name,
                        file_path=fp,
                        chunks_count=current_chunk_count or 0,
                        text_len=current_text_len or 0,
                        cfg=ChunkingConfig(target_tokens=int(token_target), overlap_tokens=int(token_overlap), max_tokens=int(token_max)),
                        strategy=chunk_strategy,
                        detected_type=detected_type,
                        section_count=res.get("section_count"),
                        tokens_est_total=res.get("tokens_est_total"),
                        tokens_est_mean=res.get("tokens_est_mean"),
                        parse_ms=res.get("parse_ms"),
                        chunk_ms=res.get("chunk_ms"),
                        embed_ms=embed_ms,
                        insert_ms=insert_ms,
                    )

                    if done_count % 10 == 0 or inserted > 0:
                        print(f"[beta_worker] {session_name}: OK {done_count}/{total_files}, chunks+={inserted}, total_chunks={processed_chunks}", flush=True)
                    if os.environ.get("AUSLEGALSEARCH_DEBUG_COUNTS", "0") == "1" and done_count % 50 == 0:
                        _maybe_print_counts(dbs, session_name, f"after {done_count} files")
                    break  # process next completion

                _submit_next()

    # Write logs
    paths = _write_logs(log_dir, session_name, successes, failures)
    print(f"[beta_worker] Session {session_name} complete. Files OK: {len(successes)}, failed: {len(failures)}", flush=True)
    print(f"[beta_worker] Success log: {paths['success_log']}", flush=True)
    print(f"[beta_worker] Error log:   {paths['error_log']}", flush=True)
    complete_session(session_name)


def run_worker(
    session_name: str,
    root_dir: Optional[str],
    partition_file: Optional[str],
    embedding_model: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
    log_dir: str,
    resume: bool = False,
) -> None:
    """
    Main loop for this worker.
    """
    # Use pipelined mode when multiple CPU workers are configured
    if CPU_WORKERS > 1 or int(os.environ.get("AUSLEGALSEARCH_PIPELINE_PREFETCH", str(PIPELINE_PREFETCH))) > 0:
        return run_worker_pipelined(session_name, root_dir, partition_file, embedding_model, token_target, token_overlap, token_max, log_dir, resume)

    # Early diagnostics
    try:
        from urllib.parse import urlparse as _urlparse
        _p = _urlparse(DB_URL)
        _safe_db = f"{_p.scheme}://{_p.hostname}:{_p.port}/{_p.path.lstrip('/')}"
    except Exception:
        _safe_db = DB_URL
    print(f"[beta_worker] start (single) session={session_name} cwd={os.getcwd()} DB={_safe_db}", flush=True)
    try:
        with engine.connect() as _conn:
            _conn.execute(text("SELECT 1"))
        print("[beta_worker] DB ping OK", flush=True)
    except Exception as _e:
        print(f"[beta_worker] DB ping FAILED: {_e}", flush=True)
        raise
    print("[beta_worker] Ensuring schema...", flush=True)
    create_all_tables()  # Ensure extensions, indexes, triggers
    print("[beta_worker] Schema OK", flush=True)

    # Resolve file list
    if partition_file:
        files = read_partition_file(partition_file)
    else:
        if not root_dir:
            raise ValueError("Either --partition_file or --root must be provided")
        files = find_all_supported_files(root_dir)
    if resume or _truthy(os.environ.get("OS_RESUME_FROM_LOGS")):
        before = len(files)
        done = _completed_from_success_logs(log_dir)
        if done:
            files = [fp for fp in files if fp not in done]
        print(f"[beta_worker] {session_name}: resume enabled -> skipping {before - len(files)} already-completed files", flush=True)
    # Worker-local ordering strategy (default: smooth size-zigzag, not strict largest-first)
    try:
        files, order_mode = _order_worker_files(files)
        print(f"[beta_worker] {session_name}: ordered {len(files)} files using {order_mode}", flush=True)
    except Exception as e:
        print(f"[beta_worker] {session_name}: note: could not apply worker file order: {e}", flush=True)

    embedder = Embedder(embedding_model) if embedding_model else Embedder()
    cfg = ChunkingConfig(
        target_tokens=int(token_target),
        overlap_tokens=int(token_overlap),
        max_tokens=int(token_max),
    )

    batch_size = int(os.environ.get("AUSLEGALSEARCH_EMBED_BATCH", "64"))

    processed_chunks = 0
    successes: List[str] = []
    failures: List[str] = []

    total_files = len(files)
    # Ensure log dir exists; print DB target and log file paths for diagnostics
    os.makedirs(log_dir, exist_ok=True)
    try:
        from urllib.parse import urlparse as _urlparse
        _p = _urlparse(DB_URL)
        _safe_db = f"{_p.scheme}://{_p.hostname}:{_p.port}/{_p.path.lstrip('/')}"
    except Exception:
        _safe_db = DB_URL
    _succ_path = os.path.join(log_dir, f"{session_name}.success.log")
    _err_path = os.path.join(log_dir, f"{session_name}.error.log")
    print(f"[beta_worker] {session_name}: DB = {_safe_db}", flush=True)
    print(f"[beta_worker] {session_name}: logs: success->{_succ_path}, error->{_err_path}", flush=True)
    print(f"[beta_worker] {session_name}: starting. files={total_files}, batch_size={batch_size}", flush=True)

    with SessionLocal() as dbs:
        current_stage: Optional[str] = None
        stage_start: float = 0.0
        current_text_len: Optional[int] = None
        current_chunk_count: Optional[int] = None
        chunk_strategy: Optional[str] = None
        # Per-stage timings (ms)
        parse_ms: Optional[int] = None
        chunk_ms: Optional[int] = None
        embed_ms: Optional[int] = None
        insert_ms: Optional[int] = None
        _maybe_print_counts(dbs, session_name, "baseline")
        for idx_f, filepath in enumerate(files, start=1):
            try:
                # Track per-file status row with bounded DB deadlines/retries (prevents hangs before 'parse')
                esf = _db_get_esf_with_retry(dbs, session_name, filepath)
                if esf and getattr(esf, "status", None) == "complete":
                    if idx_f % 200 == 0:
                        print(f"[beta_worker] {session_name}: SKIP already complete {idx_f}/{total_files} -> {filepath}", flush=True)
                    continue
                if not esf:
                    esf = _db_ensure_pending_esf(dbs, session_name, filepath)
                # Announce the start of active processing for this file (helps identify a stuck file quickly)
                print(f"[beta_worker] {session_name}: start {idx_f}/{total_files} -> {filepath}", flush=True)

                current_stage = "parse"
                stage_start = time.time()
                print(f"[beta_worker] {session_name}: parse start -> {filepath}", flush=True)
                with _deadline(PARSE_TIMEOUT):
                    base_doc = parse_file(filepath)
                current_text_len = len(base_doc.get("text", ""))
                parse_ms = int((time.time() - stage_start) * 1000)
                print(f"[beta_worker] {session_name}: parse done {parse_ms}ms len={current_text_len}", flush=True)
                if not base_doc or not base_doc.get("text"):
                    esf.status = "error"
                    dbs.commit()
                    failures.append(filepath)
                    _append_log_line(log_dir, session_name, filepath, success=False)
                    print(f"[beta_worker] {session_name}: FAILED (empty) {idx_f}/{total_files} -> {filepath}", flush=True)
                    _append_error_detail(
                        log_dir, session_name, filepath,
                        current_stage or "parse", "EmptyText", "Parsed file has no text",
                        int((time.time() - stage_start) * 1000) if stage_start else None,
                        {"text_len": current_text_len}
                    )
                    continue

                # Enrich metadata
                path_meta = derive_path_metadata(filepath, root_dir or "")
                base_meta = dict(base_doc.get("chunk_metadata") or {})
                base_meta.update(path_meta)

                # Detect content type for analytics
                detected_type = detect_doc_type(base_meta, base_doc.get("text", ""))
                if detected_type and not base_meta.get("type"):
                    base_meta["type"] = detected_type

                # Semantic chunking
                current_stage = "chunk"
                stage_start = time.time()
                print(f"[beta_worker] {session_name}: chunk start", flush=True)
                with _deadline(CHUNK_TIMEOUT):
                    # Try dashed-header aware chunking first (works for any dashed-header format).
                    file_chunks = chunk_legislation_dashed_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
                    if file_chunks:
                        chunk_strategy = "dashed-semantic"
                    else:
                        # Fallback to generic semantic chunking (heading/sentences with same cfg)
                        file_chunks = chunk_document_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
                        if file_chunks:
                            chunk_strategy = "semantic"
                    # Optional RCTS fallback for generic types (env-controlled)
                    if (not file_chunks) and os.environ.get("AUSLEGALSEARCH_USE_RCTS_GENERIC", "0") == "1":
                        rcts_chunks = chunk_generic_rcts(base_doc["text"], base_meta=base_meta, cfg=cfg)
                        if rcts_chunks:
                            file_chunks = rcts_chunks
                            chunk_strategy = "rcts-generic"
                current_chunk_count = len(file_chunks)
                chunk_ms = int((time.time() - stage_start) * 1000)
                print(f"[beta_worker] {session_name}: chunk done {chunk_ms}ms chunks={current_chunk_count}", flush=True)
                if not file_chunks:
                    esf.status = "complete"
                    dbs.commit()
                    successes.append(filepath)
                    _append_success_metrics_line(
                        log_dir=log_dir,
                        session_name=session_name,
                        file_path=filepath,
                        chunks_count=0,
                        text_len=current_text_len or 0,
                        cfg=cfg,
                        strategy=chunk_strategy or "no-chunks",
                        detected_type=detected_type,
                        section_count=0,
                        tokens_est_total=0,
                        tokens_est_mean=0,
                        parse_ms=parse_ms,
                        chunk_ms=chunk_ms,
                        embed_ms=None,
                        insert_ms=None,
                    )
                    print(f"[beta_worker] {session_name}: OK (0 chunks) {idx_f}/{total_files} -> {filepath}", flush=True)
                    continue

                # Batch embed
                texts = [c["text"] for c in file_chunks]
                current_stage = "embed"
                stage_start = time.time()
                print(f"[beta_worker] {session_name}: embed start batch={batch_size} texts={len(texts)}", flush=True)
                vecs = _embed_in_batches(embedder, texts, batch_size=batch_size)
                embed_ms = int((time.time() - stage_start) * 1000)
                print(f"[beta_worker] {session_name}: embed done {embed_ms}ms", flush=True)

                # Insert
                current_stage = "insert"
                stage_start = time.time()
                fmt = base_doc.get("format", Path(filepath).suffix.lower().strip("."))
                print(f"[beta_worker] {session_name}: insert start", flush=True)
                inserted = _db_insert_with_retry(
                    session=dbs,
                    chunks=file_chunks,
                    vectors=vecs,
                    source_path=filepath,
                    fmt=fmt,
                )
                insert_ms = int((time.time() - stage_start) * 1000)
                print(f"[beta_worker] {session_name}: insert done {insert_ms}ms rows={inserted}", flush=True)
                processed_chunks += inserted

                # Update progress
                esf.status = "complete"
                dbs.commit()
                try:
                    with _deadline(SELECT_TIMEOUT):
                        update_session_progress(session_name, last_file=filepath, last_chunk=inserted - 1, processed_chunks=processed_chunks)
                except _Timeout:
                    print(f"[beta_worker] {session_name}: progress update timeout for {filepath}", flush=True)
                successes.append(filepath)
                # Compute light-weight per-file metrics for logging (O(n) in chunks; negligible vs embedding)
                try:
                    section_idxs = set()
                    token_vals = []
                    for c in file_chunks:
                        md = c.get("chunk_metadata") or {}
                        if isinstance(md, dict):
                            si = md.get("section_idx")
                            if si is not None:
                                section_idxs.add(si)
                            tv = md.get("tokens_est")
                            if isinstance(tv, int):
                                token_vals.append(tv)
                    section_count = len(section_idxs) if section_idxs else 0
                    tokens_est_total = sum(token_vals) if token_vals else None
                    tokens_est_mean = int(round(tokens_est_total / len(token_vals))) if token_vals else None
                except Exception:
                    section_count = None
                    tokens_est_total = None
                    tokens_est_mean = None
                _append_success_metrics_line(
                    log_dir=log_dir,
                    session_name=session_name,
                    file_path=filepath,
                    chunks_count=current_chunk_count or 0,
                    text_len=current_text_len or 0,
                    cfg=cfg,
                    strategy=chunk_strategy or "semantic",
                    detected_type=detected_type,
                    section_count=section_count,
                    tokens_est_total=tokens_est_total,
                    tokens_est_mean=tokens_est_mean,
                    parse_ms=parse_ms,
                    chunk_ms=chunk_ms,
                    embed_ms=embed_ms,
                    insert_ms=insert_ms,
                )

                if idx_f % 10 == 0 or inserted > 0:
                    print(f"[beta_worker] {session_name}: OK {idx_f}/{total_files}, chunks+={inserted}, total_chunks={processed_chunks}", flush=True)
                if os.environ.get("AUSLEGALSEARCH_DEBUG_COUNTS", "0") == "1" and idx_f % 50 == 0:
                    _maybe_print_counts(dbs, session_name, f"after {idx_f} files")

            except _Timeout as e:
                # Try fallback if timeout occurred during chunking
                if (current_stage == "chunk"
                    and os.environ.get("AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT", "1") != "0"
                    and "base_doc" in locals()
                    and isinstance(base_doc, dict)
                    and base_doc.get("text")):
                    try:
                        fb_chunks = _fallback_chunk_text(base_doc["text"], base_meta, cfg)
                        texts = [c["text"] for c in fb_chunks]
                        t0 = time.time()
                        vecs = _embed_in_batches(embedder, texts, batch_size=batch_size)
                        fb_embed_ms = int((time.time() - t0) * 1000)
                        t1 = time.time()
                        inserted = _batch_insert_chunks(
                            session=dbs,
                            chunks=fb_chunks,
                            vectors=vecs,
                            source_path=filepath,
                            fmt=base_doc.get("format", Path(filepath).suffix.lower().strip(".")),
                        )
                        fb_insert_ms = int((time.time() - t1) * 1000)
                        processed_chunks += inserted
                        try:
                            esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                            if esf:
                                esf.status = "complete"
                                dbs.commit()
                        except Exception:
                            pass
                        update_session_progress(session_name, last_file=filepath, last_chunk=inserted - 1, processed_chunks=processed_chunks)
                        successes.append(filepath)
                        _append_success_metrics_line(
                            log_dir=log_dir,
                            session_name=session_name,
                            file_path=filepath,
                            chunks_count=inserted,
                            text_len=len(base_doc.get("text","")),
                            cfg=cfg,
                            strategy="fallback-naive",
                            detected_type=detected_type,
                            parse_ms=parse_ms,
                            chunk_ms=None,
                            embed_ms=fb_embed_ms,
                            insert_ms=fb_insert_ms,
                        )
                        _append_error_detail(
                            log_dir, session_name, filepath,
                            "chunk", "Timeout", "Chunk timeout; fallback succeeded",
                            int((time.time() - stage_start) * 1000) if stage_start else None,
                            {"text_len": len(base_doc.get("text","")), "fallback": True, "fallback_chunks": len(fb_chunks), "batch_size": batch_size}
                        )
                        if idx_f % 10 == 0 or inserted > 0:
                            print(f"[beta_worker] {session_name}: OK (fallback) {idx_f}/{total_files}, chunks+={inserted}, total_chunks={processed_chunks}", flush=True)
                        continue
                    except Exception as e2:
                        # Fall through to treat as failure
                        e = e2
                # Failure path
                try:
                    esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                    if esf:
                        esf.status = "error"
                        dbs.commit()
                except Exception:
                    pass
                failures.append(filepath)
                _append_log_line(log_dir, session_name, filepath, success=False)
                print(f"[beta_worker] {session_name}: TIMEOUT {idx_f}/{total_files} -> {filepath} :: {e}", flush=True)
                _append_error_detail(
                    log_dir, session_name, filepath,
                    current_stage or "unknown", "Timeout", str(e),
                    int((time.time() - stage_start) * 1000) if stage_start else None,
                    {"text_len": current_text_len, "chunk_count": current_chunk_count, "batch_size": batch_size, "fallback": False}
                )
                continue
            except KeyboardInterrupt:
                print(f"[beta_worker] {session_name}: INTERRUPTED at {idx_f}/{total_files} -> {filepath}", flush=True)
                raise
            except Exception as e:
                # If failure in chunk stage, attempt naive fallback chunking
                if (current_stage == "chunk"
                    and os.environ.get("AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT", "1") != "0"
                    and "base_doc" in locals()
                    and isinstance(base_doc, dict)
                    and base_doc.get("text")):
                    try:
                        fb_chunks = _fallback_chunk_text(base_doc["text"], base_meta, cfg)
                        texts = [c["text"] for c in fb_chunks]
                        t0 = time.time()
                        vecs = _embed_in_batches(embedder, texts, batch_size=batch_size)
                        fb_embed_ms = int((time.time() - t0) * 1000)
                        t1 = time.time()
                        inserted = _batch_insert_chunks(
                            session=dbs,
                            chunks=fb_chunks,
                            vectors=vecs,
                            source_path=filepath,
                            fmt=base_doc.get("format", Path(filepath).suffix.lower().strip(".")),
                        )
                        fb_insert_ms = int((time.time() - t1) * 1000)
                        processed_chunks += inserted
                        try:
                            esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                            if esf:
                                esf.status = "complete"
                                dbs.commit()
                        except Exception:
                            pass
                        update_session_progress(session_name, last_file=filepath, last_chunk=inserted - 1, processed_chunks=processed_chunks)
                        successes.append(filepath)
                        _append_success_metrics_line(
                            log_dir=log_dir,
                            session_name=session_name,
                            file_path=filepath,
                            chunks_count=inserted,
                            text_len=len(base_doc.get("text","")),
                            cfg=cfg,
                            strategy="fallback-naive",
                            detected_type=detected_type,
                            parse_ms=parse_ms,
                            chunk_ms=None,
                            embed_ms=fb_embed_ms,
                            insert_ms=fb_insert_ms,
                        )
                        _append_error_detail(
                            log_dir, session_name, filepath,
                            "chunk", type(e).__name__, "Primary chunker error; fallback succeeded",
                            int((time.time() - stage_start) * 1000) if stage_start else None,
                            {"text_len": len(base_doc.get("text","")), "fallback": True, "fallback_chunks": len(fb_chunks), "batch_size": batch_size},
                            traceback.format_exc()
                        )
                        if idx_f % 10 == 0 or inserted > 0:
                            print(f"[beta_worker] {session_name}: OK (fallback) {idx_f}/{total_files}, chunks+={inserted}, total_chunks={processed_chunks}", flush=True)
                        continue
                    except Exception as e2:
                        # Fall through to regular failure handling
                        e = e2
                # Regular failure handling
                try:
                    esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                    if esf:
                        esf.status = "error"
                        dbs.commit()
                except Exception:
                    pass
                failures.append(filepath)
                _append_log_line(log_dir, session_name, filepath, success=False)
                print(f"[beta_worker] {session_name}: FAILED {idx_f}/{total_files} -> {filepath} :: {e}", flush=True)
                _append_error_detail(
                    log_dir, session_name, filepath,
                    current_stage or "unknown", type(e).__name__, str(e),
                    int((time.time() - stage_start) * 1000) if stage_start else None,
                    {"text_len": current_text_len, "chunk_count": current_chunk_count, "batch_size": batch_size},
                    traceback.format_exc()
                )
                continue

    # Write logs
    paths = _write_logs(log_dir, session_name, successes, failures)
    print(f"[beta_worker] Session {session_name} complete. Files OK: {len(successes)}, failed: {len(failures)}", flush=True)
    print(f"[beta_worker] Success log: {paths['success_log']}", flush=True)
    print(f"[beta_worker] Error log:   {paths['error_log']}", flush=True)

    # Mark session complete (do not hard-fail on partial errors; logs capture details)
    complete_session(session_name)


def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    import argparse
    ap = argparse.ArgumentParser(description="Beta ingestion worker: semantic chunking + embeddings + DB write.")
    ap.add_argument("session_name", help="Child session name (e.g., sess-...-gpu0)")
    ap.add_argument("--root", default=None, help="Root directory (used if --partition_file not provided)")
    ap.add_argument("--partition_file", default=None, help="Text file listing file paths for this worker")
    ap.add_argument("--model", default=None, help="Embedding model (optional)")
    ap.add_argument("--target_tokens", type=int, default=512, help="Chunking target tokens (default 512)")
    ap.add_argument("--overlap_tokens", type=int, default=64, help="Chunk overlap tokens (default 64)")
    ap.add_argument("--max_tokens", type=int, default=640, help="Hard max per chunk (default 640)")
    ap.add_argument("--log_dir", default="./logs", help="Directory to write per-worker success/error logs")
    ap.add_argument("--resume", action="store_true", help="Skip files already present in *.success.log files under --log_dir")
    return vars(ap.parse_args(argv))


def run_worker_opensearch(
    session_name: str,
    root_dir: Optional[str],
    partition_file: Optional[str],
    embedding_model: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
    log_dir: str,
    resume: bool = False,
) -> None:
    """
    OpenSearch-safe worker path that avoids direct SQLAlchemy session usage.
    P0 scale hardening:
    - bulk upsert per file,
    - idempotent keys via db.store bulk helper,
    - stage deadlines and retry/backoff,
    - optional RCTS + naive fallback chunking.
    """
    # Use pipelined CPU prep + GPU embed/index overlap when configured.
    if _opensearch_pipeline_enabled():
        return run_worker_opensearch_pipelined(
            session_name=session_name,
            root_dir=root_dir,
            partition_file=partition_file,
            embedding_model=embedding_model,
            token_target=token_target,
            token_overlap=token_overlap,
            token_max=token_max,
            log_dir=log_dir,
            resume=resume,
        )

    # Schema/index bootstrap is usually handled by orchestrator once before launching workers.
    # Re-running this in every worker is expensive and can conflict with temporary ingest-time
    # index tuning (e.g., replicas forced to 0 during bulk windows).
    if _truthy(os.environ.get("AUSLEGALSEARCH_WORKER_SCHEMA_INIT", "0")):
        create_all_tables()
    else:
        print(f"[beta_worker/opensearch] {session_name}: skipping create_all_tables (AUSLEGALSEARCH_WORKER_SCHEMA_INIT=0)", flush=True)
    if partition_file:
        files = read_partition_file(partition_file)
    else:
        if not root_dir:
            raise ValueError("Either --partition_file or --root must be provided")
        files = find_all_supported_files(root_dir)
    if resume or _truthy(os.environ.get("OS_RESUME_FROM_LOGS")):
        before = len(files)
        done = _completed_from_success_logs(log_dir)
        if done:
            files = [fp for fp in files if fp not in done]
        print(f"[beta_worker/opensearch] {session_name}: resume enabled -> skipping {before - len(files)} already-completed files", flush=True)
    try:
        files, order_mode = _order_worker_files(files)
        print(f"[beta_worker/opensearch] {session_name}: ordered {len(files)} files using {order_mode}", flush=True)
    except Exception as e:
        print(f"[beta_worker/opensearch] {session_name}: note: could not apply worker file order: {e}", flush=True)

    embedder = Embedder(embedding_model) if embedding_model else Embedder()
    cfg = ChunkingConfig(
        target_tokens=int(token_target),
        overlap_tokens=int(token_overlap),
        max_tokens=int(token_max),
    )
    batch_size = int(os.environ.get("AUSLEGALSEARCH_EMBED_BATCH", "64"))
    insert_retries = int(os.environ.get("AUSLEGALSEARCH_DB_MAX_RETRIES", str(DB_MAX_RETRIES)))

    processed_chunks = 0
    successes: List[str] = []
    failures: List[str] = []
    governor = _RuntimeGovernor()
    governor.initialize(session_name)
    state = _OpenSearchIngestState(enabled=_truthy(os.environ.get("OS_INGEST_STATE_ENABLE")) and not _throughput_mode_enabled())
    state.upsert(f"{session_name}::summary", {
        "type": "session",
        "session": session_name,
        "status": "running",
        "total_files": len(files),
        "files_ok": 0,
        "files_failed": 0,
        "total_indexed": 0,
    })

    for idx_f, filepath in enumerate(files, start=1):
        t_file = time.time()
        parse_ms: Optional[int] = None
        chunk_ms: Optional[int] = None
        embed_ms: Optional[int] = None
        index_ms: Optional[int] = None
        try:
            row = get_session_file(session_name, filepath)
            if row and getattr(row, "status", None) == "complete":
                continue
            if not row:
                upsert_session_file_status(session_name, filepath, "pending")
            state.upsert(f"{session_name}::{filepath}", {
                "type": "file",
                "session": session_name,
                "file": filepath,
                "status": "pending",
            })

            t0 = time.time()
            with _deadline(PARSE_TIMEOUT):
                base_doc = parse_file(filepath)
            parse_ms = int((time.time() - t0) * 1000)
            if not base_doc or not base_doc.get("text"):
                upsert_session_file_status(session_name, filepath, "error")
                failures.append(filepath)
                _append_log_line(log_dir, session_name, filepath, success=False)
                _append_failed_reingest_entry(
                    log_dir=log_dir,
                    session_name=session_name,
                    file_path=filepath,
                    stage="parse",
                    error_type="EmptyText",
                    message="Parsed file has no text",
                    meta={"parse_ms": parse_ms},
                )
                _append_metrics_ndjson(log_dir, session_name, {
                    "file": filepath,
                    "status": "error",
                    "stage": "parse",
                    "parse_ms": parse_ms,
                    "duration_ms": int((time.time() - t_file) * 1000),
                })
                state.upsert(f"{session_name}::{filepath}", {
                    "type": "file", "session": session_name, "file": filepath,
                    "status": "error", "stage": "parse", "parse_ms": parse_ms,
                })
                continue

            path_meta = derive_path_metadata(filepath, root_dir or "")
            base_meta = dict(base_doc.get("chunk_metadata") or {})
            base_meta.update(path_meta)

            detected_type = detect_doc_type(base_meta, base_doc.get("text", ""))
            if detected_type and not base_meta.get("type"):
                base_meta["type"] = detected_type

            t1 = time.time()
            try:
                with _deadline(CHUNK_TIMEOUT):
                    file_chunks = chunk_legislation_dashed_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
                    if not file_chunks:
                        file_chunks = chunk_document_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)

                    if (not file_chunks) and os.environ.get("AUSLEGALSEARCH_USE_RCTS_GENERIC", "0") == "1":
                        file_chunks = chunk_generic_rcts(base_doc["text"], base_meta=base_meta, cfg=cfg)

                    if not file_chunks and os.environ.get("AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT", "1") != "0":
                        file_chunks = _fallback_chunk_text(base_doc["text"], base_meta, cfg)
            except _Timeout:
                if os.environ.get("AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT", "1") != "0":
                    file_chunks = _fallback_chunk_text(base_doc["text"], base_meta, cfg)
                    print(f"[beta_worker/opensearch] {session_name}: chunk timeout -> fallback chunker used for {filepath}", flush=True)
                else:
                    raise
            chunk_ms = int((time.time() - t1) * 1000)

            if not file_chunks:
                upsert_session_file_status(session_name, filepath, "complete")
                successes.append(filepath)
                _append_log_line(log_dir, session_name, filepath, success=True)
                _append_metrics_ndjson(log_dir, session_name, {
                    "file": filepath,
                    "status": "ok",
                    "chunks": 0,
                    "parse_ms": parse_ms,
                    "chunk_ms": chunk_ms,
                    "duration_ms": int((time.time() - t_file) * 1000),
                })
                state.upsert(f"{session_name}::{filepath}", {
                    "type": "file", "session": session_name, "file": filepath,
                    "status": "complete", "chunks": 0,
                    "parse_ms": parse_ms, "chunk_ms": chunk_ms,
                })
                continue

            fmt = base_doc.get("format", Path(filepath).suffix.lower().strip("."))

            inserted, embed_ms, index_ms = _embed_and_index_file_chunks_opensearch(
                embedder=embedder,
                source_path=filepath,
                fmt=fmt,
                file_chunks=file_chunks,
                batch_size=batch_size,
                insert_retries=insert_retries,
                governor=governor,
                session_name=session_name,
            )

            processed_chunks += inserted
            upsert_session_file_status(session_name, filepath, "complete")
            update_session_progress(session_name, last_file=filepath, last_chunk=inserted - 1, processed_chunks=processed_chunks)
            successes.append(filepath)
            _append_log_line(log_dir, session_name, filepath, success=True)
            _append_metrics_ndjson(log_dir, session_name, {
                "file": filepath,
                "status": "ok",
                "chunks": len(file_chunks),
                "parse_ms": parse_ms,
                "chunk_ms": chunk_ms,
                "embed_ms": embed_ms,
                "index_ms": index_ms,
                "duration_ms": int((time.time() - t_file) * 1000),
            })
            state.upsert(f"{session_name}::{filepath}", {
                "type": "file", "session": session_name, "file": filepath,
                "status": "complete", "chunks": len(file_chunks),
                "parse_ms": parse_ms, "chunk_ms": chunk_ms,
                "embed_ms": embed_ms, "index_ms": index_ms,
            })
            if idx_f % 10 == 0:
                print(f"[beta_worker/opensearch] {session_name}: {idx_f}/{len(files)} processed, chunks={processed_chunks}", flush=True)
        except Exception as e:
            try:
                upsert_session_file_status(session_name, filepath, "error")
            except Exception:
                pass
            failures.append(filepath)
            _append_log_line(log_dir, session_name, filepath, success=False)
            _append_failed_reingest_entry(
                log_dir=log_dir,
                session_name=session_name,
                file_path=filepath,
                stage="opensearch",
                error_type=type(e).__name__,
                message=str(e),
                meta={
                    "parse_ms": parse_ms,
                    "chunk_ms": chunk_ms,
                    "embed_ms": embed_ms,
                    "index_ms": index_ms,
                    "duration_ms": int((time.time() - t_file) * 1000),
                },
            )
            print(f"[beta_worker/opensearch] FAILED {idx_f}/{len(files)} -> {filepath}: {e}", flush=True)
            _append_metrics_ndjson(log_dir, session_name, {
                "file": filepath,
                "status": "error",
                "error": str(e),
                "parse_ms": parse_ms,
                "chunk_ms": chunk_ms,
                "embed_ms": embed_ms,
                "index_ms": index_ms,
                "duration_ms": int((time.time() - t_file) * 1000),
            })
            state.upsert(f"{session_name}::{filepath}", {
                "type": "file", "session": session_name, "file": filepath,
                "status": "error", "error": str(e),
                "parse_ms": parse_ms, "chunk_ms": chunk_ms,
                "embed_ms": embed_ms, "index_ms": index_ms,
            })

    paths = _write_logs(log_dir, session_name, successes, failures)
    print(f"[beta_worker/opensearch] Session {session_name} complete. OK={len(successes)} failed={len(failures)}", flush=True)
    print(f"[beta_worker/opensearch] Success log: {paths['success_log']}", flush=True)
    print(f"[beta_worker/opensearch] Error log:   {paths['error_log']}", flush=True)
    state.upsert(f"{session_name}::summary", {
        "type": "session",
        "session": session_name,
        "status": "complete",
        "total_files": len(files),
        "files_ok": len(successes),
        "files_failed": len(failures),
        "total_indexed": processed_chunks,
    })
    complete_session(session_name)


def run_worker_opensearch_pipelined(
    session_name: str,
    root_dir: Optional[str],
    partition_file: Optional[str],
    embedding_model: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
    log_dir: str,
    resume: bool = False,
) -> None:
    """
    OpenSearch pipelined worker:
      - Stage A in ProcessPool: parse+chunk (CPU parallel)
      - Stage B in main: GPU embedding
      - Stage C in main: OpenSearch bulk upsert
    """
    if _truthy(os.environ.get("AUSLEGALSEARCH_WORKER_SCHEMA_INIT", "0")):
        create_all_tables()
    else:
        print(f"[beta_worker/opensearch] {session_name}: skipping create_all_tables (AUSLEGALSEARCH_WORKER_SCHEMA_INIT=0)", flush=True)

    if partition_file:
        files = read_partition_file(partition_file)
    else:
        if not root_dir:
            raise ValueError("Either --partition_file or --root must be provided")
        files = find_all_supported_files(root_dir)

    if resume or _truthy(os.environ.get("OS_RESUME_FROM_LOGS")):
        before = len(files)
        done = _completed_from_success_logs(log_dir)
        if done:
            files = [fp for fp in files if fp not in done]
        print(f"[beta_worker/opensearch] {session_name}: resume enabled -> skipping {before - len(files)} already-completed files", flush=True)

    try:
        files, order_mode = _order_worker_files(files)
        print(f"[beta_worker/opensearch] {session_name}: ordered {len(files)} files using {order_mode}", flush=True)
    except Exception as e:
        print(f"[beta_worker/opensearch] {session_name}: note: could not apply worker file order: {e}", flush=True)

    total_files = len(files)
    batch_size = int(os.environ.get("AUSLEGALSEARCH_EMBED_BATCH", "64"))
    insert_retries = int(os.environ.get("AUSLEGALSEARCH_DB_MAX_RETRIES", str(DB_MAX_RETRIES)))

    processed_chunks = 0
    successes: List[str] = []
    failures: List[str] = []
    governor = _RuntimeGovernor()
    governor.initialize(session_name)
    state = _OpenSearchIngestState(enabled=_truthy(os.environ.get("OS_INGEST_STATE_ENABLE")) and not _throughput_mode_enabled())
    state.upsert(f"{session_name}::summary", {
        "type": "session",
        "session": session_name,
        "status": "running",
        "total_files": total_files,
        "files_ok": 0,
        "files_failed": 0,
        "total_indexed": 0,
    })

    print(f"[beta_worker/opensearch] {session_name}: starting. files={total_files}, batch_size={batch_size}, cpu_workers={CPU_WORKERS}, prefetch={PIPELINE_PREFETCH}", flush=True)

    file_iter = iter(files)
    inflight = {}
    inflight_started: Dict[str, float] = {}
    submitted = 0
    done_count = 0

    with ProcessPoolExecutor(max_workers=CPU_WORKERS) as pool:
        embedder = Embedder(embedding_model) if embedding_model else Embedder()

        def _submit_next():
            nonlocal submitted
            while len(inflight) < PIPELINE_PREFETCH:
                try:
                    fp = next(file_iter)
                except StopIteration:
                    return

                row = get_session_file(session_name, fp)
                if row and getattr(row, "status", None) == "complete":
                    continue
                if not row:
                    upsert_session_file_status(session_name, fp, "pending")

                submitted += 1
                inflight_started[fp] = time.time()
                state.upsert(f"{session_name}::{fp}", {
                    "type": "file",
                    "session": session_name,
                    "file": fp,
                    "status": "pending",
                })
                print(f"[beta_worker/opensearch] {session_name}: start {submitted}/{total_files} -> {fp}", flush=True)
                fut = pool.submit(_cpu_prepare_file, fp, root_dir, int(token_target), int(token_overlap), int(token_max))
                inflight[fut] = fp

        _submit_next()

        while inflight:
            for fut in as_completed(list(inflight.keys()), timeout=None):
                fp = inflight.pop(fut)
                done_count += 1
                t_file = inflight_started.pop(fp, time.time())

                parse_ms: Optional[int] = None
                chunk_ms: Optional[int] = None
                embed_ms: Optional[int] = None
                index_ms: Optional[int] = None

                try:
                    res = fut.result()
                except Exception as e:
                    try:
                        upsert_session_file_status(session_name, fp, "error")
                    except Exception:
                        pass
                    failures.append(fp)
                    _append_log_line(log_dir, session_name, fp, success=False)
                    _append_failed_reingest_entry(
                        log_dir=log_dir,
                        session_name=session_name,
                        file_path=fp,
                        stage="prep",
                        error_type=type(e).__name__,
                        message=str(e),
                        meta={"duration_ms": int((time.time() - t_file) * 1000)},
                    )
                    _append_metrics_ndjson(log_dir, session_name, {
                        "file": fp,
                        "status": "error",
                        "stage": "prep",
                        "error": str(e),
                        "duration_ms": int((time.time() - t_file) * 1000),
                    })
                    state.upsert(f"{session_name}::{fp}", {
                        "type": "file", "session": session_name, "file": fp,
                        "status": "error", "stage": "prep", "error": str(e),
                    })
                    print(f"[beta_worker/opensearch] FAILED (prep) {done_count}/{total_files} -> {fp}: {e}", flush=True)
                    break

                status = res.get("status")
                parse_ms = res.get("parse_ms")
                chunk_ms = res.get("chunk_ms")

                if status in ("empty", "zero_chunks"):
                    upsert_session_file_status(session_name, fp, "complete")
                    successes.append(fp)
                    _append_log_line(log_dir, session_name, fp, success=True)
                    _append_metrics_ndjson(log_dir, session_name, {
                        "file": fp,
                        "status": "ok",
                        "chunks": 0,
                        "parse_ms": parse_ms,
                        "chunk_ms": chunk_ms,
                        "duration_ms": int((time.time() - t_file) * 1000),
                    })
                    state.upsert(f"{session_name}::{fp}", {
                        "type": "file", "session": session_name, "file": fp,
                        "status": "complete", "chunks": 0,
                        "parse_ms": parse_ms, "chunk_ms": chunk_ms,
                    })
                    break

                if status == "error":
                    upsert_session_file_status(session_name, fp, "error")
                    failures.append(fp)
                    _append_log_line(log_dir, session_name, fp, success=False)
                    _append_failed_reingest_entry(
                        log_dir=log_dir,
                        session_name=session_name,
                        file_path=fp,
                        stage="prep",
                        error_type="PrepError",
                        message=str(res.get("error") or ""),
                        meta={
                            "parse_ms": parse_ms,
                            "chunk_ms": chunk_ms,
                            "duration_ms": int((time.time() - t_file) * 1000),
                        },
                    )
                    _append_metrics_ndjson(log_dir, session_name, {
                        "file": fp,
                        "status": "error",
                        "stage": "prep",
                        "error": str(res.get("error") or ""),
                        "parse_ms": parse_ms,
                        "chunk_ms": chunk_ms,
                        "duration_ms": int((time.time() - t_file) * 1000),
                    })
                    state.upsert(f"{session_name}::{fp}", {
                        "type": "file", "session": session_name, "file": fp,
                        "status": "error", "stage": "prep", "error": str(res.get("error") or ""),
                        "parse_ms": parse_ms, "chunk_ms": chunk_ms,
                    })
                    break

                file_chunks = res.get("chunks") or []
                fmt = res.get("format") or Path(fp).suffix.lower().strip(".")
                try:
                    inserted, embed_ms, index_ms = _embed_and_index_file_chunks_opensearch(
                        embedder=embedder,
                        source_path=fp,
                        fmt=fmt,
                        file_chunks=file_chunks,
                        batch_size=batch_size,
                        insert_retries=insert_retries,
                        governor=governor,
                        session_name=session_name,
                    )

                    processed_chunks += inserted
                    upsert_session_file_status(session_name, fp, "complete")
                    update_session_progress(session_name, last_file=fp, last_chunk=inserted - 1, processed_chunks=processed_chunks)
                    successes.append(fp)
                    _append_log_line(log_dir, session_name, fp, success=True)
                    _append_metrics_ndjson(log_dir, session_name, {
                        "file": fp,
                        "status": "ok",
                        "chunks": len(file_chunks),
                        "parse_ms": parse_ms,
                        "chunk_ms": chunk_ms,
                        "embed_ms": embed_ms,
                        "index_ms": index_ms,
                        "duration_ms": int((time.time() - t_file) * 1000),
                    })
                    state.upsert(f"{session_name}::{fp}", {
                        "type": "file", "session": session_name, "file": fp,
                        "status": "complete", "chunks": len(file_chunks),
                        "parse_ms": parse_ms, "chunk_ms": chunk_ms,
                        "embed_ms": embed_ms, "index_ms": index_ms,
                    })
                    if done_count % 10 == 0:
                        print(f"[beta_worker/opensearch] {session_name}: {done_count}/{total_files} processed, chunks={processed_chunks}", flush=True)
                except Exception as e:
                    try:
                        upsert_session_file_status(session_name, fp, "error")
                    except Exception:
                        pass
                    failures.append(fp)
                    _append_log_line(log_dir, session_name, fp, success=False)
                    _append_failed_reingest_entry(
                        log_dir=log_dir,
                        session_name=session_name,
                        file_path=fp,
                        stage="opensearch",
                        error_type=type(e).__name__,
                        message=str(e),
                        meta={
                            "parse_ms": parse_ms,
                            "chunk_ms": chunk_ms,
                            "embed_ms": embed_ms,
                            "index_ms": index_ms,
                            "duration_ms": int((time.time() - t_file) * 1000),
                        },
                    )
                    _append_metrics_ndjson(log_dir, session_name, {
                        "file": fp,
                        "status": "error",
                        "error": str(e),
                        "parse_ms": parse_ms,
                        "chunk_ms": chunk_ms,
                        "embed_ms": embed_ms,
                        "index_ms": index_ms,
                        "duration_ms": int((time.time() - t_file) * 1000),
                    })
                    state.upsert(f"{session_name}::{fp}", {
                        "type": "file", "session": session_name, "file": fp,
                        "status": "error", "error": str(e),
                        "parse_ms": parse_ms, "chunk_ms": chunk_ms,
                        "embed_ms": embed_ms, "index_ms": index_ms,
                    })
                    print(f"[beta_worker/opensearch] FAILED {done_count}/{total_files} -> {fp}: {e}", flush=True)

                break

            _submit_next()

    paths = _write_logs(log_dir, session_name, successes, failures)
    print(f"[beta_worker/opensearch] Session {session_name} complete. OK={len(successes)} failed={len(failures)}", flush=True)
    print(f"[beta_worker/opensearch] Success log: {paths['success_log']}", flush=True)
    print(f"[beta_worker/opensearch] Error log:   {paths['error_log']}", flush=True)
    state.upsert(f"{session_name}::summary", {
        "type": "session",
        "session": session_name,
        "status": "complete",
        "total_files": total_files,
        "files_ok": len(successes),
        "files_failed": len(failures),
        "total_indexed": processed_chunks,
    })
    complete_session(session_name)


if __name__ == "__main__":
    args = _parse_cli_args(sys.argv[1:])
    if STORAGE_BACKEND == "opensearch":
        run_worker_opensearch(
            session_name=args["session_name"],
            root_dir=args.get("root"),
            partition_file=args.get("partition_file"),
            embedding_model=args.get("model"),
            token_target=args.get("target_tokens") or 512,
            token_overlap=args.get("overlap_tokens") or 64,
            token_max=args.get("max_tokens") or 640,
            log_dir=args.get("log_dir") or "./logs",
            resume=bool(args.get("resume")),
        )
        raise SystemExit(0)
    run_worker(
        session_name=args["session_name"],
        root_dir=args.get("root"),
        partition_file=args.get("partition_file"),
        embedding_model=args.get("model"),
        token_target=args.get("target_tokens") or 512,
        token_overlap=args.get("overlap_tokens") or 64,
        token_max=args.get("max_tokens") or 640,
        log_dir=args.get("log_dir") or "./logs",
        resume=bool(args.get("resume")),
    )
