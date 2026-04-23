"""Aggregate ingestion telemetry from worker metrics/error ndjson files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List


def _iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def _safe_num(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize ingest telemetry (metrics/errors ndjson)")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--session-prefix", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log dir not found: {log_dir}")

    metrics_files = sorted(log_dir.glob("*.metrics.ndjson"))
    error_files = sorted(log_dir.glob("*.errors.ndjson"))

    if args.session_prefix:
        metrics_files = [p for p in metrics_files if p.name.startswith(args.session_prefix)]
        error_files = [p for p in error_files if p.name.startswith(args.session_prefix)]

    files_total = 0
    files_complete = 0
    files_error = 0
    chunks_total = 0
    parse_ms: List[float] = []
    chunk_ms: List[float] = []
    embed_ms: List[float] = []
    insert_ms: List[float] = []
    duration_ms: List[float] = []

    for mf in metrics_files:
        for rec in _iter_ndjson(mf):
            if rec.get("type") != "file":
                continue
            files_total += 1
            status = str(rec.get("status") or "").lower()
            if status == "complete":
                files_complete += 1
            elif status == "error":
                files_error += 1
            chunks_total += int(_safe_num(rec.get("chunks"), 0))
            if rec.get("parse_ms") is not None:
                parse_ms.append(_safe_num(rec.get("parse_ms")))
            if rec.get("chunk_ms") is not None:
                chunk_ms.append(_safe_num(rec.get("chunk_ms")))
            if rec.get("embed_ms") is not None:
                embed_ms.append(_safe_num(rec.get("embed_ms")))
            idx = rec.get("index_ms", rec.get("insert_ms"))
            if idx is not None:
                insert_ms.append(_safe_num(idx))
            if rec.get("duration_ms") is not None:
                duration_ms.append(_safe_num(rec.get("duration_ms")))

    err_stage = Counter()
    err_type = Counter()
    retry_like = 0
    for ef in error_files:
        for rec in _iter_ndjson(ef):
            stage = str(rec.get("stage") or "unknown")
            err_stage[stage] += 1
            err = str(rec.get("error") or "")
            et = err.split(":", 1)[0].strip() if err else "unknown"
            err_type[et] += 1
            low = err.lower()
            if "retry" in low or "timeout" in low or "429" in low:
                retry_like += 1

    avg_duration_ms = mean(duration_ms) if duration_ms else 0.0
    files_per_min = (files_complete / (sum(duration_ms) / 60000.0)) if duration_ms and sum(duration_ms) > 0 else 0.0
    chunks_per_min = (chunks_total / (sum(duration_ms) / 60000.0)) if duration_ms and sum(duration_ms) > 0 else 0.0

    out: Dict[str, object] = {
        "metrics_files": len(metrics_files),
        "error_files": len(error_files),
        "files_total": files_total,
        "files_complete": files_complete,
        "files_error": files_error,
        "chunks_total": chunks_total,
        "files_per_min": round(files_per_min, 2),
        "chunks_per_min": round(chunks_per_min, 2),
        "avg_file_duration_ms": round(avg_duration_ms, 2),
        "avg_parse_ms": round(mean(parse_ms), 2) if parse_ms else 0.0,
        "avg_chunk_ms": round(mean(chunk_ms), 2) if chunk_ms else 0.0,
        "avg_embed_ms": round(mean(embed_ms), 2) if embed_ms else 0.0,
        "avg_insert_ms": round(mean(insert_ms), 2) if insert_ms else 0.0,
        "retry_like_errors": retry_like,
        "error_stage_counts": dict(err_stage),
        "top_error_types": err_type.most_common(10),
    }

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"metrics_files={out['metrics_files']} error_files={out['error_files']}")
    print(f"files_total={out['files_total']} complete={out['files_complete']} error={out['files_error']}")
    print(f"chunks_total={out['chunks_total']} files_per_min={out['files_per_min']} chunks_per_min={out['chunks_per_min']}")
    print(
        "avg_ms parse={avg_parse_ms} chunk={avg_chunk_ms} embed={avg_embed_ms} insert={avg_insert_ms} file={avg_file_duration_ms}".format(
            **out
        )
    )
    print(f"retry_like_errors={out['retry_like_errors']}")
    print(f"error_stage_counts={out['error_stage_counts']}")
    print(f"top_error_types={out['top_error_types']}")


if __name__ == "__main__":
    main()
