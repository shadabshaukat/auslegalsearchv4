"""Collect and compare ingestion benchmark run results from log files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


_KV_RE = re.compile(r"^#\s*([a-zA-Z0-9_]+)=(.*)$")


def _parse_header_kv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                continue
            m = _KV_RE.match(s)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def _safe_int(v: Optional[str], default: int = 0) -> int:
    try:
        return int(float(v or default))
    except Exception:
        return default


def _safe_float(v: Optional[str], default: float = 0.0) -> float:
    try:
        return float(v or default)
    except Exception:
        return default


def _session_row(log_dir: Path, session: str) -> Dict[str, object]:
    succ = log_dir / f"{session}.success.log"
    err = log_dir / f"{session}.error.log"
    skv = _parse_header_kv(succ)
    ekv = _parse_header_kv(err)

    duration_sec = _safe_int(skv.get("duration_sec"), 0)
    files_ok = _safe_int(skv.get("files_ok"), 0)
    files_failed = _safe_int(ekv.get("files_failed"), 0)
    total_files = files_ok + files_failed

    files_per_min = (files_ok / (duration_sec / 60.0)) if duration_sec > 0 else 0.0
    fail_rate_pct = ((files_failed / total_files) * 100.0) if total_files > 0 else 0.0

    return {
        "session": session,
        "duration_sec": duration_sec,
        "files_ok": files_ok,
        "files_failed": files_failed,
        "total_files": total_files,
        "files_per_min": round(files_per_min, 2),
        "fail_rate_pct": round(fail_rate_pct, 2),
        "success_log": str(succ),
        "error_log": str(err),
    }


def _print_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "session",
        "duration_sec",
        "files_ok",
        "files_failed",
        "files_per_min",
        "fail_rate_pct",
    ]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))

    def _line(vals: List[str]) -> str:
        return " | ".join(v.ljust(widths[h]) for v, h in zip(vals, headers))

    print(_line(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        vals = [str(r.get(h, "")) for h in headers]
        print(_line(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect benchmark results from orchestrator master logs")
    ap.add_argument("--log-dir", required=True, help="Directory containing <session>.success.log and <session>.error.log")
    ap.add_argument("--session-prefix", default="os-bench", help="Only include sessions starting with this prefix")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of table")
    ap.add_argument("--top", type=int, default=0, help="Show top N by files_per_min (0=all)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log dir not found: {log_dir}")

    sessions: List[str] = []
    for p in sorted(log_dir.glob("*.success.log")):
        name = p.name[:-len(".success.log")]
        if not name.startswith(args.session_prefix):
            continue
        if not (log_dir / f"{name}.error.log").exists():
            continue
        sessions.append(name)

    rows = [_session_row(log_dir, s) for s in sessions]
    rows.sort(key=lambda r: _safe_float(str(r.get("files_per_min", 0.0))), reverse=True)
    if args.top and args.top > 0:
        rows = rows[: args.top]

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
