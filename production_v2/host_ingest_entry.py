from __future__ import annotations

import argparse
import json
import time

from production_v2.ingest_v2 import run_ingestion


def main() -> int:
    ap = argparse.ArgumentParser(description="Host-native v2 ingestion entrypoint")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--limit-files", default="")
    ap.add_argument("--include-html", default="1")
    args = ap.parse_args()

    limit = None
    if str(args.limit_files).strip() != "":
        try:
            limit = int(args.limit_files)
        except Exception:
            limit = None

    include_html = str(args.include_html).strip().lower() in {"1", "true", "yes", "on"}

    def _progress(p):
        print(json.dumps({"job_id": args.job_id, "ts": time.time(), "progress": p}, ensure_ascii=False), flush=True)

    try:
        result = run_ingestion(
            root_dir=args.root_dir,
            limit_files=limit,
            include_html=include_html,
            progress_cb=_progress,
            should_stop_cb=lambda: False,
        )
        print(json.dumps({"job_id": args.job_id, "result": result}, ensure_ascii=False), flush=True)
        return 0
    except Exception as e:
        print(json.dumps({"job_id": args.job_id, "error": str(e)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
