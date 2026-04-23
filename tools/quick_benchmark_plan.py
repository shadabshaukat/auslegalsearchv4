"""Generate a quick OpenSearch ingestion benchmark command matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict


def build_block(
    session: str,
    root: str,
    log_dir: str,
    model: str,
    gpus: int,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
    envs: Dict[str, str],
) -> str:
    export_lines = [f"export {k}={v}" for k, v in envs.items()]
    cmd_lines = [
        "python3 -m ingest.beta_orchestrator \\",
        f"  --root \"{root}\" \\",
        f"  --session \"{session}\" \\",
        f"  --gpus {int(gpus)} \\",
        f"  --model \"{model}\" \\",
        f"  --target_tokens {int(target_tokens)} --overlap_tokens {int(overlap_tokens)} --max_tokens {int(max_tokens)} \\",
        "  --resume \\",
        f"  --log_dir \"{log_dir}\"",
    ]
    return "\n".join(export_lines + [""] + cmd_lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Print benchmark plan commands")
    ap.add_argument("--root", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--session-prefix", default="os-bench")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--target_tokens", type=int, default=3000)
    ap.add_argument("--overlap_tokens", type=int, default=250)
    ap.add_argument("--max_tokens", type=int, default=3500)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    common = {
        "AUSLEGALSEARCH_STORAGE_BACKEND": "opensearch",
        "OPENSEARCH_TUNE_INDEX": "1",
        "OPENSEARCH_ENFORCE_SHARDS": "0",
        "AUSLEGALSEARCH_MAX_THROUGHPUT_MODE": "1",
        "AUSLEGALSEARCH_LOG_METRICS": "0",
        "OS_METRICS_NDJSON": "0",
        "OS_INGEST_STATE_ENABLE": "0",
        "AUSLEGALSEARCH_CPU_WORKERS": "6",
        "AUSLEGALSEARCH_PIPELINE_PREFETCH": "96",
        "OPENSEARCH_BULK_CHUNK_SIZE": "1000",
        "OPENSEARCH_BULK_MAX_BYTES": "104857600",
        "OPENSEARCH_BULK_QUEUE_SIZE": "16",
    }

    scenarios = [
        ("baseline", {**common, "AUSLEGALSEARCH_EMBED_BATCH": "64", "OPENSEARCH_BULK_CONCURRENCY": "4", "AUSLEGALSEARCH_OS_STREAM_CHUNK_FLUSH_SIZE": "1200"}),
        ("embed96", {**common, "AUSLEGALSEARCH_EMBED_BATCH": "96", "OPENSEARCH_BULK_CONCURRENCY": "4", "AUSLEGALSEARCH_OS_STREAM_CHUNK_FLUSH_SIZE": "1200"}),
        ("bulk6", {**common, "AUSLEGALSEARCH_EMBED_BATCH": "96", "OPENSEARCH_BULK_CONCURRENCY": "6", "AUSLEGALSEARCH_OS_STREAM_CHUNK_FLUSH_SIZE": "1200"}),
        ("flush800", {**common, "AUSLEGALSEARCH_EMBED_BATCH": "96", "OPENSEARCH_BULK_CONCURRENCY": "4", "AUSLEGALSEARCH_OS_STREAM_CHUNK_FLUSH_SIZE": "800"}),
    ]

    print("# Quick benchmark plan (run one scenario at a time)")
    print("# Suggested run duration per scenario: 10-15 minutes")

    for label, envs in scenarios:
        session = f"{args.session_prefix}-{label}-{stamp}"
        print("\n" + "=" * 80)
        print(f"# Scenario: {label}")
        print(build_block(
            session=session,
            root=args.root,
            log_dir=args.log_dir,
            model=args.model,
            gpus=args.gpus,
            target_tokens=args.target_tokens,
            overlap_tokens=args.overlap_tokens,
            max_tokens=args.max_tokens,
            envs=envs,
        ))


if __name__ == "__main__":
    main()
