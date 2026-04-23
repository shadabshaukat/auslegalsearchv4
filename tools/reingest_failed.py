"""
Helper utility to re-ingest only failed files from beta worker OpenSearch logs.

Reads `*.failed.paths.txt` produced by ingest/beta_worker.py and creates one or more
partition files that can be passed to `python -m ingest.beta_worker --partition_file ...`.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple


def _natural_sort_key(s: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s or "")]


def _read_failed_paths(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = (ln or "").strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    # dedupe, keep stable order
    return list(dict.fromkeys(out))


def _latest_failed_paths_file(log_dir: str) -> str:
    p = Path(log_dir)
    cands = sorted(p.glob("*.failed.paths.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No *.failed.paths.txt found under {log_dir}")
    return str(cands[0])


def _split_round_robin(items: List[str], shards: int) -> List[List[str]]:
    buckets: List[List[str]] = [[] for _ in range(max(1, shards))]
    for i, fp in enumerate(items):
        buckets[i % len(buckets)].append(fp)
    return buckets


def _split_balance_by_size(items: List[str], shards: int) -> List[List[str]]:
    def sz(fp: str) -> int:
        try:
            return int(os.path.getsize(fp))
        except Exception:
            return 0

    buckets: List[List[str]] = [[] for _ in range(max(1, shards))]
    sums = [0 for _ in buckets]
    for fp in sorted(items, key=sz, reverse=True):
        idx = min(range(len(buckets)), key=lambda i: sums[i])
        buckets[idx].append(fp)
        sums[idx] += sz(fp)
    return buckets


def _write_partition(path: str, items: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for fp in items:
            f.write(fp + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Create retry partition files from failed ingest logs")
    ap.add_argument("--logs_dir", default="./logs", help="Directory containing *.failed.paths.txt logs")
    ap.add_argument("--session", default=None, help="Session name to use (reads <session>.failed.paths.txt)")
    ap.add_argument("--failed_file", default=None, help="Explicit failed paths file to use")
    ap.add_argument("--output_dir", default=".", help="Where to write retry partition files")
    ap.add_argument("--output_prefix", default=None, help="Prefix for generated partition files")
    ap.add_argument("--shards", type=int, default=1, help="Number of partition shards to generate")
    ap.add_argument(
        "--balance_by_size",
        action="store_true",
        help="Greedy size-balanced sharding (recommended for mixed file sizes)",
    )
    ap.add_argument("--print_worker_commands", action="store_true", help="Print beta_worker command templates")
    ap.add_argument("--root", default=None, help="Optional root path to include in printed command templates")
    ap.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5", help="Model for printed command templates")
    ap.add_argument("--target_tokens", type=int, default=3000)
    ap.add_argument("--overlap_tokens", type=int, default=250)
    ap.add_argument("--max_tokens", type=int, default=3500)
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.failed_file:
        failed_file = args.failed_file
    elif args.session:
        failed_file = os.path.join(args.logs_dir, f"{args.session}.failed.paths.txt")
    else:
        failed_file = _latest_failed_paths_file(args.logs_dir)

    if not os.path.exists(failed_file):
        raise FileNotFoundError(f"Failed paths file not found: {failed_file}")

    paths = _read_failed_paths(failed_file)
    if not paths:
        print(f"[reingest_failed] No failed paths found in: {failed_file}")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)

    shards = max(1, int(args.shards))
    if args.balance_by_size:
        buckets = _split_balance_by_size(paths, shards)
        strategy = "size-balanced"
    else:
        buckets = _split_round_robin(paths, shards)
        strategy = "round-robin"

    ts = time.strftime("%Y%m%d-%H%M%S")
    prefix = args.output_prefix or f"retry-failed-{ts}"

    written: List[Tuple[str, int]] = []
    for i, bucket in enumerate(buckets):
        out = os.path.join(args.output_dir, f".beta-gpu-partition-{prefix}-shard{i}.txt")
        _write_partition(out, bucket)
        written.append((out, len(bucket)))

    print(f"[reingest_failed] input={failed_file}")
    print(f"[reingest_failed] files={len(paths)} shards={shards} strategy={strategy}")
    for out, cnt in written:
        print(f"[reingest_failed] wrote {out} ({cnt} files)")

    if args.print_worker_commands:
        base_sess = prefix
        for i, (out, cnt) in enumerate(written):
            if cnt == 0:
                continue
            cmd = (
                f"CUDA_VISIBLE_DEVICES={i} python3 -m ingest.beta_worker {base_sess}-gpu{i} "
                f"--partition_file \"{out}\" "
                f"--model \"{args.model}\" "
                f"--target_tokens {args.target_tokens} --overlap_tokens {args.overlap_tokens} --max_tokens {args.max_tokens} "
                f"--log_dir \"{args.logs_dir}\""
            )
            if args.root:
                cmd += f" --root \"{args.root}\""
            print(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
