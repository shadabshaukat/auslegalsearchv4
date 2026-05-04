"""
Production orchestrator wrapper.

Reuses all beta orchestrator logic (partitioning, dynamic scheduling, resume,
OpenSearch tuning, log aggregation) but launches `ingest.production_worker`
instead of `ingest.beta_worker`.
"""

from __future__ import annotations

import os
import subprocess
import sys

from ingest import beta_orchestrator as _beta_orch


def launch_worker(
    child_session,
    root_dir,
    partition_file,
    embedding_model,
    target_tokens,
    overlap_tokens,
    max_tokens,
    log_dir,
    gpu_index=None,
    python_exec=None,
    resume=False,
):
    python_exec = python_exec or os.environ.get("PYTHON_EXEC", sys.executable)
    cmd = [
        python_exec,
        "-m",
        "ingest.production_worker",
        child_session,
        "--root",
        root_dir,
        "--partition_file",
        partition_file,
        "--target_tokens",
        str(int(target_tokens)),
        "--overlap_tokens",
        str(int(overlap_tokens)),
        "--max_tokens",
        str(int(max_tokens)),
        "--log_dir",
        log_dir,
    ]
    if embedding_model:
        cmd.extend(["--model", embedding_model])
    if resume:
        cmd.append("--resume")

    env = dict(os.environ)
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return subprocess.Popen(cmd, env=env)


_beta_orch.launch_worker = launch_worker


def _parse_cli_args(argv):
    return _beta_orch._parse_cli_args(argv)


if __name__ == "__main__":
    import json

    args = _parse_cli_args(sys.argv[1:])
    summary = _beta_orch.orchestrate(
        root_dir=args["root"],
        session_name=args["session"],
        embedding_model=args.get("model"),
        num_gpus=args.get("gpus"),
        sample_per_folder=bool(args.get("sample_per_folder")),
        skip_year_dirs_in_sample=not bool(args.get("no_skip_years_in_sample")),
        target_tokens=args.get("target_tokens") or 512,
        overlap_tokens=args.get("overlap_tokens") or 64,
        max_tokens=args.get("max_tokens") or 640,
        log_dir=args.get("log_dir") or "./logs",
        shards=args.get("shards") or 0,
        balance_by_size=bool(args.get("balance_by_size")),
        resume=bool(args.get("resume")),
        wait=not bool(args.get("no_wait")),
    )
    print(json.dumps(summary, indent=2))
