"""Production v2 FastAPI app (parallel to existing v1 app)."""

from __future__ import annotations

import os
import json
import secrets
import shlex
import signal
import subprocess
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from production_v2.config import settings, loaded_env_file, validate_v2_runtime
from production_v2.dsl_templates import SCENARIOS
from production_v2.ingest_v2 import run_ingestion
from production_v2.opensearch_v2 import ensure_indexes, recreate_indexes
from production_v2.search_v2 import run_search


app = FastAPI(
    title="AUSLegalSearch Production v2 API",
    description="Parallel v2 API for new 3-index OpenSearch architecture and routed DSL/hybrid search.",
    version="2.0.0",
)

_INGEST_JOBS: Dict[str, Dict[str, Any]] = {}
_INGEST_LOCK = threading.Lock()
_INGEST_OFFLOAD_PROCS: Dict[str, Dict[str, Any]] = {}


def _set_job(job_id: str, payload: Dict[str, Any]) -> None:
    with _INGEST_LOCK:
        cur = dict(_INGEST_JOBS.get(job_id) or {})
        cur.update(payload)
        _INGEST_JOBS[job_id] = cur


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _INGEST_LOCK:
        j = _INGEST_JOBS.get(job_id)
        return dict(j) if j else None


def _build_offload_cmd(template: str, job_id: str, req: V2IngestReq) -> str:
    limit_raw = "" if req.limit_files is None else str(int(req.limit_files))
    include_html_raw = "1" if bool(req.include_html) else "0"
    values = {
        "job_id": str(job_id),
        "root_dir": str(req.root_dir),
        "limit_files": limit_raw,
        "include_html": include_html_raw,
        "job_id_q": shlex.quote(str(job_id)),
        "root_dir_q": shlex.quote(str(req.root_dir)),
        "limit_files_q": shlex.quote(limit_raw),
    }
    # Be resilient to env/shell quoting behavior by applying explicit token
    # replacement instead of relying solely on str.format().
    rendered = str(template or "")
    for k, v in values.items():
        rendered = rendered.replace("{" + k + "}", str(v))
        rendered = rendered.replace("{{" + k + "}}", str(v))
    return rendered


def _map_root_dir_for_offload(root_dir: str) -> str:
    """
    Translate container-visible ingest path into host path for offloaded runs.
    - exact /app/data -> configured host ingest dir
    - /app/data/sub/path -> <host_ingest_dir>/sub/path
    """
    req_root = str(root_dir or "").strip()
    if not req_root:
        return req_root

    host_root = str(settings.host_ingest_dir or "").strip()
    container_root = str(settings.container_ingest_dir or "/app/data").strip()
    if not host_root or not container_root:
        return req_root

    if req_root == container_root:
        return host_root
    if req_root.startswith(container_root.rstrip("/") + "/"):
        suffix = req_root[len(container_root.rstrip("/")) :]
        return host_root.rstrip("/") + suffix
    return req_root


def _refresh_offload_job_state(job_id: str) -> None:
    meta = _INGEST_OFFLOAD_PROCS.get(job_id)
    if not meta:
        return
    proc = meta.get("proc")
    if proc is None:
        return

    # Stream host-side progress/result updates into in-memory job state so
    # Gradio polling shows live progress even when ingestion executes off-container.
    try:
        log_path = str(meta.get("log_path") or "").strip()
        if log_path and os.path.exists(log_path):
            start_pos = int(meta.get("log_pos") or 0)
            with open(log_path, "r", encoding="utf-8") as lf:
                lf.seek(max(0, start_pos))
                for raw in lf:
                    line = (raw or "").strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    if isinstance(rec, dict) and isinstance(rec.get("progress"), dict):
                        _set_job(
                            job_id,
                            {
                                "progress": rec.get("progress"),
                                "updated_at": time.time(),
                            },
                        )
                    if isinstance(rec, dict) and isinstance(rec.get("result"), dict):
                        _set_job(
                            job_id,
                            {
                                "result": rec.get("result"),
                                "progress": {
                                    "phase": "index",
                                    "total_files": rec.get("result", {}).get("total_files"),
                                    "ok_files": rec.get("result", {}).get("ok_files"),
                                    "failed_files": rec.get("result", {}).get("failed_files"),
                                    "indexed_chunks": rec.get("result", {}).get("indexed_chunks"),
                                    "citation_edges": rec.get("result", {}).get("citation_edges"),
                                },
                                "updated_at": time.time(),
                            },
                        )
                    if isinstance(rec, dict) and rec.get("error"):
                        _set_job(
                            job_id,
                            {
                                "offload_error": str(rec.get("error")),
                                "updated_at": time.time(),
                            },
                        )
                meta["log_pos"] = int(lf.tell())
    except Exception:
        pass

    rc = proc.poll()
    if rc is None:
        return

    job = _get_job(job_id) or {}
    status_val = str(job.get("status") or "").lower()
    if status_val in {"completed", "failed", "stopped"}:
        return

    if bool(job.get("cancel_requested", False)):
        _set_job(
            job_id,
            {
                "status": "stopped",
                "finished_at": time.time(),
                "updated_at": time.time(),
                "offload_returncode": int(rc),
            },
        )
        return

    if int(rc) == 0:
        _set_job(
            job_id,
            {
                "status": "completed",
                "finished_at": time.time(),
                "updated_at": time.time(),
                "offload_returncode": 0,
            },
        )
    else:
        _set_job(
            job_id,
            {
                "status": "failed",
                "finished_at": time.time(),
                "updated_at": time.time(),
                "offload_returncode": int(rc),
                "error": f"Offload command exited with code {rc}",
            },
        )

    try:
        fh = meta.get("log_fh")
        if fh:
            fh.flush()
            fh.close()
            meta["log_fh"] = None
    except Exception:
        pass


@app.on_event("startup")
def _startup_validate_runtime() -> None:
    validate_v2_runtime()

security = HTTPBasic()


def _auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    ok_user = secrets.compare_digest(credentials.username, settings.api_user)
    ok_pass = secrets.compare_digest(credentials.password, settings.api_pass)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class V2IngestReq(BaseModel):
    root_dir: str = Field(..., description="Root directory to ingest from scratch")
    limit_files: Optional[int] = Field(default=None, ge=1)
    include_html: bool = True


class V2SearchReq(BaseModel):
    query: str
    scenario: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    use_hybrid: bool = True
    use_reranker: Optional[bool] = None
    rerank_top_n: Optional[int] = Field(default=None, ge=1, le=200)
    filters: Optional[Dict[str, Any]] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "app": "production_v2"}


@app.get("/v2/config/effective", tags=["v2"])
def effective_config(_: str = Depends(_auth)):
    return {
        "loaded_env_file": loaded_env_file(),
        "opensearch_host": settings.os_host,
        "opensearch_user": settings.os_user,
        "opensearch_verify_certs": settings.os_verify_certs,
        "indexes": {
            "authorities": settings.index_authorities,
            "chunks_lex": settings.index_chunks_lex,
            "chunks_vec": settings.index_chunks_vec,
            "citation_graph": settings.index_citation_graph,
        },
        "reranker": {
            "enabled_default": settings.reranker_enable_default,
            "model": settings.reranker_model,
            "top_n": settings.reranker_top_n,
        },
    }


@app.get("/v2/scenarios", tags=["v2"])
def list_scenarios(_: str = Depends(_auth)):
    return SCENARIOS


@app.post("/v2/indexes/bootstrap", tags=["v2"])
def bootstrap_indexes(_: str = Depends(_auth)):
    return {"status": "ok", "indexes": ensure_indexes()}


@app.post("/v2/indexes/recreate", tags=["v2"])
def reset_and_recreate_indexes(_: str = Depends(_auth)):
    return {"status": "ok", "indexes": recreate_indexes()}


@app.post("/v2/ingest/run", tags=["v2"])
def ingest_run(req: V2IngestReq, _: str = Depends(_auth)):
    return run_ingestion(
        root_dir=req.root_dir,
        limit_files=req.limit_files,
        include_html=req.include_html,
    )


@app.post("/v2/ingest/start", tags=["v2"])
def ingest_start(req: V2IngestReq, _: str = Depends(_auth)):
    mapped_root = _map_root_dir_for_offload(req.root_dir) if bool(settings.ingest_offload_enable) else req.root_dir
    offload_req = V2IngestReq(root_dir=mapped_root, limit_files=req.limit_files, include_html=req.include_html)

    job_id = str(uuid.uuid4())
    now = time.time()
    _set_job(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "request": offload_req.model_dump(),
            "request_original": req.model_dump(),
        },
    )

    if bool(settings.ingest_offload_enable):
        cmd_template = str(settings.ingest_offload_start_cmd or "").strip()
        if not cmd_template:
            raise HTTPException(
                status_code=500,
                detail="V2_INGEST_OFFLOAD_ENABLE=1 but V2_INGEST_OFFLOAD_START_CMD is empty",
            )

        cmd = _build_offload_cmd(cmd_template, job_id, offload_req)
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        offload_log = os.path.join(logs_dir, f"v2-ingest-offload-{job_id}.log")
        log_fh = open(offload_log, "a", encoding="utf-8")

        cwd = str(settings.ingest_offload_workdir or "").strip() or None
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        _INGEST_OFFLOAD_PROCS[job_id] = {
            "proc": proc,
            "log_fh": log_fh,
            "log_path": offload_log,
            "cmd": cmd,
        }
        _set_job(
            job_id,
            {
                "status": "running",
                "started_at": time.time(),
                "updated_at": time.time(),
                "mode": "offload",
                "offload": {
                    "pid": int(proc.pid),
                    "log_path": offload_log,
                    "cwd": cwd,
                    "command": cmd,
                },
                "progress": {"phase": "offload_submitted", "total_files": None},
            },
        )
        return {"job_id": job_id, "status": "queued"}

    def _runner() -> None:
        _set_job(job_id, {"status": "running", "started_at": time.time(), "updated_at": time.time()})

        def _on_progress(p: Dict[str, Any]) -> None:
            _set_job(
                job_id,
                {
                    "progress": p,
                    "updated_at": time.time(),
                },
            )

        def _should_stop() -> bool:
            j = _get_job(job_id) or {}
            return bool(j.get("cancel_requested", False))

        try:
            result = run_ingestion(
                root_dir=offload_req.root_dir,
                limit_files=offload_req.limit_files,
                include_html=offload_req.include_html,
                progress_cb=_on_progress,
                should_stop_cb=_should_stop,
            )
            if _should_stop():
                _set_job(
                    job_id,
                    {
                        "status": "stopped",
                        "finished_at": time.time(),
                        "updated_at": time.time(),
                        "result": result,
                    },
                )
                return
            _set_job(
                job_id,
                {
                    "status": "completed",
                    "finished_at": time.time(),
                    "updated_at": time.time(),
                    "result": result,
                },
            )
        except Exception as e:
            jnow = _get_job(job_id) or {}
            if bool(jnow.get("cancel_requested", False)):
                _set_job(
                    job_id,
                    {
                        "status": "stopped",
                        "finished_at": time.time(),
                        "updated_at": time.time(),
                        "error": str(e),
                    },
                )
                return
            _set_job(
                job_id,
                {
                    "status": "failed",
                    "finished_at": time.time(),
                    "updated_at": time.time(),
                    "error": str(e),
                    "traceback": traceback.format_exc()[-4000:],
                },
            )

    threading.Thread(target=_runner, daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/v2/ingest/status/{job_id}", tags=["v2"])
def ingest_status(job_id: str, _: str = Depends(_auth)):
    _refresh_offload_job_state(job_id)
    j = _get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job_id not found")
    return j


@app.post("/v2/ingest/stop/{job_id}", tags=["v2"])
def ingest_stop(job_id: str, _: str = Depends(_auth)):
    _refresh_offload_job_state(job_id)
    j = _get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job_id not found")
    status_val = str(j.get("status") or "").lower()
    if status_val in {"completed", "failed", "stopped"}:
        return {"job_id": job_id, "status": status_val, "message": "job already finalized"}
    _set_job(job_id, {"cancel_requested": True, "updated_at": time.time(), "status": "stop_requested"})

    if str(j.get("mode") or "") == "offload":
        meta = _INGEST_OFFLOAD_PROCS.get(job_id) or {}
        proc = meta.get("proc")
        stop_template = str(settings.ingest_offload_stop_cmd or "").strip()
        if stop_template:
            req_dict = j.get("request") or {}
            req = V2IngestReq(
                root_dir=str(req_dict.get("root_dir") or ""),
                limit_files=req_dict.get("limit_files"),
                include_html=bool(req_dict.get("include_html", True)),
            )
            stop_cmd = _build_offload_cmd(stop_template, job_id, req)
            subprocess.Popen(stop_cmd, shell=True, start_new_session=True)
        elif proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

    return {"job_id": job_id, "status": "stop_requested"}


@app.get("/v2/ingest/jobs", tags=["v2"])
def ingest_jobs(_: str = Depends(_auth), limit: int = 20):
    with _INGEST_LOCK:
        jobs = list(_INGEST_JOBS.values())
    jobs = sorted(jobs, key=lambda x: float(x.get("updated_at") or 0), reverse=True)
    lim = max(1, min(int(limit), 200))
    return jobs[:lim]


@app.get("/v2/ingest/jobs/latest", tags=["v2"])
def ingest_latest_job(_: str = Depends(_auth)):
    with _INGEST_LOCK:
        jobs = list(_INGEST_JOBS.values())
    if not jobs:
        return {"status": "none"}
    jobs = sorted(jobs, key=lambda x: float(x.get("updated_at") or 0), reverse=True)
    return jobs[0]


@app.post("/v2/search", tags=["v2"])
def search_v2(req: V2SearchReq, _: str = Depends(_auth)):
    return run_search(
        query=req.query,
        scenario=req.scenario,
        top_k=req.top_k,
        filters=req.filters,
        use_hybrid=req.use_hybrid,
        use_reranker=req.use_reranker,
        rerank_top_n=req.rerank_top_n,
    )
