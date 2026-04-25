"""Production v2 FastAPI app (parallel to existing v1 app)."""

from __future__ import annotations

import secrets
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


def _set_job(job_id: str, payload: Dict[str, Any]) -> None:
    with _INGEST_LOCK:
        cur = dict(_INGEST_JOBS.get(job_id) or {})
        cur.update(payload)
        _INGEST_JOBS[job_id] = cur


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _INGEST_LOCK:
        j = _INGEST_JOBS.get(job_id)
        return dict(j) if j else None


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
    job_id = str(uuid.uuid4())
    now = time.time()
    _set_job(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "request": req.model_dump(),
        },
    )

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
                root_dir=req.root_dir,
                limit_files=req.limit_files,
                include_html=req.include_html,
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
    j = _get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job_id not found")
    return j


@app.post("/v2/ingest/stop/{job_id}", tags=["v2"])
def ingest_stop(job_id: str, _: str = Depends(_auth)):
    j = _get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job_id not found")
    status_val = str(j.get("status") or "").lower()
    if status_val in {"completed", "failed", "stopped"}:
        return {"job_id": job_id, "status": status_val, "message": "job already finalized"}
    _set_job(job_id, {"cancel_requested": True, "updated_at": time.time()})
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
