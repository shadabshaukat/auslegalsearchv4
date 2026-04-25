"""Production v2 Gradio app for routed DSL and hybrid legal search scenarios."""

from __future__ import annotations

import html
import json
import time
from typing import Any, Dict, List, Tuple

import gradio as gr
import requests

from production_v2.config import settings

API_ROOT = settings.gradio_api_url
AUTH = (settings.api_user, settings.api_pass)


def _api_get(path: str):
    return requests.get(f"{API_ROOT}{path}", auth=AUTH, timeout=30)


def _api_post(path: str, payload: Dict[str, Any]):
    return requests.post(f"{API_ROOT}{path}", json=payload, auth=AUTH, timeout=120)


def load_scenarios() -> List[Tuple[str, str]]:
    try:
        r = _api_get("/v2/scenarios")
        if not r.ok:
            return [("Keyword", "keyword")]
        rows = r.json()
        out: List[Tuple[str, str]] = []
        for row in rows:
            out.append((row.get("label", row.get("id")), row.get("id")))
        return out or [("Keyword", "keyword")]
    except Exception:
        return [("Keyword", "keyword")]


def bootstrap_indexes() -> str:
    try:
        r = _api_post("/v2/indexes/bootstrap", {})
        data = r.json() if r.ok else {"error": r.text}
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"bootstrap error: {e}"


def recreate_indexes() -> str:
    try:
        r = _api_post("/v2/indexes/recreate", {})
        data = r.json() if r.ok else {"error": r.text}
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"recreate error: {e}"


def _render_ingest_status(data: Dict[str, Any]) -> str:
    status = str(data.get("status") or "unknown").lower()
    progress = data.get("progress") or {}
    total = int(progress.get("total_files") or 0)
    done = int(progress.get("files_completed") or progress.get("prepared_files") or 0)
    okf = int(progress.get("ok_files") or 0)
    failf = int(progress.get("failed_files") or 0)
    chunks = int(progress.get("indexed_chunks") or 0)
    edges = int(progress.get("citation_edges") or 0)
    phase = str(progress.get("phase") or "")
    pct = int((done / total) * 100) if total > 0 else 0

    badge = {
        "queued": "🟡 QUEUED",
        "running": "🔵 RUNNING",
        "stop_requested": "🟠 STOP REQUESTED",
        "stopped": "🟠 STOPPED",
        "completed": "🟢 COMPLETED",
        "failed": "🔴 FAILED",
        "none": "⚪ NONE",
    }.get(status, status.upper())

    return (
        f"### Ingestion Job Status: {badge}\n"
        f"- Job ID: `{data.get('job_id', '-')}`\n"
        f"- Phase: `{phase or '-'}`\n"
        f"- Progress: **{done}/{total} ({pct}%)**\n"
        f"- Files: ok={okf}, failed={failf}\n"
        f"- Indexed chunks: {chunks}\n"
        f"- Citation edges: {edges}\n"
    )


def ingest_start(root_dir: str, limit_files: int, include_html: bool) -> Tuple[str, str, str]:
    payload = {
        "root_dir": root_dir,
        "limit_files": None if limit_files <= 0 else int(limit_files),
        "include_html": bool(include_html),
    }
    try:
        r = _api_post("/v2/ingest/start", payload)
        data = r.json() if r.ok else {"error": r.text}
        if "error" in data:
            return "", "### Ingestion Job Status: 🔴 FAILED\n- Could not start job", json.dumps(data, indent=2)
        job_id = str(data.get("job_id") or "")
        status_doc = {"job_id": job_id, "status": data.get("status", "queued")}
        return job_id, _render_ingest_status(status_doc), json.dumps(status_doc, indent=2)
    except Exception as e:
        err = {"error": str(e)}
        return "", "### Ingestion Job Status: 🔴 FAILED\n- Exception while starting job", json.dumps(err, indent=2)


def ingest_poll(job_id: str) -> Tuple[str, str]:
    jid = (job_id or "").strip()
    if not jid:
        return "### Ingestion Job Status: ⚪ NONE\n- No active job", ""
    try:
        r = _api_get(f"/v2/ingest/status/{jid}")
        data = r.json() if r.ok else {"job_id": jid, "status": "failed", "error": r.text}
        return _render_ingest_status(data), json.dumps(data, indent=2)
    except Exception as e:
        err = {"job_id": jid, "status": "failed", "error": str(e)}
        return _render_ingest_status(err), json.dumps(err, indent=2)


def ingest_resume_latest() -> Tuple[str, str, str]:
    try:
        r = _api_get("/v2/ingest/jobs/latest")
        data = r.json() if r.ok else {"status": "none", "error": r.text}
        job_id = str(data.get("job_id") or "")
        if not job_id:
            return "", "### Ingestion Job Status: ⚪ NONE\n- No recent job found", json.dumps(data, indent=2)
        return job_id, _render_ingest_status(data), json.dumps(data, indent=2)
    except Exception as e:
        err = {"status": "failed", "error": str(e)}
        return "", "### Ingestion Job Status: 🔴 FAILED\n- Could not resume latest job", json.dumps(err, indent=2)


def ingest_stop(job_id: str) -> Tuple[str, str]:
    jid = (job_id or "").strip()
    if not jid:
        return "### Ingestion Job Status: ⚪ NONE\n- No active job", ""
    try:
        r = _api_post(f"/v2/ingest/stop/{jid}", {})
        data = r.json() if r.ok else {"job_id": jid, "status": "failed", "error": r.text}
        return _render_ingest_status(data), json.dumps(data, indent=2)
    except Exception as e:
        err = {"job_id": jid, "status": "failed", "error": str(e)}
        return _render_ingest_status(err), json.dumps(err, indent=2)


def _result_cards(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "<div>No results</div>"
    cards = []
    for i, h in enumerate(results, 1):
        title = html.escape(str(h.get("title") or "(untitled)"))
        source = html.escape(str(h.get("source") or ""))
        url = str(h.get("url") or "")
        cite_list = h.get("citations") or []
        cite = html.escape(str(cite_list[0] if cite_list else ""))
        snippet = html.escape(str(h.get("text") or h.get("text_preview") or "")[:900])
        score = h.get("rrf_score", h.get("score", 0))
        link = f'<a href="{html.escape(url)}" target="_blank">open</a>' if url else ""
        cards.append(
            f"""
            <div style='border:1px solid #ddd;border-radius:10px;padding:10px;margin:8px 0;background:#fff;'>
              <div style='font-weight:700;margin-bottom:4px;'>{i}. {title}</div>
              <div style='font-size:12px;color:#555;margin-bottom:6px;'>score={score} | citation={cite} {link}</div>
              <div style='font-size:12px;color:#666;margin-bottom:6px;'>{source}</div>
              <div style='font-size:14px;line-height:1.4;white-space:pre-wrap;'>{snippet}</div>
            </div>
            """
        )
    return "\n".join(cards)


def search_run(
    query: str,
    scenario: str,
    top_k: int,
    use_hybrid: bool,
    use_reranker: bool,
    rerank_top_n: int,
    jurisdiction: str,
    database: str,
    court: str,
    date_from: str,
    date_to: str,
) -> Tuple[str, str, str]:
    filters: Dict[str, Any] = {}
    if jurisdiction.strip():
        filters["jurisdiction"] = jurisdiction.strip().lower()
    if database.strip():
        filters["database"] = database.strip().lower()
    if court.strip():
        filters["court"] = court.strip().lower()
    if date_from.strip():
        filters["date_from"] = date_from.strip()
    if date_to.strip():
        filters["date_to"] = date_to.strip()

    payload = {
        "query": query,
        "scenario": scenario,
        "top_k": int(top_k),
        "use_hybrid": bool(use_hybrid),
        "use_reranker": bool(use_reranker),
        "rerank_top_n": int(rerank_top_n),
        "filters": filters,
    }

    try:
        r = _api_post("/v2/search", payload)
        data = r.json() if r.ok else {"error": r.text}
    except Exception as e:
        data = {"error": str(e)}

    if "error" in data:
        return data["error"], "", ""

    dsl = json.dumps(data.get("dsl") or {}, indent=2)
    stats = json.dumps({
        "scenario": data.get("scenario"),
        "counts": data.get("counts"),
        "reranker": data.get("reranker"),
        "citation_graph_edges": len(data.get("citation_graph") or []),
    }, indent=2)
    cards = _result_cards(data.get("results") or [])
    return stats, dsl, cards


def build_ui() -> gr.Blocks:
    scenarios = load_scenarios()
    default_scenario = scenarios[0][1] if scenarios else "keyword"

    with gr.Blocks(title="AUSLegalSearch Production v2") as demo:
        gr.Markdown("## AUSLegalSearch Production v2\nRouted DSL + Hybrid (BM25 + Vector) Search")

        with gr.Tab("Index Bootstrap"):
            boot_btn = gr.Button("Bootstrap v2 Indexes")
            recreate_btn = gr.Button("Delete + Recreate ALL v2 Indexes", variant="stop")
            boot_out = gr.Code(label="Bootstrap Output", language="json")
            boot_btn.click(bootstrap_indexes, inputs=[], outputs=[boot_out])
            recreate_btn.click(recreate_indexes, inputs=[], outputs=[boot_out])

        with gr.Tab("Ingestion (from scratch)"):
            gr.Markdown(
                "**Path must be visible inside runtime/container.** "
                "For Docker default use `/app/data` (mapped from host `./data` or `V2_HOST_INGEST_DIR`)."
            )
            root_dir = gr.Textbox(label="Root Directory", placeholder="/app/data", value="/app/data")
            limit_files = gr.Number(label="Limit Files (0 = no limit)", value=0)
            include_html = gr.Checkbox(label="Include HTML", value=True)
            ingest_btn = gr.Button("Start Async Ingestion Job")
            stop_btn = gr.Button("Stop Current Job", variant="stop")
            resume_btn = gr.Button("Resume Latest Job")
            status_btn = gr.Button("Refresh Job Status")
            ingest_job_id = gr.Textbox(label="Current Job ID", interactive=False)
            ingest_status_md = gr.Markdown("### Ingestion Job Status: ⚪ NONE")
            ingest_out = gr.Code(label="Ingestion Output", language="json")
            ingest_btn.click(ingest_start, inputs=[root_dir, limit_files, include_html], outputs=[ingest_job_id, ingest_status_md, ingest_out])
            stop_btn.click(ingest_stop, inputs=[ingest_job_id], outputs=[ingest_status_md, ingest_out])
            resume_btn.click(ingest_resume_latest, inputs=[], outputs=[ingest_job_id, ingest_status_md, ingest_out])
            status_btn.click(ingest_poll, inputs=[ingest_job_id], outputs=[ingest_status_md, ingest_out])

            poll_interval_s = max(1, int(settings.ingest_status_poll_seconds))
            ingest_timer = gr.Timer(value=float(poll_interval_s))
            ingest_timer.tick(ingest_poll, inputs=[ingest_job_id], outputs=[ingest_status_md, ingest_out])

        demo.load(ingest_resume_latest, inputs=[], outputs=[ingest_job_id, ingest_status_md, ingest_out])

        with gr.Tab("Search Scenarios"):
            query = gr.Textbox(label="Query", lines=2, placeholder="e.g. Fair Work Act 2009 s 351")
            scenario = gr.Dropdown(choices=[v for _, v in scenarios], value=default_scenario, label="Scenario")
            top_k = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Top K")
            use_hybrid = gr.Checkbox(label="Use Hybrid (Lexical + Vector)", value=True)
            use_reranker = gr.Checkbox(label="Use Reranker (accuracy-first)", value=True)
            rerank_top_n = gr.Slider(minimum=5, maximum=200, value=50, step=1, label="Reranker Candidate Pool")

            with gr.Accordion("Metadata Filters", open=False):
                jurisdiction = gr.Textbox(label="Jurisdiction", placeholder="au / cth / nsw / vic")
                database = gr.Textbox(label="Database", placeholder="hca / fca / consol_act / sydlawrw")
                court = gr.Textbox(label="Court", placeholder="high court")
                date_from = gr.Textbox(label="Date From", placeholder="YYYY-MM-DD")
                date_to = gr.Textbox(label="Date To", placeholder="YYYY-MM-DD")

            search_btn = gr.Button("Search")
            stats_out = gr.Code(label="Search Stats", language="json")
            dsl_out = gr.Code(label="Executed DSL", language="json")
            cards_out = gr.HTML(label="Results")

            search_btn.click(
                search_run,
                inputs=[query, scenario, top_k, use_hybrid, use_reranker, rerank_top_n, jurisdiction, database, court, date_from, date_to],
                outputs=[stats_out, dsl_out, cards_out],
            )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name=settings.gradio_host, server_port=int(settings.gradio_port), show_error=True)
