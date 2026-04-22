"""
AUSLegalSearch v4 – Main App Page (ENHANCED: rich metadata chunk display)
- Sidebar batch convert with 'Start' button
- Legal hybrid search, ingestion, RAG, all chunk context returns rich metadata
"""

import streamlit as st
from pathlib import Path
import requests
from ingest import loader
from embedding.embedder import Embedder, DEFAULT_MODEL
from db.store import (
    start_session, update_session_progress, complete_session, fail_session,
    get_active_sessions, get_session, add_document, add_embedding, search_vector, search_bm25, search_hybrid,
    count_session_files
)
from rag.rag_pipeline import RAGPipeline, list_ollama_models
import os
from datetime import datetime
import subprocess
import time
import re

import legal_html2text

st.set_page_config(page_title="AUSLegalSearch v4", layout="wide")
st.title("AUSLegalSearch v4 – Legal Document Search, Background Embedding & RAG")

API_ROOT = os.environ.get("AUSLEGALSEARCH_API_URL", "http://localhost:8000")

def get_num_gpus():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        gpus = [line for line in result.stdout.split("\n") if "GPU" in line]
        return len(gpus)
    except Exception:
        return 1

if "user" not in st.session_state:
    st.warning("You must login to continue.")
    if hasattr(st, "switch_page"):
        st.switch_page("pages/login.py")
    else:
        st.stop()

if "directories" not in st.session_state:
    st.session_state["directories"] = set()
if "session_page_state" not in st.session_state:
    st.session_state["session_page_state"] = None
if "run_ingest" not in st.session_state:
    st.session_state["run_ingest"] = False
if "selected_session" not in st.session_state:
    st.session_state["selected_session"] = None
if "run_session_type" not in st.session_state:
    st.session_state["run_session_type"] = None
if "html2txt_run" not in st.session_state:
    st.session_state["html2txt_run"] = False

LEGAL_SYSTEM_PROMPT = """You are an expert Australian legal research and compliance AI assistant.
Answer strictly from the provided sources and context. Always cite the source section/citation for every statement. If you do not know the answer from the context, reply: "Not found in the provided legal documents."
When summarizing, be neutral and factual. Never invent legal advice."""

EMBEDDING_MODELS = [
    "nomic-ai/nomic-embed-text-v1.5",
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "maastrichtlawtech/bge-legal-en-v1.5",
    "nlpaueb/legal-bert-base-uncased",
]
selected_embedding_model = st.sidebar.selectbox(
    "Embedding Model (for chunk/vectorization)",
    EMBEDDING_MODELS,
    index=0
)
st.sidebar.caption(f"Embedding model in use: `{selected_embedding_model}`")

db_url = os.environ.get("AUSLEGALSEARCH_DB_URL", "")
if db_url:
    db_url_masked = re.sub(r'(://[^:]+:)[^@]+@', r'\1*****@', db_url)
    st.sidebar.markdown(f"**DB URL:** `{db_url_masked}`")
    st.sidebar.code("""
Example DB URL:
postgresql+psycopg2://username:password@host:port/databasename

   |__ protocol/driver    |__ user    |__ pass  |__ host    |__ port |__ db name
postgresql+psycopg2       myuser      secret    localhost   5432     auslegalsearch
    """, language="text")
else:
    db_url = os.environ.get("AUSLEGALSEARCH_DB_HOST", "localhost")
    db_name = os.environ.get("AUSLEGALSEARCH_DB_NAME", "auslegalsearch")
    st.sidebar.markdown(f"**DB Host:** `{db_url}`<br>**DB Name:** `{db_name}`", unsafe_allow_html=True)

model_list = list_ollama_models()
if not model_list:
    st.sidebar.warning("No Ollama models found locally. Is Ollama running?")
    model_list = ["llama3"]
selected_model = st.sidebar.selectbox("RAG LLM model (Ollama, legal-tuned preferred)", model_list, index=0)

st.markdown("### Custom System Prompt for Legal QA")
user_system_prompt = st.text_area(
    "System Prompt",
    value=LEGAL_SYSTEM_PROMPT,
    help="Set a custom prompt for the legal assistant. This guides legal compliance output."
)

# --- TOP LEVEL SIDEBAR ACTION BUTTONS ---
st.sidebar.header("Corpus & Conversion Workflow")
start_load_clicked = st.sidebar.button("🔴 Start Data Load", key="start_new_btn")
convert_files_clicked = st.sidebar.button("🟦 Convert Files", key="convert_files_btn")
st.sidebar.caption("⬆️ Use these actions to load/embed dataset, or convert .html/.pdf to .txt in batch.")

# --- HTML2Text Conversion Controls with Start Button below fields ---
st.sidebar.markdown("---")
st.sidebar.header("HTML & PDF to Text Batch Converter")
html2txt_in_dir = st.sidebar.text_input("Input Directory", value="")
html2txt_out_dir = st.sidebar.text_input("Output Directory", value="")
html2txt_sess = st.sidebar.text_input("Conversion Session Name", value=f"files2txt-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
num_gpus = get_num_gpus()
start_conversion_clicked = st.sidebar.button("Start", key="start_conversion_btn")

if start_conversion_clicked and html2txt_in_dir and html2txt_out_dir:
    st.session_state["html2txt_run"] = True
    st.session_state["html2txt_vals"] = {
        "input_dir": html2txt_in_dir,
        "output_dir": html2txt_out_dir,
        "session_name": html2txt_sess,
        "num_gpus": num_gpus,
    }

if st.session_state.get("html2txt_run"):
    html2txt_vals = st.session_state.get("html2txt_vals", {})
    st.markdown(f"### Converting Files (.html & .pdf) in Parallel")
    st.markdown(f"**Input:** `{html2txt_vals.get('input_dir', '')}`  \n**Output:** `{html2txt_vals.get('output_dir', '')}`  \n**Session:** `{html2txt_vals.get('session_name', '')}`")
    conversion_status = st.empty()
    def write_status(msg):
        conversion_status.write(msg)
    legal_html2text.streamlit_conversion_runner(
        html2txt_vals["input_dir"],
        html2txt_vals["output_dir"],
        html2txt_vals["session_name"],
        html2txt_vals["num_gpus"],
        status_write_func=write_status
    )
    st.session_state["html2txt_run"] = False
    st.success("Batch file conversion is complete. See above for details.")

# --- Data Load Session Management ---

sessions = get_active_sessions()
if st.session_state["session_page_state"] is None:
    if start_load_clicked:
        st.session_state["session_page_state"] = "NEW"

if st.session_state["session_page_state"] == "NEW":
    # Always generate a fresh session name per new entry
    if "just_set_new_session" not in st.session_state or not st.session_state["just_set_new_session"]:
        unique_default = f"sess-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        st.session_state["session_name"] = unique_default
        st.session_state["just_set_new_session"] = True
    session_name = st.sidebar.text_input("New Session Name", st.session_state.get("session_name", ""))
    st.session_state["session_name"] = session_name
    st.sidebar.header("Corpus Directories")
    add_dir = st.sidebar.text_input("Add a directory (absolute path)", "")
    if st.sidebar.button("Add Directory"):
        if Path(add_dir).exists():
            st.session_state["directories"].add(str(Path(add_dir).resolve()))
            st.sidebar.success(f"Added: {add_dir}")
        else:
            st.sidebar.error("Directory does not exist.")
    for d in sorted(st.session_state["directories"]):
        st.sidebar.write("📁", d)
    if st.sidebar.button("Clear Directory List"):
        st.session_state["directories"] = set()
    selected_dirs = sorted(st.session_state["directories"])
    if st.sidebar.button("Start Ingestion", disabled=not(session_name and selected_dirs), key="start_ingest_btn"):
        st.session_state["run_ingest"] = True
        st.session_state["run_session_type"] = "NEW"

def partition(l, n):
    k, m = divmod(len(l), n)
    return [l[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def write_partition_file(partition, fname):
    with open(fname, "w") as f:
        for item in partition:
            f.write(item + "\n")

def launch_embedding_worker(session_name, embedding_model, gpu=None, partition_file=None):
    python_exec = os.environ.get("PYTHON_EXEC", "python3")
    script_path = os.path.join(os.getcwd(), "embedding_worker.py")
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    args = [python_exec, script_path, session_name, embedding_model]
    if partition_file:
        args.append("--partition_file")
        args.append(partition_file)
    proc = subprocess.Popen(args, env=env)
    return proc

def get_completed_file_count(session_name):
    sess = get_session(session_name)
    if sess:
        return get_completed_file_count_sessname(session_name)
    total = 0
    for i in range(16):
        sfx = f"{session_name}-gpu{i}"
        child_sess = get_session(sfx)
        if child_sess:
            total += count_session_files(sfx, status="complete")
    return total

def get_completed_file_count_sessname(sessname):
    return count_session_files(sessname, status="complete")

def get_total_files(session_name):
    sess = get_session(session_name)
    if sess:
        return sess.total_files or 0
    total = 0
    for i in range(16):
        sfx = f"{session_name}-gpu{i}"
        child_sess = get_session(sfx)
        if child_sess:
            total += child_sess.total_files or 0
    return total

def get_processed_chunks(session_name):
    sess = get_session(session_name)
    if sess:
        return sess.processed_chunks or 0
    total = 0
    for i in range(16):
        sfx = f"{session_name}-gpu{i}"
        child_sess = get_session(sfx)
        if child_sess and getattr(child_sess, "processed_chunks", None) is not None:
            total += child_sess.processed_chunks
    return total

def poll_session_progress_bars(session_name, file_bar=None, chunk_bar_text=None, stat_line=None):
    completed_files = get_completed_file_count(session_name)
    total_files = get_total_files(session_name)
    processed_chunks = get_processed_chunks(session_name)
    sess = get_session(session_name)
    stat = sess.status if sess else '-'
    if total_files > 0:
        file_ratio = min(1.0, completed_files / total_files)
    else:
        file_ratio = 0
    if stat_line: stat_line.write(f"**Status:** {stat}")
    if file_bar: file_bar.progress(file_ratio)
    if file_bar: file_bar_text.write(f"Files embedded: {completed_files} / {total_files}")
    if chunk_bar_text: chunk_bar_text.write(f"Chunks processed: {processed_chunks}")
    return stat

def stop_current_ingest_sessions(session_name, multi_gpu=False):
    fail_session(session_name)
    if multi_gpu:
        for i in range(16):
            sfx = f"{session_name}-gpu{i}"
            sess = get_session(sfx)
            if sess:
                fail_session(sfx)

if st.session_state.get("run_ingest"):
    session_type = st.session_state.get("run_session_type")
    if session_type == "NEW":
        selected_dirs = sorted(st.session_state["directories"])
        session_name = st.session_state["session_name"]
        file_list = list(loader.walk_legal_files(selected_dirs))
        total_files = len(file_list)
        prev = get_session(session_name)
        if prev:
            st.warning("Session already exists. Please pick a new name.")
        else:
            num_gpus = get_num_gpus()
            if num_gpus and num_gpus > 1:
                sublists = partition(file_list, num_gpus)
                sessions = []
                procs = []
                for i in range(num_gpus):
                    sess_name = f"{session_name}-gpu{i}"
                    partition_fname = f".gpu-partition-{sess_name}.txt"
                    write_partition_file(sublists[i], partition_fname)
                    sess = start_session(sess_name, selected_dirs[0], total_files=len(sublists[i]), total_chunks=None)
                    sessions.append(sess)
                    proc = launch_embedding_worker(sess_name, selected_embedding_model, gpu=i, partition_file=partition_fname)
                    procs.append(proc)
                st.success(f"Started {num_gpus} embedding workers in parallel (each on a separate GPU). Sessions: {[s.session_name for s in sessions]}")
            else:
                sess = start_session(session_name, selected_dirs[0], total_files=total_files, total_chunks=None)
                proc = launch_embedding_worker(session_name, selected_embedding_model)
                st.success(f"Started session {session_name} for dir {selected_dirs[0]} (PID {proc.pid}) in the background using embedding model `{selected_embedding_model}`.")
            poll_sec = 1.0
            stat_line = st.empty()
            file_bar = st.empty()
            file_bar_text = st.empty()
            chunk_bar_text = st.empty()
            for _ in range(100000):
                stat = poll_session_progress_bars(session_name, file_bar, chunk_bar_text, stat_line)
                time.sleep(poll_sec)
                if stat in {"complete", "error"}: break
            st.success(f"Ingest session `{session_name}` finished with status {stat}.")
            # Gracefully reset session state to avoid duplicate session names
            st.session_state["run_ingest"] = False
            st.session_state["session_page_state"] = None
            st.session_state["just_set_new_session"] = False
            st.session_state["session_name"] = ""
            st.rerun()

st.markdown("## Legal Document & Hybrid Search")
query = st.text_input("Enter a legal research question or statutory query...")
top_k = st.slider("How many results?", min_value=1, max_value=10, value=5)
alpha = st.slider("Hybrid weighting (higher: more semantic, lower: more keyword)", min_value=0.0, max_value=1.0, value=0.5)

if st.button("Hybrid Search & RAG"):
    st.write("🔎 Performing hybrid legal-aware search and running RAG LLM...")
    try:
        resp = requests.post(f"{API_ROOT}/search/hybrid", json={"query": query, "top_k": top_k, "alpha": alpha}, timeout=30,
            auth=(os.environ.get("FASTAPI_API_USER","legal_api"), os.environ.get("FASTAPI_API_PASS","letmein")))
        resp.raise_for_status()
        hits = resp.json()
    except Exception as e:
        st.error(f"Hybrid search error: {e}")
        hits = []
    if not hits:
        st.warning("No results found for this query in the legal corpus.")
    else:
        st.markdown("**Relevant Document Chunks (Cite & Score):**")
        for i, h in enumerate(hits, 1):
            with st.expander(f"{i}. {h['citation']} | Score: {h.get('hybrid_score',0):.3f}"):
                if h.get("chunk_metadata"):
                    st.markdown("**Metadata:**")
                    for k, v in h["chunk_metadata"].items():
                        st.write(f"- {k}: {v}")
                st.write(f"**Source:** {h['source']}\n**Chunk:** {h.get('chunk_index','?')}\n**Format:** {h.get('format','?')}")
                st.write(f"**Text:**\n{h['text'][:1200]}{'...' if len(h['text'])>1200 else ''}")
                st.write(f"> Confidence: {h.get('hybrid_score',0):.3f} | Vector: {h.get('vector_score',0):.3f} | BM25: {h.get('bm25_score',0):.3f}")
        # Collect context and metadata for RAG:
        context_chunks = [h["text"] for h in hits]
        sources = [h["citation"] for h in hits]
        chunk_metadata = [h.get("chunk_metadata") for h in hits]
        rag = RAGPipeline(model=selected_model)
        with st.spinner(f"Calling {selected_model} via Ollama with legal prompt..."):
            result = rag.query(
                query,
                top_k=top_k,
                context_chunks=context_chunks,
                sources=sources,
                chunk_metadata=chunk_metadata,
                custom_prompt=user_system_prompt
            )
            answer = result.get("answer", "").strip()
            if answer:
                st.markdown(f"**LLM Answer (from {selected_model}, RAG):**")
                st.success(answer)
                st.markdown("**Sources/Citations Used:**")
                for i, src in enumerate(sources):
                    md = chunk_metadata[i] if i < len(chunk_metadata) else None
                    if md:
                        st.markdown("**Chunk Metadata:**")
                        for k, v in md.items():
                            st.write(f"- {k}: {v}")
                    st.info(src)
                good = st.button("👍 Mark answer as correct (QA)")
                bad = st.button("👎 Mark answer as incorrect (QA)")
                if good or bad:
                    feedback = {"question": query, "answer": answer, "sources": sources, "correct": good, "incorrect": bad}
                    st.success("Feedback received! Thank you for enhancing legal RAG QA.")
            else:
                st.warning("No answer returned from Ollama/LLM.")

st.markdown("---")
st.caption("© 2025 Legalsearch Demo | Parallel legal ingestion, explainable legal AI, pgvector & Ollama")
