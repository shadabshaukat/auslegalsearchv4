"""
AUSLegalSearchv2 – Legal Assistant Chat (Full RAG Chatbot Experience)
Enhanced: Top 10 retrieved chunks shown with clickable URL header, all metadata, score, and cleaner cards.
"""

import streamlit as st
from db.store import search_vector, search_bm25, get_file_contents, save_chat_session
from embedding.embedder import Embedder
from rag.rag_pipeline import RAGPipeline, list_ollama_models
from datetime import datetime, timezone
import time
import uuid

# AUTH WALL: force login if no session, always at top
if "user" not in st.session_state:
    st.warning("You must login to continue.")
    if hasattr(st, "switch_page"):
        st.switch_page("pages/login.py")
    else:
        st.stop()

st.set_page_config(page_title="Legal Assistant Chat", layout="wide")

st.markdown("""
<style>
.user-bubble { background: #e3f2fd; border-radius: 18px; padding: 14px 18px 14px 16px; margin: 10px 0 3px 0; min-width: 32px; max-width: 870px; }
.llm-bubble  { background: #f4f7fe; border-radius: 10px 22px 18px 10px; padding: 14px 16px 14px 20px; margin: 3px 0 12px 18px; min-width: 32px; max-width: 860px; }
.code-pop    { font-family: monospace; font-size:0.94em; background:#232345; color:#d7ffff; border-radius:7px; padding:10px 14px 8px 12px; }
.retr-chunk-card{ border-radius: 8px; border: 1.9px solid #bdd2eb; background: #f7fafd; margin:16px 0 12px 0; padding:13px 19px 10px 20px; }
.chunk-header-link{ font-size:1.18em; font-weight: 600; color:#2057a6; text-decoration:underline; margin-bottom:6px; }
.chunk-metadata{ color:#274052; font-size:0.99em; margin: 4px 0 7px 0; }
.chunk-score{ color:#46807a; font-size:0.98em; margin: 4px 0 3px 0; font-weight:600;}
.chunk-body{ font-family:monospace; font-size:0.98em; margin:7px 0 3px 0; background:#eef4f9; border-left:4px solid #bcdffc; border-radius:4px; padding:9px 14px;}
.streaming {color:#3994c2;font-size:1.09em;}
.session-line { color: #566787; font-size: 0.97em; margin-top: 12px; margin-bottom: 10px !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display: flex; align-items: center; gap: 16px;'>
  <span style='font-size: 2em;'>🤖</span>
  <span style='font-size: 1.51em; font-weight:700;'>Legal Assistant Chat</span>
</div>
<p style='margin-bottom:1.75em;color:#357bb5;font-weight:500;'>
Memory-based chat with Retrieval Augmented Generation over the entire legal corpus – grounded, cited, ready for lawyers.
</p>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "custom_prompt" not in st.session_state:
    st.session_state["custom_prompt"] = (
        "You are a legal research assistant. Always answer based ONLY on the provided sources, and cite sources explicitly."
        " If you have no evidence, reply 'No factual source found.' Use clear, lawyer-friendly drafting and never speculate."
    )

if "chat_session_id" not in st.session_state:
    st.session_state["chat_session_id"] = str(uuid.uuid4())
if "session_started_at" not in st.session_state:
    st.session_state["session_started_at"] = datetime.now(timezone.utc).isoformat()

model_list = list_ollama_models()
selected_model = st.sidebar.selectbox("LLM for RAG", model_list, index=0) if model_list else "llama3"
embedder = Embedder()

with st.sidebar:
    st.markdown("#### Session")
    session_id = st.session_state["chat_session_id"]
    started = st.session_state["session_started_at"]
    st.markdown(f"<span class='session-line'>Session ID:<br><b>{session_id}</b><br>Start: <b>{started.split('.')[0].replace('T', ' ')}</b></span>", unsafe_allow_html=True)
    if st.button("Start New Chat Session"):
        # Capture username as string (use email) and first user question in history
        user = st.session_state.get("user")
        username = user["email"] if isinstance(user, dict) and "email" in user else str(user)
        chat_history = st.session_state["chat_history"]
        question = None
        for msg in chat_history:
            if msg.get("role") == "user" and msg.get("content"):
                question = msg["content"]
                break
        save_chat_session(
            chat_history=chat_history,
            llm_params={
                "model": selected_model,
                "temperature": float(st.session_state.get("temperature", 0.2)),
                "top_p": float(st.session_state.get("top_p", 0.95)),
                "max_tokens": int(st.session_state.get("max_tokens", 1024)),
                "repeat_penalty": float(st.session_state.get("repeat_penalty", 1.1)),
                "custom_prompt": st.session_state["custom_prompt"],
                "top_k": int(st.session_state.get("top_k", 10)),
            },
            ended_at=datetime.now(timezone.utc),
            username=username,
            question=question
        )
        st.session_state["chat_history"] = []
        st.session_state["chat_session_id"] = str(uuid.uuid4())
        st.session_state["session_started_at"] = datetime.now(timezone.utc).isoformat()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            raise st.script_runner.RerunException(st.script_request_queue.RerunData())

    st.markdown("---")
    st.markdown("### LLM Controls", help="Set LLM/decoder params below")
    temperature = st.slider("Temperature", 0.0, 2.0, float(st.session_state.get("temperature", 0.2)), step=0.05, key="temperature")
    top_p = st.slider("Top-p", 0.0, 1.0, float(st.session_state.get("top_p", 0.95)), step=0.01, key="top_p")
    max_tokens = st.number_input("Max tokens", min_value=128, max_value=4096, value=int(st.session_state.get("max_tokens", 1024)), step=32, key="max_tokens")
    repeat_penalty = st.slider("Repeat penalty", 1.0, 2.0, float(st.session_state.get("repeat_penalty", 1.1)), step=0.05, key="repeat_penalty")
    top_k = st.number_input("Max sources per answer", min_value=3, max_value=12, value=int(st.session_state.get("top_k", 10)), step=1, key="top_k")
    st.markdown("**Session system prompt:**")
    st.text_area("Prompt", key="custom_prompt", value=st.session_state["custom_prompt"], height=90)
    st.markdown("---")
    st.caption("All answers grounded in your legal corpus.")

def hybrid_search(query, top=10):
    text_hits = search_bm25(query, top_k=top)
    vec_hits = search_vector(embedder.embed([query])[0], top_k=top)
    seen = set()
    all_hits = []
    for x in text_hits + vec_hits:
        k = (x["source"], x.get("span", ""))
        if k not in seen:
            seen.add(k)
            all_hits.append(x)
    all_hits.sort(key=lambda x: -x.get("score", 0))
    return all_hits[:top], vec_hits

def get_recent_chat_history_str(max_turns=3):
    chat = st.session_state["chat_history"][-2*max_turns:] if len(st.session_state["chat_history"]) > 0 else []
    transcript = []
    for entry in chat:
        if entry["role"] == "user":
            transcript.append(f"User: {entry['content']}")
        elif entry["role"] == "llm":
            transcript.append(f"Assistant: {entry['content']}")
    return "\n".join(transcript)

def rag_llm(query, context_chunks, sources, custom_prompt, model, stream_cb, _temperature, _top_p, _max_tokens, _repeat_penalty):
    rag = RAGPipeline(model=model)
    chat_history_str = get_recent_chat_history_str(max_turns=3)
    sys_and_history = f"{custom_prompt.strip()}\n\nChat:\n{chat_history_str}\n" if chat_history_str else custom_prompt.strip()
    chunks = context_chunks
    try:
        rag_out = rag.query(
            question=query,
            context_chunks=chunks,
            sources=sources,
            custom_prompt=sys_and_history,
            temperature=_temperature,
            top_p=_top_p,
            max_tokens=int(_max_tokens),
            repeat_penalty=_repeat_penalty
        )["answer"]
        buf = ""
        for ch in rag_out:
            buf += ch
            stream_cb(buf + "▌")
            time.sleep(0.012)
        stream_cb(buf)
        return buf
    except Exception as e:
        stream_cb(f"❗ Error: {str(e)}")
        return f"Error: {str(e)}"

def show_chunks(chunks, vec_hits):
    st.markdown("<b>Top Retrieved Chunks</b>", unsafe_allow_html=True)
    for i, c in enumerate(chunks[:10], 1):  # Always display at most 10
        url = c.get("url", "")
        source_display = c.get("source", "(unknown)")
        # clickable header: url, else source string
        if url:
            hdr = f'<a href="{url}" class="chunk-header-link" target="_blank">{source_display if source_display else url}</a>'
        else:
            hdr = f'<span class="chunk-header-link">{source_display}</span>'
        meta_out = ""
        meta = {k: v for k, v in c.items() if k not in ("text","source","url","score")}
        for mk, mv in meta.items():
            meta_out += f"<div><b>{mk}:</b> <span>{str(mv)}</span></div>"
        score = c.get("score", 0)
        st.markdown(
            f"""
            <div class="retr-chunk-card">
                {hdr}
                <div class="chunk-score">Score: {score:.4f}</div>
                <div class="chunk-metadata">{meta_out}</div>
                <div class="chunk-body">{c.get('text','')[:500]}{' ...' if len(c.get('text',''))>500 else ''}</div>
            </div>
            """, unsafe_allow_html=True)
    if vec_hits:
        st.caption("Semantic (vector-only) top results for reference:")
        for i,v in enumerate(vec_hits[:10],1):
            url = v.get("url", "")
            source_display = v.get("source", "(unknown)")
            if url:
                hdr = f'<a href="{url}" class="chunk-header-link" target="_blank">{source_display if source_display else url}</a>'
            else:
                hdr = f'<span class="chunk-header-link">{source_display}</span>'
            meta_out = ""
            meta = {k: v for k, v in v.items() if k not in ("text","source","url","score")}
            for mk, mv in meta.items():
                meta_out += f"<div><b>{mk}:</b> <span>{str(mv)}</span></div>"
            score = v.get("score", 0)
            st.markdown(
                f"""
                <div class="retr-chunk-card">
                    {hdr}
                    <div class="chunk-score">Score: {score:.4f}</div>
                    <div class="chunk-metadata">{meta_out}</div>
                    <div class="chunk-body">{v.get('text','')[:340]}{' ...' if len(v.get('text',''))>340 else ''}</div>
                </div>
                """, unsafe_allow_html=True)

def show_message(role, msg):
    if role == "user":
        st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='llm-bubble'>{msg}</div>", unsafe_allow_html=True)

with st.container():
    for turn in st.session_state["chat_history"]:
        show_message(turn["role"], turn["content"])
        if turn.get("chunks"):
            show_chunks(turn["chunks"], turn.get("vec_hits"))
    user_prompt = st.chat_input("Ask your legal question or request a summary, clause extraction, etc…")
    if user_prompt:
        st.session_state["chat_history"].append({"role":"user","content":user_prompt})
        doc_hits, vec_hits = hybrid_search(user_prompt, top=10)  # Always top 10
        st.session_state["chat_history"].append({"role": "llm", "content": "<i>Retrieving and synthesizing answer…</i>", "chunks": doc_hits, "vec_hits": vec_hits})
        answer_slot = st.empty()
        def stream_cb(val): answer_slot.markdown(f"<div class='llm-bubble streaming'>{val}</div>", unsafe_allow_html=True)
        context_chunks = [d["text"] for d in doc_hits]
        sources = [d.get("source","") for d in doc_hits]
        answer = rag_llm(
            user_prompt, context_chunks, sources, st.session_state["custom_prompt"], selected_model, stream_cb,
            temperature, top_p, max_tokens, repeat_penalty
        )
        st.session_state["chat_history"][-1] = {
            "role": "llm",
            "content": answer,
            "chunks": doc_hits,
            "vec_hits": vec_hits
        }
        show_message("llm", answer)
        show_chunks(doc_hits, vec_hits)
