"""
Legal Assistant Chat (SaaS-style) for auslegalsearchv2
- This is a completely separate, dedicated chat interface with memory,
  hybrid legal search, source popups, custom prompt, citation-enforced LLM answers,
  and user experience for lawyers/law firms.
- Launched via sidebar/floating chat icon in app.py,
  and appears only at /chat route. No "search & RAG" functionality here.
"""

import streamlit as st
from db.store import search_vector, search_bm25, get_file_contents
from embedding.embedder import Embedder
from rag.rag_pipeline import RAGPipeline, list_ollama_models
import os

st.set_page_config(page_title="Legal Assistant Chat", layout="wide")

st.markdown("""
<div style='display: flex; align-items: center; gap: 16px; margin-bottom:0;'>
  <span style='font-size: 2.2em;'></span>
  <span style='font-size: 1.5em; font-weight:700;'>Legal Assistant Chat</span>
</div>
<p style='margin-bottom:2.2em;color:#357bb5;font-weight:500;'>
Your secure, conversational research assistant for legal queries. Powered by RAG, vector and BM25 search, and LLM.
</p>
""", unsafe_allow_html=True)

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "custom_prompt" not in st.session_state:
    st.session_state["custom_prompt"] = (
        "You are a legal research AI assistant. Ground all answers strictly in the provided sources. "
        "Reply 'No factual source of information found' if facts are not present. Always cite sources by name or section."
        " Reply concisely and clearly for attorneys."
    )

model_list = list_ollama_models()
selected_model = st.sidebar.selectbox("RAG LLM model (Ollama)", model_list, index=0) if model_list else "llama3"
embedder = Embedder()

def run_hybrid_search(query, top_k=7):
    # Hybrid: union of BM25 and vector, dedup by file+span, rerank by score
    text_hits = search_bm25(query, top_k=top_k)
    vec_hits = search_vector(embedder.embed([query])[0], top_k=top_k)
    all_hits = { (h["source"], h.get("span","")):h for h in text_hits }
    for h in vec_hits:
        key = (h["source"], h.get("span",""))
        if key in all_hits:
            # Combine scores for union
            all_hits[key]["score"] = (all_hits[key]["score"] + h["score"])/2
        else:
            all_hits[key] = h
    sorted_hits = sorted(all_hits.values(), key=lambda x: -x["score"])
    return sorted_hits[:top_k]

def display_sources(hits):
    st.markdown("<hr style='margin-top:2em;margin-bottom:0;'>", unsafe_allow_html=True)
    st.markdown("<b>Sources used in answer:</b>", unsafe_allow_html=True)
    for i, h in enumerate(hits, 1):
        file_src = h.get("source", "unknown")
        chunk = (h.get("text") or "")[:500]
        code_id = f"src-chunk-{i}"
        if st.button(f"Source [{i}] {file_src}", key=f"src-btn-{i}", help="View excerpt"):
            with st.expander(f"Source View [{file_src}]"):
                st.code(chunk, language="text")
                # If desired, show the full file:
                st.markdown("Click below to view the full source file.")
                if st.button(f"Show full file: {file_src}", key=code_id+"-full"):
                    st.code(get_file_contents(file_src), language="text")

def get_llm_reply(query, hits, custom_prompt, model):
    # Compose full RAG input
    context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    sources = "\n".join(f"[{i}] {h.get('source','')}" for i, h in enumerate(hits, 1))
    prompt = (
        f"{custom_prompt}\n\nRelevant Sources:\n{context}\n"
        f"(Source Files: {sources})\nUser: {query}"
    )
    rag = RAGPipeline(model=model)
    result = rag.query(
        query,
        context_chunks=[h["text"] for h in hits],
        sources=[h.get("source","") for h in hits],
        system_prompt=custom_prompt
    )
    answer = result.get("answer", "").strip()
    if (
        not answer
        or "NO FACTUAL" in answer.upper()
        or all(len((h.get('text') or '')) < 10 for h in hits)
    ):
        return "No factual source of information found."
    return answer

# Chat UI layout (persistent memory)
with st.sidebar:
    st.markdown("## Custom System Prompt")
    st.text_area("Prompt", key="custom_prompt", value=st.session_state["custom_prompt"])
    st.markdown("---")
    st.caption("Powered by hybrid RAG search & Llama3 LLM. Fully legal-focused, shows cited sources.")

chat_container = st.container()
with chat_container:
    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
    query = st.chat_input("Ask your legal question here…")
    if query:
        st.session_state["chat_messages"].append({"role":"user", "content": query})
        hits = run_hybrid_search(query, top_k=7)
        with st.spinner("Retrieving sources and generating answer..."):
            reply = get_llm_reply(query, hits, st.session_state["custom_prompt"], selected_model)
            st.session_state["chat_messages"].append({"role":"assistant", "content": reply})
        st.chat_message("assistant").markdown("##### Sources below ⬇️")
        display_sources(hits)
