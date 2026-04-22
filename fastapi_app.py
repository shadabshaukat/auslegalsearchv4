"""
FastAPI backend for AUSLegalSearchv3, with legal-tuned hybrid QA, legal-aware chunking, RAG, reranker interface, OCI GenAI, Oracle 23ai DB, and full system prompt config.
"""

# Always load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os
import secrets
import json
import inspect

from db.store import (
    create_user, get_user_by_email, hash_password, check_password,
    add_document, add_embedding, start_session, get_session, complete_session, fail_session, update_session_progress,
    search_vector, search_bm25, search_hybrid, get_chat_session, save_chat_session,
    get_active_sessions, get_resume_sessions, search_fts, create_all_tables,
    list_documents, get_document_by_id
)
from embedding.embedder import Embedder
from rag.rag_pipeline import RAGPipeline, list_ollama_models
from rag.oci_rag_pipeline import OCIGenAIPipeline
from db.oracle23ai_connector import Oracle23AIConnector
from ingest.loader import walk_legal_files, parse_txt, parse_html, chunk_document

RERANKER_DATA_PATH = "./reranker_models.json"

_DEFAULT_RERANKER_MODELS = [
    {"name": "mxbai-rerank-xsmall", "desc": "General XC; top-3 HuggingFace scorer", "hf": "mixedbread-ai/mxbai-rerank-xsmall"},
    {"name": "ms-marco-MiniLM-L-6-v2", "desc": "MS MARCO cross-encoder baseline", "hf": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    {"name": "nlpaueb-legal-bert-small-uncased", "desc": "Legal-domain (smaller) classifier", "hf": "nlpaueb/legal-bert-small-uncased"},
]

def load_reranker_models():
    if os.path.exists(RERANKER_DATA_PATH):
        with open(RERANKER_DATA_PATH, "r") as f:
            data = json.load(f)
            return {m["name"]: m for m in data}
    return {m["name"]: m for m in _DEFAULT_RERANKER_MODELS}

def save_reranker_models(registry):
    with open(RERANKER_DATA_PATH, "w") as f:
        json.dump(list(registry.values()), f, indent=2)

_INSTALLED_RERANKERS = load_reranker_models()
_DEFAULT_MODEL = "mxbai-rerank-xsmall"

def available_rerankers():
    return list(_INSTALLED_RERANKERS.values())

def get_reranker_model(model_name):
    if not model_name or model_name not in _INSTALLED_RERANKERS:
        model_name = _DEFAULT_MODEL
    m = _INSTALLED_RERANKERS[model_name]
    return m

def download_hf_model(hf_repo):
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder(hf_repo)
    except Exception as e:
        print(f"Could not download {hf_repo}: {e}")

app = FastAPI(
    title="AUSLegalSearchv3 API",
    description="REST API for legal vector search, ingestion, RAG, chat, reranker, system prompt, OCI GenAI, and Oracle 23ai DB.",
    version="0.40"
)

# Ensure DB schema on API startup when enabled
@app.on_event("startup")
async def _bootstrap_db_schema():
    if os.environ.get("AUSLEGALSEARCH_AUTO_DDL", "1") == "1":
        try:
            create_all_tables()
            print("[fastapi] DB schema ensured (AUTO_DDL=1)")
        except Exception as e:
            print(f"[fastapi] DB schema ensure failed: {e}")

security = HTTPBasic()
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    api_user = os.environ.get("FASTAPI_API_USER", "legal_api")
    api_pass = os.environ.get("FASTAPI_API_PASS", "letmein")
    correct_username = secrets.compare_digest(credentials.username, api_user)
    correct_password = secrets.compare_digest(credentials.password, api_pass)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect API credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- User endpoints ---
class UserCreateReq(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

@app.post("/users", tags=["users"])
def api_create_user(user: UserCreateReq, _: str = Depends(get_current_user)):
    db_user = get_user_by_email(user.email)
    if db_user:
        raise HTTPException(400, detail="Email already registered")
    return create_user(email=user.email, password=user.password, name=user.name)

# --- Ingestion endpoints ---
class IngestStartReq(BaseModel):
    directory: str
    session_name: str

@app.post("/ingest/start", tags=["ingest"])
def api_ingest_start(req: IngestStartReq, _: str = Depends(get_current_user)):
    session = start_session(session_name=req.session_name, directory=req.directory)
    return {"session_name": session.session_name, "status": session.status, "started_at": session.started_at}

@app.get("/ingest/sessions", tags=["ingest"])
def api_active_ingest_sessions(_: str = Depends(get_current_user)):
    return [s.session_name for s in get_active_sessions()]

@app.post("/ingest/stop", tags=["ingest"])
def api_stop_ingest(session_name: str, _: str = Depends(get_current_user)):
    fail_session(session_name)
    return {"session": session_name, "status": "stopped"}

# --- Document endpoints ---
@app.get("/documents", tags=["documents"])
def api_list_documents(_: str = Depends(get_current_user)):
    return list_documents(limit=100)

@app.get("/documents/{doc_id}", tags=["documents"])
def api_get_document(doc_id: int, _: str = Depends(get_current_user)):
    d = get_document_by_id(doc_id)
    if not d:
        raise HTTPException(404, "Document not found")
    return d

# --- Embedding Search endpoints ---
class SearchReq(BaseModel):
    query: str
    top_k: int = 5
    model: Optional[str] = None

@app.post("/search/vector", tags=["search"])
def api_search_vector(req: SearchReq, _: str = Depends(get_current_user)):
    embedder = Embedder()
    query_vec = embedder.embed([req.query])[0]
    hits = search_vector(query_vec, top_k=req.top_k)
    # Include metadata in results
    for hit in hits:
        if "chunk_metadata" not in hit:
            hit["chunk_metadata"] = None
    return hits

@app.post("/search/rerank", tags=["search"])
def api_search_rerank(req: SearchReq, _: str = Depends(get_current_user)):
    reranker_info = get_reranker_model(req.model)
    embedder = Embedder()
    query_vec = embedder.embed([req.query])[0]
    hits = search_vector(query_vec, top_k=max(20, req.top_k))
    hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)[:req.top_k]
    for h in hits:
        h["reranker"] = reranker_info["name"]
        if "chunk_metadata" not in h:
            h["chunk_metadata"] = None
    return hits

@app.get("/models/reranker", tags=["models"])
def api_reranker_models(_: str = Depends(get_current_user)):
    return available_rerankers()

@app.get("/models/rerankers", tags=["models"])
def api_rerankers_list(_: str = Depends(get_current_user)):
    # Return just a list of installed reranker names for dropdowns etc.
    return list(_INSTALLED_RERANKERS.keys())

class RerankerDownloadReq(BaseModel):
    name: str
    hf_repo: str
    desc: Optional[str] = None

@app.post("/models/reranker/download", tags=["models"])
def download_reranker_model(req: RerankerDownloadReq, background_tasks: BackgroundTasks, _: str = Depends(get_current_user)):
    if req.name in _INSTALLED_RERANKERS:
        return {"message": f"Model {req.name} is already installed.", "models": available_rerankers()}
    background_tasks.add_task(download_hf_model, req.hf_repo)
    entry = {"name": req.name, "desc": req.desc or "", "hf": req.hf_repo}
    _INSTALLED_RERANKERS[req.name] = entry
    save_reranker_models(_INSTALLED_RERANKERS)
    return {"message": f"Download started for {req.name}", "models": available_rerankers()}

# --- Hybrid Legal Search endpoint ---
class HybridSearchReq(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5

@app.post("/search/hybrid", tags=["search"])
def api_search_hybrid(req: HybridSearchReq, _: str = Depends(get_current_user)):
    """
    Hybrid (vector+bm25) legal search with score and citation output, now with metadata.
    """
    results = search_hybrid(req.query, top_k=req.top_k, alpha=req.alpha)
    fields = ["doc_id", "chunk_index", "hybrid_score", "citation", "score", "vector_score", "bm25_score", "text", "source", "format", "chunk_metadata"]
    filtered = [
        {k: r[k] for k in fields if k in r}
        for r in results
    ]
    return filtered

# --- RAG/Llama Legal QA with memory ---
class RagReq(BaseModel):
    question: str
    context_chunks: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    chunk_metadata: Optional[List[Dict[str, Any]]] = None
    custom_prompt: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.90
    max_tokens: int = 1024
    repeat_penalty: float = 1.1
    model: Optional[str] = "llama3"
    reranker_model: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

@app.post("/search/rag", tags=["search"])
def api_search_rag(req: RagReq, _: str = Depends(get_current_user)):
    rag = RAGPipeline(model=req.model or "llama3")
    query_kwargs = dict(
        context_chunks=req.context_chunks,
        sources=req.sources,
        chunk_metadata=req.chunk_metadata,
        custom_prompt=req.custom_prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        repeat_penalty=req.repeat_penalty,
    )
    query_sig = inspect.signature(rag.query)
    if req.chat_history is not None and "chat_history" in query_sig.parameters:
        query_kwargs["chat_history"] = req.chat_history
    return rag.query(
        req.question,
        **query_kwargs
    )

# === OCI GenAI RAG endpoints ===
class OCIConfig(BaseModel):
    compartment_id: str
    model_id: str
    region: Optional[str] = None
    # optionally allow oci_config dict if needed in future

class OCIRagReq(BaseModel):
    oci_config: OCIConfig
    question: str
    context_chunks: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    chunk_metadata: Optional[List[Dict[str, Any]]] = None
    custom_prompt: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    repeat_penalty: float = 1.1
    chat_history: Optional[List[Dict[str, Any]]] = None

@app.post("/search/oci_rag", tags=["search", "oci"])
def api_search_oci_rag(req: OCIRagReq, _: str = Depends(get_current_user)):
    pipeline = OCIGenAIPipeline(
        compartment_id=req.oci_config.compartment_id,
        model_id=req.oci_config.model_id,
        region=req.oci_config.region
    )
    return pipeline.query(
        question=req.question,
        context_chunks=req.context_chunks,
        sources=req.sources,
        chunk_metadata=req.chunk_metadata,
        custom_prompt=req.custom_prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        repeat_penalty=req.repeat_penalty,
        chat_history=req.chat_history,
    )

# --- Conversational Chat endpoint ---
class ChatConversationReq(BaseModel):
    llm_source: str  # "ollama" or "oci_genai"
    model: Optional[str] = None
    message: str
    chat_history: Optional[list] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1024
    repeat_penalty: Optional[float] = 1.1
    # oci config, if applicable
    oci_config: Optional[Dict[str, Any]] = None

# --- Agentic Chat endpoint ---
class ChatAgenticReq(BaseModel):
    llm_source: str  # "ollama" or "oci_genai"
    model: Optional[str] = None
    message: str
    chat_history: Optional[list] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1024
    repeat_penalty: Optional[float] = 1.1
    top_k: Optional[int] = 10
    oci_config: Optional[Dict[str, Any]] = None

@app.post("/chat/agentic", tags=["chat"])
def api_agentic_chat(req: ChatAgenticReq, _: str = Depends(get_current_user)):
    """
    Agentic Chat endpoint: Produces a chain-of-thought (multi-step) response from agent, with structured step trace for both Ollama and OCI GenAI.
    """
    # System prompt: force stepwise reasoning w/ explicit format (JSON or Markdown).
    agent_prompt = (
        (req.system_prompt or "")
        + "\n\nYou are an expert legal agent. For every question, reason step by step with an explicit agentic chain of thought workflow." 
          " For each step, output as:\n"
          "Step X - Thought: ...\nStep X - Action: ...\nStep X - Evidence: ...\nStep X - Reasoning: ...\n"
          "At the end, output Final Conclusion: ...\nBe sure to cite sources in each step using extracted evidence. Return all steps and final conclusion in Markdown format."
    )

    # Retrieve Top K as context (like RAG)
    from db.store import search_hybrid
    top_k = getattr(req, "top_k", 10) or 10
    hybrid_hits = search_hybrid(req.message, top_k=top_k, alpha=0.5)
    context_chunks = [h.get("text", "") for h in hybrid_hits]
    sources = [h.get("citation") or ((h.get("chunk_metadata") or {}).get("url") if (h.get("chunk_metadata") or {}) else "?") for h in hybrid_hits]
    chunk_metadata = [h.get("chunk_metadata") or {} for h in hybrid_hits]

    out = {}
    query_args = dict(
        context_chunks=context_chunks,
        sources=sources,
        chunk_metadata=chunk_metadata,
        custom_prompt=agent_prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        repeat_penalty=req.repeat_penalty,
        chat_history=req.chat_history,
    )

    if req.llm_source.lower() == "ollama":
        rag = RAGPipeline(model=req.model or "llama3")
        llm_resp = rag.query(req.message, **query_args)
        answer = llm_resp.get("answer", "")
    elif req.llm_source.lower() == "oci_genai":
        config = req.oci_config or {}
        compartment_id = config.get("compartment_id") or os.environ.get("OCI_COMPARTMENT_OCID", "")
        model_id = config.get("model_id") or os.environ.get("OCI_GENAI_MODEL_OCID", "")
        region = config.get("region") or os.environ.get("OCI_REGION", "")
        pipeline = OCIGenAIPipeline(
            compartment_id=compartment_id,
            model_id=model_id,
            region=region
        )
        llm_resp = pipeline.query(
            question=req.message,
            **query_args
        )
        answer = llm_resp.get("answer", "")
    else:
        raise HTTPException(400, f"Unknown llm_source {req.llm_source}")

    out["answer"] = answer
    out["sources"] = sources
    out["context_chunks"] = context_chunks
    out["chunk_metadata"] = chunk_metadata
    # Optionally parse out steps (if you want special structure—UI may parse Markdown)
    return out

@app.post("/chat/conversation", tags=["chat"])
def api_conversational_chat(req: ChatConversationReq, _: str = Depends(get_current_user)):
    """
    Conversational endpoint with Top K vector search + context injection.
    For each turn: runs hybrid search using message, passes results to LLM (Ollama or OCI GenAI), returns both answer and sources/cards for UI display.
    """
    history = req.chat_history or []
    system_prompt = req.system_prompt

    # Legal-focused default prompt
    LEGAL_ASSISTANT_PROMPT = (
        "You are an expert Australian legal research and compliance AI assistant. "
        "Answer strictly from the provided sources and context. Always cite the source section/citation for every statement. "
        "If you do not know the answer from the context, reply: 'Not found in the provided legal documents.' "
        "Be neutral, factual, and never invent legal advice."
    )
    custom_prompt = system_prompt or LEGAL_ASSISTANT_PROMPT

    # New: Per-turn hybrid search for Top K context
    from db.store import search_hybrid
    top_k = getattr(req, "top_k", 10)
    hybrid_hits = search_hybrid(req.message, top_k=top_k, alpha=0.5)
    context_chunks = [h.get("text", "") for h in hybrid_hits]
    sources = [h.get("citation") or ((h.get("chunk_metadata") or {}).get("url") if (h.get("chunk_metadata") or {}) else "?") for h in hybrid_hits]
    chunk_metadata = [h.get("chunk_metadata") or {} for h in hybrid_hits]

    out = {}
    if req.llm_source.lower() == "ollama":
        rag = RAGPipeline(model=req.model or "llama3")
        llm_resp = rag.query(
            req.message,
            chat_history=history,
            context_chunks=context_chunks,
            sources=sources,
            chunk_metadata=chunk_metadata,
            custom_prompt=custom_prompt,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            repeat_penalty=req.repeat_penalty
        )
        out["answer"] = llm_resp.get("answer", "")
        out["sources"] = sources
        out["chunk_metadata"] = chunk_metadata
        out["context_chunks"] = context_chunks
    elif req.llm_source.lower() == "oci_genai":
        config = req.oci_config or {}
        compartment_id = config.get("compartment_id") or os.environ.get("OCI_COMPARTMENT_OCID", "")
        model_id = config.get("model_id") or os.environ.get("OCI_GENAI_MODEL_OCID", "")
        region = config.get("region") or os.environ.get("OCI_REGION", "")
        pipeline = OCIGenAIPipeline(
            compartment_id=compartment_id,
            model_id=model_id,
            region=region
        )
        llm_resp = pipeline.query(
            question=req.message,
            chat_history=history,
            context_chunks=context_chunks,
            sources=sources,
            chunk_metadata=chunk_metadata,
            custom_prompt=custom_prompt,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            repeat_penalty=req.repeat_penalty
        )
        out["answer"] = llm_resp.get("answer", "")
        out["sources"] = sources
        out["chunk_metadata"] = chunk_metadata
        out["context_chunks"] = context_chunks
    else:
        raise HTTPException(400, f"Unknown llm_source {req.llm_source}")

    return out

# --- Chat Session endpoints ---
class ChatMsg(BaseModel):
    prompt: str
    model: Optional[str] = "llama3"

@app.post("/chat/session", tags=["chat"])
def api_chat_session(msg: ChatMsg, _: str = Depends(get_current_user)):
    rag = RAGPipeline(model=msg.model or "llama3")
    results = search_bm25(msg.prompt, top_k=5)
    context_chunks = [r["text"] for r in results]
    sources = [r["source"] for r in results]
    chunk_metadata = [r.get("chunk_metadata") for r in results]
    answer = rag.query(
        msg.prompt, context_chunks=context_chunks, sources=sources, chunk_metadata=chunk_metadata
    )["answer"]
    return {
        "answer": answer,
        "sources": sources,
        "chunk_metadata": chunk_metadata
    }

# === Oracle 23ai DB endpoints ===
class Oracle23aiQueryReq(BaseModel):
    user: Optional[str] = None
    password: Optional[str] = None
    dsn: Optional[str] = None
    wallet_location: Optional[str] = None
    sql: str
    params: Optional[List[Any]] = None

@app.post("/db/oracle23ai_query", tags=["db", "oracle"])
def api_oracle23ai_query(req: Oracle23aiQueryReq, _: str = Depends(get_current_user)):
    connector = Oracle23AIConnector(
        user=req.user,
        password=req.password,
        dsn=req.dsn,
        wallet_location=req.wallet_location
    )
    try:
        results = connector.run_query(req.sql, tuple(req.params or ()))
        return results
    finally:
        connector.close()

# --- Full Text Search endpoint ---
class FtsSearchReq(BaseModel):
    query: str
    top_k: int = 10
    mode: str = "both"  # "documents", "metadata", "both"

@app.post("/search/fts", tags=["search"])
def api_search_fts(req: FtsSearchReq, _: str = Depends(get_current_user)):
    """
    Full text search using stemming and phrase logic over document_fts (in documents),
    chunk_metadata (in embeddings), or both, as chosen by 'mode' param.
    """
    return search_fts(req.query, req.top_k, req.mode)

# --- Utility/model endpoints ---
@app.get("/models/oci_genai", tags=["models"])
def api_oci_genai_models(_: str = Depends(get_current_user)):
    """
    List available OCI GenAI LLM models for the configured compartment/region.
    Returns OCID and display name for each model supporting TextGeneration.
    """
    import os
    try:
        import oci
        from oci.generative_ai_inference import GenerativeAiInferenceClient
        compartment_id = os.environ.get("OCI_COMPARTMENT_OCID", "")
        region = os.environ.get("OCI_REGION", "ap-sydney-1")
        config = {
            "user": os.environ.get("OCI_USER_OCID", ""),
            "key_file": os.environ.get("OCI_KEY_FILE", os.path.expanduser("~/.oci/oci_api_key.pem")),
            "fingerprint": os.environ.get("OCI_KEY_FINGERPRINT", ""),
            "tenancy": os.environ.get("OCI_TENANCY_OCID", ""),
            "region": region,
        }
        profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
        try:
            file_conf = oci.config.from_file("~/.oci/config", profile_name=profile)
            config.update(file_conf)
        except Exception:
            pass
        config["user"] = os.environ.get("OCI_USER_OCID", config.get("user", ""))
        config["key_file"] = os.environ.get("OCI_KEY_FILE", config.get("key_file", os.path.expanduser("~/.oci/oci_api_key.pem")))
        config["fingerprint"] = os.environ.get("OCI_KEY_FINGERPRINT", config.get("fingerprint", ""))
        config["tenancy"] = os.environ.get("OCI_TENANCY_OCID", config.get("tenancy", ""))
        config["region"] = os.environ.get("OCI_REGION", config.get("region", "ap-sydney-1"))
        client = GenerativeAiInferenceClient(config)
        # Prefer to only list Oracle-provided PRETRAINED chat models
        try:
            resp = client.list_models(
                compartment_id=compartment_id,
                lifecycle_state="ACTIVE",
                model_listing_type="PRETRAINED"
            )
        except TypeError:
            # Fallback for SDKs that don't support model_listing_type param
            resp = client.list_models(
                compartment_id=compartment_id,
                lifecycle_state="ACTIVE"
            )
        filtered = []
        debug_list = []
        for model in getattr(resp, "data", []):
            attrs = {}
            for k in dir(model):
                if k.startswith("_") or k == "parent":
                    continue
                try:
                    v = getattr(model, k)
                    json.dumps(v, default=str)
                    attrs[k] = v
                except Exception:
                    continue
            debug_list.append(attrs)
            # Keep only base/prebuilt chat models: use category/model_type/operation_types
            category = str(attrs.get("category", "")).lower()
            modeltype = str(attrs.get("model_type", "")).lower() if "model_type" in attrs else ""
            op_types = [str(op).lower() for op in attrs.get("operation_types", [])]
            if "llm" in category or "chat" in modeltype or any("chat" in op for op in op_types):
                filtered.append(attrs)
        print("DEBUG: Returning Oracle PRETRAINED/Chat models:", json.dumps(filtered, indent=2, default=str))
        try:
            with open("oracle_model_debug_list.json", "w") as f:
                json.dump(filtered, f, indent=2, default=str)
        except Exception as e:
            print("WARNING: Could not write oracle_model_debug_list.json:", e)
        return {"all": filtered}
    except Exception as e:
        return [{"error": f"Failed to fetch OCI models, check backend logs. {e}"}]

@app.get("/models/ollama", tags=["models"])
def api_ollama_models(_: str = Depends(get_current_user)):
    models = list_ollama_models()
    print("DEBUG: /models/ollama returned models:", models)
    return models

@app.get("/files/ls", tags=["files"])
def api_ls(path: str, _: str = Depends(get_current_user)):
    allowed_roots = [os.getcwd(), "/home/ubuntu/data"]
    real = os.path.realpath(path)
    if not any(real.startswith(os.path.realpath(x)) for x in allowed_roots):
        raise HTTPException(403, "Not allowed")
    if os.path.isdir(real):
        return os.listdir(real)
    else:
        return [real]

@app.get("/health", tags=["utility"])
def healthcheck():
    return {"ok": True}
