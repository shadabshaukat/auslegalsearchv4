# AUSLegalSearch v3 — RAG Pipelines

Retrieval Augmented Generation (RAG) runtime for local Ollama LLMs and Oracle Cloud GenAI. These pipelines accept pre-retrieved context chunks and format them (with metadata) into prompts, call the LLM, and return structured results for UI and API layers.

Modules
- rag/rag_pipeline.py — RAGPipeline for Ollama (local models), plus list_ollama_models()
- rag/oci_rag_pipeline.py — OCIGenAIPipeline for Oracle Cloud GenAI (remote service)


## Design and integration

- Separation of concerns
  - Retrieval is not hard-wired in the pipeline. Upstream layers (FastAPI, Streamlit UI) are expected to perform retrieval (e.g., hybrid vector+BM25) and pass context chunks, source URIs, and chunk metadata.
  - The pipeline focuses on prompt construction and LLM invocation: formatting context blocks, enforcing legal-grade instructions, and returning answer + sources + metadata for display.

- Upstream components
  - FastAPI endpoints (fastapi_app.py)
    - /search/rag uses RAGPipeline
    - /search/oci_rag uses OCIGenAIPipeline
    - /chat/agentic uses agentic prompting (Chain-of-Thought) with RAGPipeline or OCIGenAIPipeline depending on llm_source
    - /chat/conversation threads conversational context through custom prompts
  - Streamlit pages (pages/chat.py)
    - Performs a hybrid search (BM25 + vector) locally and passes top-k chunks into RAGPipeline for answer generation
    - Renders cards with per-chunk metadata and URLs

- Metadata-rich context
  - All pipelines can accept a parallel list of chunk_metadata (one dict per context chunk)
  - The context block includes selected metadata keys above the text to improve grounding, citations, and legal explainability


## RAGPipeline (Ollama, local)

File: rag/rag_pipeline.py

- list_ollama_models(ollama_url="http://localhost:11434")
  - Returns installed model names via Ollama REST API /api/tags
  - Used by UI/API to populate model dropdowns

- RAGPipeline(ollama_url="http://localhost:11434", model="llama3")
  - Arguments
    - ollama_url: Base URL for Ollama runtime (default http://localhost:11434)
    - model: LLM name/tag (e.g., llama3, llama3:8b-instruct, mistral, etc.)
  - Methods
    - retrieve(query, k=5)
      - Placeholder in v3 (UI/API retrieve separately). Returns sample chunks/sources for legacy paths
    - query(
        question, top_k=5, context_chunks=None, sources=None, chunk_metadata=None,
        custom_prompt=None, temperature=0.2, top_p=0.95, max_tokens=1024, repeat_penalty=1.1, chat_history=None
      ) -> dict
      - Accepts pre-retrieved chunks and optional metadata; builds a composite prompt
      - Calls llama4_rag() which posts to Ollama /api/generate with options
      - Returns: { "answer": str, "sources": list[str], "contexts": list[str], "chunk_metadata": list[dict] }
    - llama4_rag(...)
      - Formats the final prompt with CONTEXT blocks + QUESTION + ANSWER:
      - Posts to Ollama with temperature/top_p/num_predict/repeat_penalty

- Prompt building nuances
  - _generate_context_block emits metadata header (key: value) lines + a delimiter (---) before the chunk text
  - custom_prompt lets upstream enforce a system/role prompt; see FastAPI /chat/agentic for CoT-formatted instructions


Example (Python)
```python
from rag.rag_pipeline import RAGPipeline

rag = RAGPipeline(model="llama3")
contexts = ["Clause 1 text...", "Clause 2 text..."]
metas = [{"title": "Agreement XYZ", "jurisdiction": "AU"}, {"title": "Schedule A", "section": "2"}]
resp = rag.query(
    question="Summarize the assignment clause obligations.",
    context_chunks=contexts,
    chunk_metadata=metas,
    custom_prompt="You are a legal assistant. Cite sources and be concise."
)
print(resp["answer"])
```


## OCIGenAIPipeline (Oracle Cloud GenAI)

File: rag/oci_rag_pipeline.py

- OCIGenAIPipeline(compartment_id, model_id, oci_config=None, region=None)
  - Credentials and config resolution in _default_oci_config():
    - Uses environment variables: OCI_USER_OCID, OCI_KEY_FILE, OCI_KEY_FINGERPRINT, OCI_TENANCY_OCID, OCI_REGION (default ap-sydney-1), OCI_CONFIG_PROFILE (DEFAULT)
    - Attempts to load ~/.oci/config if available; env values override file
  - Builds a GenerativeAiInferenceClient and uses the chat() API with GenericChatRequest for broad model compatibility

- query(
    question, context_chunks=None, sources=None, chunk_metadata=None,
    custom_prompt=None, temperature=0.2, top_p=0.95, max_tokens=1024, repeat_penalty=1.1,
    chat_history=None, system_prompt=None, model_info=None
  ) -> dict
  - Builds a system prompt (custom or default legal instruction)
  - Formats context as metadata header + text per chunk, separated by ---
  - Creates a ChatDetails request and calls genai_client.chat(...)
  - Returns: { "answer": str, "sources": list[str], "contexts": list[str], "chunk_metadata": list[dict] }
  - On exception, returns an error string in "answer"

- Required env/inputs
  - OCI_USER_OCID, OCI_TENANCY_OCID, OCI_KEY_FILE, OCI_KEY_FINGERPRINT, OCI_REGION
  - OCID for compartment and model (passed to the constructor or via env in the FastAPI layer)


Example (Python)
```python
from rag.oci_rag_pipeline import OCIGenAIPipeline

pipeline = OCIGenAIPipeline(
    compartment_id="ocid1.compartment.oc1..example",
    model_id="ocid1.generativeaiocid..example",
    region="ap-sydney-1"
)
chunks = ["Section 96: ...", "Case summary: ..."]
metas = [{"citation": "Succession Act 2006 (NSW) s 96"}, {"case": "Smith v Doe [2022] NSWSC 789"}]
resp = pipeline.query(
    question="Explain the procedure for contesting a will in NSW.",
    context_chunks=chunks,
    chunk_metadata=metas,
    custom_prompt="You are a legal assistant. Answer only from context and cite sources."
)
print(resp["answer"])
```


## Prompting patterns

- Legal-grade instructions (typical)
  - "Answer only from the provided context. Cite sources in every step. If not found, say 'Not found in the provided legal documents.'"
  - Agentic CoT (Chain-of-Thought) format used in FastAPI /chat/agentic:
    - Step X - Thought / Action / Evidence / Reasoning, ending with Final Conclusion

- Metadata-first context
  - Provide critical metadata fields first to enhance grounding and citations:
    - title, regulation/section, citation, year, jurisdiction, database, url


## Configuration and environment

Ollama (RAGPipeline)
- OLLAMA host (default http://localhost:11434); override via RAGPipeline(ollama_url=...)
- Model selection aligns with list_ollama_models()
- Long answers might need higher max_tokens; tune temperature/top_p for verbosity and determinism

OCI GenAI (OCIGenAIPipeline)
- Ensure OCI Python SDK is installed (oci) and credentials are valid
- Set region and compartment/model OCIDs
- Chat API is used by default to match Oracle chat-style LLMs


## Error handling and timeouts

- RAGPipeline:
  - Returns "Error querying Llama4: ..." when Ollama REST call fails (non-200 status)
- OCIGenAIPipeline:
  - Raises ImportError if oci is not installed
  - Returns "Error querying OCI GenAI: ..." string if chat invocation raises an exception
- Upstream layers should display the error string as an LLM answer and log details server-side


## Best practices

- Retrieval upstream
  - Prefer hybrid retrieval with selective filters to construct concise, high-signal context (top 8–12 chunks)
  - Include chunk_metadata with citation/jurisdiction/section/title for clearer answers and citations
- Phrasing for legal queries
  - Direct, scoped, include jurisdiction if possible (e.g., "Under NSW law, …")
- Token budgets
  - Keep context blocks reasonably sized; trim to the portions most relevant to the user’s question


## References

- fastapi_app.py: /search/rag, /search/oci_rag, /chat/agentic, /chat/conversation
- pages/chat.py: Streamlit UI for chat + retrieval + display
- db/store.py: search_hybrid/search_vector/search_fts helpers for building context upstream
- ingest/README.md: how context ends up in the DB with metadata for better grounding/citation
