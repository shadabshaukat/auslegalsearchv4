# AUSLegalSearchv3 REST API Specification
**For any frontend, including advanced Gradio UIs, to consume backend model search & chat services**

## Authentication

**All endpoints require HTTP Basic Auth**:
- Username: `legal_api` (default, update as needed)
- Password: `letmein`  (default, update as needed)

All requests must include:

```
-H "Authorization: Basic [base64-encoded-credentials]"
```
Example:  
For username `legal_api` and password `letmein`, the header is:
```
-H "Authorization: Basic bGVnYWxfYXBpOmxldG1laW4="
```

----

## 1. Hybrid Search

POST `/search/hybrid`

Description:  
Hybrid (vector + BM25 keyword) search with reranker. Returns context-relevant legal document chunks, citations, and metadata.

### Request
```json
{
  "query": "What is Mabo v Queensland?",
  "top_k": 5,
  "alpha": 0.5           // Between 0 (keyword-most) and 1 (semantic vector-most)
}
```

### Example curl
```bash
curl -u legal_api:letmein \
  -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Mabo v Queensland?", "top_k": 5, "alpha": 0.5}'
```

### Response
```json
[
  {
    "doc_id": 12,
    "chunk_index": 0,
    "hybrid_score": 0.85,
    "citation": "Mabo v Queensland (No 2) [1992] HCA 23",
    "score": 0.81,
    "vector_score": 0.76,
    "bm25_score": 0.71,
    "text": "The Mabo v Queensland decision recognized native title rights...",
    "source": "cases/mabo.txt",
    "format": "txt",
    "chunk_metadata": {
      "section": "DECISION",
      "url": "https://example.com/case/mabo"
    }
  }
]
```
----

## 2. Vector Search

POST `/search/vector`

Description:  
Semantic embedding (vector) search retrieving top-k relevant text chunks.

### Request

```json
{
  "query": "native title Australia",
  "top_k": 5
}
```

### Example curl

```bash
curl -u legal_api:letmein \
  -X POST http://localhost:8000/search/vector \
  -H "Content-Type: application/json" \
  -d '{"query": "native title Australia", "top_k": 5}'
```

### Response

```json
[
  {
    "doc_id": 3,
    "chunk_index": 5,
    "score": 0.77,
    "text": "Native title is a common law concept recognizing traditional Indigenous interests...",
    "chunk_metadata": {
      "section": "SUMMARY",
      "citation": "NativeTitleSummary",
      "url": "https://example.com/nt-summary"
    }
  }
]
```
----

## 3. RAG (Retrieval Augmented Generation)

POST `/search/rag`

Description:  
Retrieves top-k legal chunks and generates an answer using LLM (Ollama/local) with citations.
Supports all LLM generation params and system prompt.

### Request

```json
{
  "question": "Summarize Mabo v Queensland's outcome.",
  "model": "llama3",       // Ollama model
  "context_chunks": [ ... ],  // Optional: hybrid/vector search results
  "sources": [ ... ],      // Optional
  "chunk_metadata": [ ... ], // Optional
  "custom_prompt": "You are an expert legal AI ...",  // Optional (system prompt)
  "temperature": 0.1,
  "top_p": 0.9,
  "max_tokens": 1024,
  "repeat_penalty": 1.1,
  "reranker_model": "mxbai-rerank-xsmall"
}
```

### Example curl

```bash
curl -u legal_api:letmein \
  -X POST http://localhost:8000/search/rag \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize Mabo v Queensland","model":"llama3","temperature":0.1,"max_tokens":1024}'
```

### Response

```json
{
  "answer": "The Mabo v Queensland (No 2) decision recognized native title for Indigenous Australians. (Mabo v Queensland (No 2) [1992] HCA 23)",
  "sources": ["Mabo v Queensland (No 2) [1992] HCA 23"],
  "contexts": ["...top legal chunks included in answer..."],
  "chunk_metadata": [...]
}
```
----

## 4. OCI GenAI RAG (Retrieval Augmented Generation for Oracle GenAI)

POST `/search/oci_rag`

Description:  
Same as RAG, but powered by Oracle Cloud GenAI models (backend will choose `chat` endpoint).

### Request

```json
{
  "oci_config": {
    "compartment_id": "...",
    "model_id": "...",
    "region": "us-chicago-1"
  },
  "question": "Explain Wik decision on native title",
  "context_chunks": [ ... ],
  "sources": [ ... ],
  "chunk_metadata": [ ... ],
  "custom_prompt": "...",
  "temperature": 0.1,
  "top_p": 0.9,
  "max_tokens": 1024,
  "repeat_penalty": 1.1
}
```

### Example curl

```bash
curl -u legal_api:letmein \
  -X POST http://localhost:8000/search/oci_rag \
  -H "Content-Type: application/json" \
  -d @oci_rag_req.json
```

_Response fields/format same as /search/rag_

----

## 5. Conversational Chat (Multi-turn with Memory, RAG + LLM reasoning)

POST `/chat/conversation`

Description:  
Conversational, multi-turn chat endpoint. Does hybrid search per-turn and injects context/sources for the answer.
Supports both Ollama (local) and OCI GenAI backends.

### Request

```json
{
  "llm_source": "ollama",           // or "oci_genai"
  "model": "llama3",                // for Ollama
  "message": "Explain Wik case.",
  "chat_history": [
      {"role": "user", "content": "What is Mabo?"},
      {"role": "assistant", "content": "Mabo v Queensland (No 2) ..."} // prior turns
  ],
  "top_k": 8,
  "system_prompt": "You are an expert Australian legal assistant...",
  "temperature": 0.15,
  "top_p": 0.91,
  "max_tokens": 700,
  "repeat_penalty": 1.11,
  "oci_config": {
    "compartment_id": "...",
    "model_id": "...",
    "region": "us-chicago-1"
  }
}
```

### Example curl

```bash
curl -u legal_api:letmein \
  -X POST http://localhost:8000/chat/conversation \
  -H "Content-Type: application/json" \
  -d @chat_convo_req.json
```

### Response

```json
{
  "answer": "The Wik Peoples v Queensland case clarified native title rights in Australia...",
  "sources": [
    "Wik Peoples v Queensland [1996] HCA 40"
  ],
  "context_chunks": [
    "Key chunk 1 ...",
    "Key chunk 2 ..."
  ],
  "chunk_metadata": [
    { "section": "DECISION", ... }
  ]
}
```

## 6. Additional API endpoints

Other endpoints such as /models/ollama, /models/oci_genai, /search/vector, etc., are available for populating UI dropdowns and auxiliary functions.
All follow similar request/response patterns. See main backend source for schema.

----

## Notes

- For all POST requests, set `Content-Type: application/json`.
- All endpoints require Basic Auth.
- Model/compartment values can be retrieved using GET endpoints (use browser or authenticated curl).

----

## Status Codes

- 200 OK: Success
- 400 Bad Request: Input missing or invalid
- 401 Unauthorized: Auth missing/incorrect
- 500 Internal Server Error: Backend/system/model error

----

**For more sample responses and exact schema, see the main backend source or test interactively with curl/Postman or your Gradio UI.**
