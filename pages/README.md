# AUSLegalSearch v3 — Streamlit UI (pages)

Interactive Streamlit UI for login and legal chat with Retrieval Augmented Generation (RAG). The UI exposes a clean chat experience with per-source cards, metadata, and controls for LLM decoding parameters.

Files
- pages/login.py — Email/password login and basic account creation (stored in Postgres)
- pages/chat.py — Chat experience with hybrid retrieval and RAG generation, showing Top 10 retrieved chunks and citations


## Overview

- Authentication
  - Local email/password login and signup forms backed by Postgres (`db.store`).
  - Session state key `user` gates access to the chat page:
    - If `user` not set, `pages/chat.py` forces a redirect to the login page and stops execution.

- Chat workflow
  - Hybrid retrieval (BM25 + vector) using `db.store.search_bm25` and `db.store.search_vector` for each user question.
  - Displays Top 10 retrieved chunks with:
    - Clickable header (URL if present in metadata)
    - Score
    - Visible metadata dict (excluding text/source/score/url)
    - Truncated chunk text body
  - Uses local `rag.RAGPipeline` (Ollama) to compose a prompt with context chunks and a customizable system prompt.
  - Streams the LLM answer into the UI and shows the retrieved chunks underneath.

- Session saving
  - When the user clicks "Start New Chat Session", the current chat history and LLM params are persisted via `db.store.save_chat_session(...)` to support audit and reproducibility.


## Requirements

- Python 3.10+
- Dependencies (install once):
```bash
pip install -r requirements.txt
```

- Database (for auth and chat logs)
  - Configure Postgres connection (either AUSLEGALSEARCH_DB_URL or the AUSLEGALSEARCH_DB_HOST/PORT/USER/PASSWORD/NAME variables).
  - The DB schema is created automatically by API on startup when `AUSLEGALSEARCH_AUTO_DDL=1` or can be created manually via `db.store.create_all_tables()`.

- Embedding model (for vector retrieval in UI)
  - The chat page calls `embedding.Embedder()` to embed queries for semantic retrieval.
  - Ensure the embedding model and DB embedding dimension are aligned:
    - `AUSLEGALSEARCH_EMBED_MODEL` (default: nomic-ai/nomic-embed-text-v1.5)
    - `AUSLEGALSEARCH_EMBED_DIM` (default 768; must match the model used during ingestion)


## Running the UI

- Multipage Streamlit app (recommended):
  - If `app.py` is your Streamlit entry (common pattern), run:
    ```bash
    streamlit run app.py
    ```
  - Streamlit will list available pages in the sidebar (Login, Chat).

- Direct page runs (for development):
  - Login page only:
    ```bash
    streamlit run pages/login.py
    ```
  - Chat page only (requires session_state["user"] set; otherwise will redirect/stop):
    ```bash
    streamlit run pages/chat.py
    ```

If running behind a reverse proxy, configure Streamlit’s base URL and CORS/security settings accordingly.


## pages/login.py

- Purpose
  - Provides a minimal login/signup experience.
  - Hides the Streamlit sidebar via injected CSS for a cleaner login experience.

- Functions
  - `do_signup()`: Creates a user with `db.store.create_user(...)`, validates uniqueness, handles exceptions.
  - `do_login()`: Validates credentials via `db.store.get_user_by_email` and `db.store.check_password`, sets `session_state["user"]`, and redirects to the chat page.

- Google login placeholder
  - UI displays guidance on integrating Google OAuth.
  - Implementation is intentionally left as a placeholder. Recommended approach:
    - Use `streamlit-authenticator` or `streamlit-oauth`
    - Configure OAuth Client ID/Secret
    - Map Google email to the `users` table (set `registered_google=True`, store `google_id` if used)

- Notes
  - On successful email/password login, `set_last_login(user.id)` is called for auditing.
  - On signup, passwords are hashed using bcrypt (via `db.store.hash_password`).


## pages/chat.py

- Purpose
  - Provides a RAG chat experience with:
    - Hybrid retrieval (BM25 + vector) on each user query
    - Streaming answer rendering
    - Visual cards for the Top 10 retrieved chunks and the Top 10 semantic-only results
    - Controls for temperature, top_p, max_tokens, repeat_penalty, and max sources

- Key functions
  - `hybrid_search(query, top=10)`:
    - Combines `search_bm25` and `search_vector` results, deduplicates by `(source, span)`, sorts by score, returns top-k list and separate vector-only results.
  - `get_recent_chat_history_str(max_turns=3)`:
    - Summarizes the last few turns for inclusion in the system prompt to create a contextual conversation.
  - `rag_llm(...)`:
    - Calls `RAGPipeline(model=...)` with the current context chunks and a composed system prompt; then streams the output to the UI.
  - `show_chunks(chunks, vec_hits)`:
    - Renders Top 10 retrieved chunks (and optionally Top 10 vector-only hits) as rich cards, displaying URL (if present), score, metadata, and truncated text.

- UI controls
  - In the sidebar:
    - Temperature, Top-p, Max tokens, Repeat penalty
    - Max sources per answer (top_k)
    - Session system prompt (editable text area)
  - Model selection:
    - Populates models via `rag.list_ollama_models()` (REST call to `http://localhost:11434/api/tags`)
    - Defaults to `"llama3"` if none listed

- Session persistence
  - On "Start New Chat Session", persists:
    - chat_history
    - llm_params (model, decoding params, custom prompt, top_k)
    - username (from `session_state["user"]`)
    - first user question in history


## Security and operations

- Protect the Streamlit app behind a reverse proxy (Nginx) and TLS (e.g., Let’s Encrypt).
- Hide the app from public access unless required; a WAF or IP allowlist is recommended for non-public deployments.
- Store all secrets in environment variables or secret managers, not in code or repo.
- Ensure DB access (login/signup) is rate-limited at the proxy (if Internet-exposed).
- Use strong password policies and consider adding multi-factor auth in enterprise environments.


## Troubleshooting

- Login succeeds but redirect fails
  - `st.switch_page` exists only in newer Streamlit; fallback logic uses `rerun`. Ensure you’re running a current Streamlit version.
- “You must login to continue” on the chat page
  - `session_state["user"]` not set. Login first via pages/login.py, or mock a user in development:
    ```python
    st.session_state["user"] = {"email": "dev@local", "id": 1}
    ```
- No models in the Ollama dropdown
  - Verify Ollama is running and reachable at `http://localhost:11434`
  - Install models with `ollama pull llama3` (or other)
- Vector retrieval returns no results
  - Ensure embeddings exist in the DB and that `AUSLEGALSEARCH_EMBED_DIM` matches the model dimension
  - Confirm DB connectivity and that `create_all_tables()` ran at least once


## References

- db/README.md — DB setup and search helpers
- rag/README.md — RAG pipelines for Ollama and OCI GenAI
- ingest/README.md — Ingestion pipeline (how content+metadata are produced)
- tools/README.md — SQL benchmarking (validate index coverage/tuning)
