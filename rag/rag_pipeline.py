"""
RAG pipeline for auslegalsearchv2/v3.
- Retrieves relevant documents/chunks from the vector store.
- Sends context, user question, and options to Ollama Llama3/Llama4 via API.
- Returns model output (QA answer/summary) and relevant document sources.
- ENHANCED: Adds support for rich chunk metadata (legal meta) in prompt/context and in returned sources.
"""

import requests

def list_ollama_models(ollama_url="http://localhost:11434"):
    # Use only the Ollama REST API to get models.
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            return [m["name"] for m in result.get("models", [])]
        else:
            return []
    except Exception:
        return []

class RAGPipeline:
    def __init__(self, ollama_url="http://localhost:11434", model="llama3"):
        self.ollama_url = ollama_url
        self.model = model

    def retrieve(self, query: str, k: int = 5):
        # Placeholder
        contexts = ["Relevant chunk 1...", "Relevant chunk 2..."]
        sources = ["source_1.txt", "source_2.html"]
        metadata = [None for _ in contexts]
        return contexts, sources, metadata

    def _generate_context_block(self, text, chunk_meta):
        """Format chunk+metadata for LLM context."""
        context = ""
        if chunk_meta and isinstance(chunk_meta, dict) and len(chunk_meta) > 0:
            for k, v in chunk_meta.items():
                context += f"{k}: {v}\n"
            context += "---\n"
        context += text
        return context

    def llama4_rag(
        self, query: str, context_chunks, chunk_metadata=None, custom_prompt=None,
        temperature=0.2, top_p=0.95, max_tokens=1024, repeat_penalty=1.1
    ) -> str:
        if custom_prompt:
            sys_prompt = custom_prompt.strip()
        else:
            sys_prompt = "You are a legal assistant. Answer only from the provided context. Cite sources. Be concise."

        if chunk_metadata is None:
            blocks = [self._generate_context_block(text, None) for text in context_chunks]
        else:
            blocks = [self._generate_context_block(text, meta) for text, meta in zip(context_chunks, chunk_metadata)]
        prompt = (
            sys_prompt + "\n\n"
            "Based on the following legal documents/chunks, answer the question or summarize as requested.\n"
            "CONTEXT:\n"
            + "\n\n---\n\n".join(blocks) +
            f"\n\nQUESTION: {query}\nANSWER:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "repeat_penalty": repeat_penalty,
            },
        }
        resp = requests.post(
            f"{self.ollama_url}/api/generate", json=payload, timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            return f"Error querying Llama4: {resp.status_code} {resp.text}"

    def query(
        self, question: str, top_k: int = 5, context_chunks=None, sources=None, chunk_metadata=None, custom_prompt=None,
        temperature=0.2, top_p=0.95, max_tokens=1024, repeat_penalty=1.1, chat_history=None
    ) -> dict:
        # Optionally receive chat_history; for now do not use it in the prompt, but unblocks conversational calls from backend.
        if context_chunks is not None:
            contexts = context_chunks
            metas = chunk_metadata if chunk_metadata is not None else [None for _ in contexts]
        else:
            contexts, sources, metas = self.retrieve(question, k=top_k)
        answer = self.llama4_rag(
            question,
            contexts,
            chunk_metadata=metas,
            custom_prompt=custom_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
        )
        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "chunk_metadata": metas,
        }
