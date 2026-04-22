"""
Embedding interface for auslegalsearchv3.

- Primary path: Sentence-Transformers models (default: nomic-ai/nomic-embed-text-v1.5, 768d)
- Fallback path: Plain HuggingFace Transformer checkpoints (e.g., 'legal-bert-base-uncased') with pooling
  when they are NOT sentence-transformers repos.

- Model selection order:
  1) Explicit 'model_name' argument (if provided)
  2) Environment variable AUSLEGALSEARCH_EMBED_MODEL
  3) DEFAULT_MODEL ('nomic-ai/nomic-embed-text-v1.5')

- Provides embed(texts) method returning ndarray [batch, dim]
- Exposes .dimension attribute (embedding dimensionality)

Notes:
- 'legal-bert-base-uncased' is a plain BERT checkpoint, not a sentence-transformers repo.
  This module supports it via a HF fallback with mean pooling over token embeddings.
- Max sequence length for HF fallback can be controlled with AUSLEGALSEARCH_EMBED_MAXLEN (default 512).
"""

from typing import List
import numpy as np
import os
import warnings

# Try to import SentenceTransformers for ST-native models
try:
    from sentence_transformers import SentenceTransformer
    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
except ImportError:
    SentenceTransformer = None  # We can still use HF fallback
    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Lazily imported only when we use the HF fallback
_HF_IMPORTED = False
def _ensure_hf_imports():
    global _HF_IMPORTED, AutoTokenizer, AutoModel, torch
    if not _HF_IMPORTED:
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        _HF_IMPORTED = True

def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)

class Embedder:
    def __init__(self, model_name: str = None):
        """
        Initialize embedder. Resolution order:
          1) model_name arg
          2) AUSLEGALSEARCH_EMBED_MODEL env var
          3) DEFAULT_MODEL

        Attempts SentenceTransformer first; if load fails (or model isn't ST),
        falls back to HuggingFace AutoModel + mean pooling.
        """
        resolved = model_name or os.environ.get("AUSLEGALSEARCH_EMBED_MODEL") or DEFAULT_MODEL
        self.model_name = resolved
        self.use_hf_fallback = False
        self._st_model = None
        self._hf_model = None
        self._hf_tokenizer = None
        self.dimension = None

        # Determine if we should pass trust_remote_code to ST loader
        trust_remote = False
        flags = os.environ.get("AUSLEGALSEARCH_EMBEDDER_FLAGS", "")
        if "trust_remote_code" in flags or ("nomic-ai" in resolved):
            trust_remote = True
        if os.environ.get("AUSLEGALSEARCH_TRUST_REMOTE_CODE", "0") == "1":
            trust_remote = True
        # Optional: pin revision and run offline from local cache
        rev = os.environ.get("AUSLEGALSEARCH_EMBED_REV", None)
        local_only = os.environ.get("AUSLEGALSEARCH_HF_LOCAL_ONLY", "0") == "1"

        # Try SentenceTransformer path first (if available)
        if SentenceTransformer is not None:
            try:
                st_kwargs = {"trust_remote_code": trust_remote}
                if rev:
                    st_kwargs["revision"] = rev
                if local_only:
                    st_kwargs["local_files_only"] = True
                try:
                    self._st_model = SentenceTransformer(resolved, **st_kwargs)
                except TypeError:
                    # Older sentence_transformers versions may not support revision/local_files_only
                    self._st_model = SentenceTransformer(resolved, trust_remote_code=trust_remote)
                self.dimension = int(self._st_model.get_sentence_embedding_dimension())
            except Exception as e:
                warnings.warn(f"SentenceTransformer load failed for '{resolved}', falling back to HF: {e}")
                self._init_hf_fallback(resolved, trust_remote)
        else:
            # No ST installed; go straight to HF fallback
            self._init_hf_fallback(resolved, trust_remote)

        # Final sanity check
        if self.dimension is None:
            raise RuntimeError(f"Could not determine embedding dimension for model '{resolved}'")

    def _init_hf_fallback(self, model_name: str, trust_remote: bool = False):
        """
        Initialize HuggingFace AutoModel fallback with mean pooling.
        Works for plain encoder checkpoints like 'legal-bert-base-uncased'.
        """
        _ensure_hf_imports()
        try:
            rev = os.environ.get("AUSLEGALSEARCH_EMBED_REV", None)
            local_only = os.environ.get("AUSLEGALSEARCH_HF_LOCAL_ONLY", "0") == "1"
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote, revision=rev, local_files_only=local_only)
            self._hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote, revision=rev, local_files_only=local_only)
            # BERT-base hidden size is typically 768; grab from config
            hidden = getattr(self._hf_model.config, "hidden_size", None)
            if hidden is None:
                raise ValueError("HF model has no 'hidden_size' in config")
            self.dimension = int(hidden)
            self.use_hf_fallback = True
        except Exception as e:
            raise RuntimeError(f"HF fallback load failed for '{model_name}': {e}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts; returns ndarray [batch, dim].
        - ST path: uses SentenceTransformer.encode
        - HF fallback: mean-pools last_hidden_state with attention_mask
        """
        if not texts:
            return np.zeros((0, int(self.dimension or 0)), dtype=np.float32)

        if not self.use_hf_fallback and self._st_model is not None:
            # SentenceTransformers path
            vecs = self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
            # Some ST models already normalize; we leave as-is to avoid double-normalization.
            # If you want to enforce, uncomment:
            # vecs = _l2_normalize(vecs)
            return np.asarray(vecs, dtype=np.float32)

        # HF fallback path
        _ensure_hf_imports()
        max_len = int(os.environ.get("AUSLEGALSEARCH_EMBED_MAXLEN", "512"))
        # Tokenize as a batch
        with torch.no_grad():  # type: ignore
            toks = self._hf_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            outputs = self._hf_model(**toks)  # type: ignore
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            mask = toks["attention_mask"].unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
            # Mean pooling
            summed = (last_hidden * mask).sum(dim=1)           # [B, H]
            counts = mask.sum(dim=1).clamp(min=1e-9)           # [B, 1]
            pooled = summed / counts                           # [B, H]
            vecs = pooled.cpu().numpy().astype(np.float32)
            # Optional: normalize for cosine similarity stability with pgvector
            # vecs = _l2_normalize(vecs).astype(np.float32)
            return vecs

"""
Usage examples:

# Default (Nomic)
embedder = Embedder()
vecs = embedder.embed(["example legal text 1", "example legal text 2"])
print(vecs.shape)  # (batch_size, 768)

# Explicit model (Sentence-Transformers repo)
embedder = Embedder("maastrichtlawtech/bge-legal-en-v1.5")

# Plain HF checkpoint (fallback), e.g., Legal-BERT base
embedder = Embedder("nlpaueb/legal-bert-base-uncased")  # will use AutoModel + mean pooling (dim=768)

# Env-based selection:
# export AUSLEGALSEARCH_EMBED_MODEL="legal-bert-base-uncased"
# python -c "from embedding.embedder import Embedder; print(Embedder().dimension)"
"""
