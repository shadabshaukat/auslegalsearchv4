"""
Semantic, token-aware chunking utilities for AUSLegalSearch v3 (beta dataset).

Design goals:
- Modern, tokenizer-agnostic approximation of token counts for fast, dependency-light operation.
- Preserve semantic boundaries: headings -> paragraphs -> sentences, before hard-splitting.
- Configurable target token budget and overlap; defaults aligned with current industry practice (e.g., ~512 tokens, 10–20% overlap).
- Produce rich per-chunk metadata (e.g., section titles, indices) to improve retrieval and citation-grounded answers.
- Strict guarantees: chunk size never exceeds max_tokens after final splitting.

This module is standalone (no DB), focused purely on chunking logic.

Safety hardening:
- Uses the third-party 'regex' module (if installed) with per-call timeouts to avoid catastrophic backtracking hangs.
- Controlled by AUSLEGALSEARCH_REGEX_TIMEOUT_MS (default 200ms per regex operation).
- If timeout occurs in dashed-header parsing or sentence splitting, falls back to simplified, safe logic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

# Optional: safer regex engine with timeouts (pip install regex)
try:
    import regex as _re2  # type: ignore
except Exception:
    _re2 = None  # Fallback to stdlib re without timeouts

# Timeout (ms) for individual regex operations when using 'regex' module
_REGEX_TIMEOUT_MS = int(os.environ.get("AUSLEGALSEARCH_REGEX_TIMEOUT_MS", "200"))
_REGEX_TIMEOUT_S = max(0.01, _REGEX_TIMEOUT_MS / 1000.0)  # seconds; clamp to sane minimum

# Defaults tuned to common LLM context windows and RAG best-practices
DEFAULT_TARGET_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_MAX_TOKENS = 640  # safety upper bound

# Document-level metadata keys that must NOT be overwritten by section-level header meta.
# We keep these from the file frontmatter (base_meta) and map header 'title' into section_title instead.
PROTECTED_DOC_KEYS = {
    "title",
    "year",
    "date",
    "type",
    "jurisdiction",
    "subjurisdiction",
    "database",
    "url",
    "ext",
    "filename",
    "rel_path",
    "rel_path_no_years",
    "path_parts",
    "path_parts_no_years",
    "dataset_root",
    "court_guess",
    "jurisdiction_guess",
    "series_guess",
}


# ---- Tokenization and basic text utilities ----

# Prefer regex module with timeout-capable compiled patterns where applicable
def _compile(pattern: str, flags: int = 0):
    if _re2 is not None:
        return _re2.compile(pattern, flags)
    return re.compile(pattern, flags)

_WORD_RE = _compile(r"\w+|[^\w\s]")

def tokenize_rough(text: str) -> List[str]:
    """
    Lightweight tokenization: split into words and punctuation.
    Good proxy for transformer/BPE token counts without heavy dependencies.
    """
    return _WORD_RE.findall(text or "")

def count_tokens(text: str) -> int:
    return len(tokenize_rough(text))


# ---- Structural segmentation helpers ----

# Headings: Roman numerals / numbered / uppercase words / legal "Section" markers
_HEADING_LINE_RE = _compile(
    r"""
    ^\s*(
        (?:[IVXLCDM]+\.?)                                  # Roman numerals
        | (?:\d+(?:\.\d+)*\.?)                             # 1. or 1.2.3.
        | (?:[A-Z][A-Z0-9 ]{2,})                           # UPPERCASE headings
        | (?:Section\s+\d+[A-Za-z\-]*|s\.\s*\d+[A-Za-z\-]*) # Section 12 / s. 12
    )\b
    """,
    re.VERBOSE,
)

_SENT_SPLIT_RE = _compile(r"(?<=[.!?])\s+(?=[A-Z(])")

def split_into_sentences(text: str) -> List[str]:
    """
    Conservative sentence splitter. Avoids over-splitting abbreviations by requiring next token capital/opening bracket.
    With 'regex' installed, enforces a small timeout per split to avoid catastrophic backtracking.
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        if _re2 is not None and isinstance(_SENT_SPLIT_RE, _re2.Pattern):  # type: ignore[attr-defined]
            parts = _SENT_SPLIT_RE.split(text, timeout=_REGEX_TIMEOUT_S)  # type: ignore[arg-type]
        else:
            parts = _SENT_SPLIT_RE.split(text)
    except Exception:
        # Fallback: simple heuristic split on period + space
        parts = re.split(r"\.\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split on blank lines and normalize whitespace.
    """
    try:
        if _re2 is not None:
            parts = _re2.split(r"(?:\r?\n){2,}", text or "", timeout=_REGEX_TIMEOUT_S)  # type: ignore[attr-defined]
        else:
            parts = re.split(r"(?:\r?\n){2,}", text or "")
    except Exception:
        parts = (text or "").split("\n\n")
    paras = [p.strip() for p in parts if p.strip()]
    return paras

def split_into_blocks(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extract semantic blocks from text: a heading (if present) and its associated paragraph(s).
    Returns a list of (block_text, heading_title_or_none).
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    blocks: List[Tuple[str, Optional[str]]] = []

    current_heading: Optional[str] = None
    current_accum: List[str] = []

    def flush_block():
        nonlocal blocks, current_accum, current_heading
        if current_accum:
            block_text = "\n".join(current_accum).strip()
            if block_text:
                blocks.append((block_text, current_heading))
        current_accum = []

    for ln in lines:
        if ln.strip() == "":
            # paragraph boundary
            if current_accum:
                current_accum.append("")  # keep a blank to mark paragraph split
            continue

        is_heading = False
        try:
            if _re2 is not None and isinstance(_HEADING_LINE_RE, _re2.Pattern):  # type: ignore[attr-defined]
                is_heading = _HEADING_LINE_RE.match(ln.strip(), timeout=_REGEX_TIMEOUT_S) is not None  # type: ignore[arg-type]
            else:
                is_heading = _HEADING_LINE_RE.match(ln.strip()) is not None
        except Exception:
            # On matcher failure, treat as non-heading
            is_heading = False

        if is_heading:
            # heading encountered -> flush previous
            flush_block()
            current_heading = ln.strip()
            # Do not add heading itself to the block text; keep it as metadata
            continue

        current_accum.append(ln)

    flush_block()
    # Post-process: split any block on consecutive blanks to form tighter paragraph blocks
    refined: List[Tuple[str, Optional[str]]] = []
    try:
        splitter = _re2 if _re2 is not None else re
        parts_list = []
        for text_block, heading in blocks:
            parts = splitter.split(r"(?:\r?\n){2,}", text_block) if splitter is re else splitter.split(r"(?:\r?\n){2,}", text_block, timeout=_REGEX_TIMEOUT_S)  # type: ignore
            parts_list.append((parts, heading))
        for parts, heading in parts_list:
            for p in parts:
                p = p.strip()
                if p:
                    refined.append((p, heading))
    except Exception:
        # Fallback: keep the original blocks if paragraph split fails
        refined = blocks[:]
    return refined


# ---- Chunking core ----

@dataclass
class ChunkingConfig:
    target_tokens: int = DEFAULT_TARGET_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    max_tokens: int = DEFAULT_MAX_TOKENS
    min_sentence_tokens: int = 6   # drop ultra-short sentences unless necessary
    min_chunk_tokens: int = 40     # drop tiny chunks that don't add value

def _merge_sentences_to_chunks(
    sentences: List[str],
    cfg: ChunkingConfig
) -> List[List[str]]:
    """
    Merge sentences into chunks, aiming for target_tokens and allowing overlap.
    Always respects max_tokens by backoff-splitting long sentences if needed.
    Returns list of chunks as lists of sentences.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0

    def sentence_tokens(s: str) -> int:
        return count_tokens(s)

    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue
        stoks = sentence_tokens(sent)
        # If a single sentence exceeds max_tokens, hard-split it
        if stoks > cfg.max_tokens:
            # Hard split by approximate token size using whitespace slicing
            words = sent.split()
            piece_size = max(cfg.max_tokens - 5, cfg.min_chunk_tokens)
            acc = []
            w_acc = []
            for w in words:
                w_acc.append(w)
                if count_tokens(" ".join(w_acc)) >= piece_size:
                    acc.append(" ".join(w_acc))
                    w_acc = []
            if w_acc:
                acc.append(" ".join(w_acc))
            # Flush current first
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            # Each hard piece is its own chunk (no overlap)
            for piece in acc:
                chunks.append([piece])
            i += 1
            continue

        if current_tokens + stoks <= cfg.target_tokens or not current:
            current.append(sent)
            current_tokens += stoks
            i += 1
        else:
            # Emit current as a chunk
            chunks.append(current)
            # Prepare overlap for next chunk
            if cfg.overlap_tokens > 0:
                overlap: List[str] = []
                acc = 0
                # Walk backwards adding sentences until reach overlap_tokens
                for s in reversed(current):
                    t = sentence_tokens(s)
                    if acc + t <= cfg.overlap_tokens or not overlap:
                        overlap.append(s)
                        acc += t
                    else:
                        break
                overlap = list(reversed(overlap))
            else:
                overlap = []
            current = overlap.copy()
            current_tokens = sum(sentence_tokens(s) for s in current)

    if current:
        chunks.append(current)

    # Filter tiny chunks
    filtered: List[List[str]] = []
    for ch in chunks:
        toks = sum(count_tokens(s) for s in ch)
        if toks >= cfg.min_chunk_tokens or (len(chunks) == 1 and toks > 0):
            filtered.append(ch)
    return filtered

def chunk_text_semantic(
    text: str,
    cfg: Optional[ChunkingConfig] = None,
    section_title: Optional[str] = None,
    section_idx: Optional[int] = None
) -> List[Dict]:
    """
    Chunk a single block of text into token-aware, semantically bounded chunks.
    Returns a list of dicts with 'text' and 'chunk_metadata' keys.
    """
    cfg = cfg or ChunkingConfig()
    sentences = split_into_sentences(text)
    if not sentences:
        # Fallback: treat entire text as one sentence
        sentences = [text.strip()] if text and text.strip() else []

    # Remove ultra-short sentences (except if that would empty the set)
    if len(sentences) > 1:
        sentences = [s for s in sentences if count_tokens(s) >= cfg.min_sentence_tokens] or sentences

    sent_chunks = _merge_sentences_to_chunks(sentences, cfg)
    out: List[Dict] = []
    for idx, ch_sents in enumerate(sent_chunks):
        ch_text = " ".join(ch_sents).strip()
        if not ch_text:
            continue
        out.append({
            "text": ch_text,
            "chunk_metadata": {
                "section_title": section_title,
                "section_idx": section_idx,
                "chunk_idx": idx,
                "tokens_est": count_tokens(ch_text),
            }
        })
    return out


# ---- Document-level chunking ----

def detect_doc_type(meta: Optional[Dict], text: str) -> str:
    """
    Heuristic doc type detection: 'case', 'legislation', 'journal', or 'txt'.
    """
    meta = meta or {}
    t = (meta.get("type") or "").lower()
    if t in ("case", "legislation", "journal"):
        return t
    sample = (text or "")[:1000].lower()
    if " v " in sample or " appellant" in sample or " respondent" in sample:
        return "case"
    try:
        # simple heuristics; guard with try/except to avoid regex failures affecting flow
        if " section " in sample or re.search(r"\bs\.\s*\d+", sample):
            return "legislation"
        if re.search(r"^(?:[ivxlcdm]+\.)|^\d+\.", (text or "").strip().lower(), re.M):
            return "journal"
    except Exception:
        pass
    return "txt"

def chunk_document_semantic(
    doc_text: str,
    base_meta: Optional[Dict] = None,
    cfg: Optional[ChunkingConfig] = None
) -> List[Dict]:
    """
    Full semantic chunking pipeline:
    - Split into blocks keyed by headings (if any)
    - Chunk each block with token-aware logic
    - Attach per-chunk metadata with section titles and indices
    """
    cfg = cfg or ChunkingConfig()
    blocks = split_into_blocks(doc_text)
    chunks: List[Dict] = []
    sec_counter = 0
    for block_text, heading in blocks:
        block_chunks = chunk_text_semantic(block_text, cfg, section_title=heading, section_idx=sec_counter)
        for bc in block_chunks:
            md = dict(base_meta or {})
            cm = dict(bc.get("chunk_metadata") or {})
            md.update(cm)
            chunks.append({
                "text": bc["text"],
                "chunk_metadata": md
            })
        sec_counter += 1

    # Fallback: if no blocks produced anything, chunk whole doc
    if not chunks:
        for bc in chunk_text_semantic(doc_text, cfg, section_title=None, section_idx=None):
            md = dict(base_meta or {})
            cm = dict(bc.get("chunk_metadata") or {})
            md.update(cm)
            chunks.append({
                "text": bc["text"],
                "chunk_metadata": md
            })
    return chunks


# ---- Legislation dashed-header block parsing and chunking (for beta pipeline) ----

_DASH_LINE_RE = _compile(r"^\s*-{3,}\s*$", re.M)

def _parse_dashed_header(header_text: str) -> Dict[str, str]:
    """
    Parse key: value pairs inside a dashed header block.
    Example keys seen in data: title, regulation, chunk_id, section, etc.
    Returns a dict of lowercase keys -> stripped values.
    """
    meta: Dict[str, str] = {}
    for line in (header_text or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = (k or "").strip().lower()
        v = (v or "").strip()
        if k:
            meta[k] = v
    return meta

def parse_dashed_blocks(doc_text: str) -> List[Tuple[Dict[str, str], str]]:
    """
    Extract repeated dashed-header sections:
      -----\\n
      key: value\\n
      key: value\\n
      -----\\n
      <body until next dashed header or EOF>

    Returns list of (header_meta_dict, body_text).

    Timeout-safe implementation:
    - Uses 'regex' with timeout when available.
    - On timeout, falls back to a deterministic manual parser based on dashed-line positions.
    """
    text = doc_text or ""
    if not text.strip():
        return []

    # Pattern-based capture first (best effort with timeout)
    try:
        if _re2 is not None:
            pattern = _re2.compile(
                r"^\s*-{3,}\s*$\s*"
                r"(?P<header>.*?)"
                r"^\s*-{3,}\s*$\s*"
                r"(?P<body>.*?)(?=^\s*-{3,}\s*$|\Z)",
                _re2.M | _re2.S,
            )
            blocks: List[Tuple[Dict[str, str], str]] = []
            for m in pattern.finditer(text, timeout=_REGEX_TIMEOUT_S):  # type: ignore
                header_text = m.group("header") or ""
                body = (m.group("body") or "").strip()
                meta = _parse_dashed_header(header_text)
                if meta and body:
                    blocks.append((meta, body))
            if blocks:
                return blocks
        else:
            # stdlib re (no timeout)
            pattern = re.compile(
                r"^\s*-{3,}\s*$\s*"
                r"(?P<header>.*?)"
                r"^\s*-{3,}\s*$\s*"
                r"(?P<body>.*?)(?=^\s*-{3,}\s*$|\Z)",
                re.M | re.S,
            )
            blocks = []
            for m in pattern.finditer(text):
                header_text = m.group("header") or ""
                body = (m.group("body") or "").strip()
                meta = _parse_dashed_header(header_text)
                if meta and body:
                    blocks.append((meta, body))
            if blocks:
                return blocks
    except Exception:
        # Fallthrough to manual parser below
        pass

    # Manual parser based on dashed line positions (robust, no backtracking)
    try:
        if _re2 is not None and isinstance(_DASH_LINE_RE, _re2.Pattern):  # type: ignore[attr-defined]
            dashes = list(_DASH_LINE_RE.finditer(text, timeout=_REGEX_TIMEOUT_S))  # type: ignore[arg-type]
        else:
            dashes = list(_DASH_LINE_RE.finditer(text))
        if not dashes:
            return []

        # Expect pairs of dashed lines: [dash1, dash2] header between, body until next dash or EOF
        # We'll scan sequentially and build blocks
        blocks: List[Tuple[Dict[str, str], str]] = []
        i = 0
        L = len(dashes)
        while i + 1 < L:
            d1 = dashes[i]
            d2 = dashes[i + 1]
            header_text = text[d1.end():d2.start()]
            # Body from end of d2 to next dash (d3.start) or to end
            if i + 2 < L:
                d3 = dashes[i + 2]
                body = text[d2.end():d3.start()]
            else:
                body = text[d2.end():]
            header_meta = _parse_dashed_header(header_text or "")
            body = (body or "").strip()
            if header_meta and body:
                blocks.append((header_meta, body))
            i += 2
        return blocks
    except Exception:
        return []

def chunk_legislation_dashed_semantic(
    doc_text: str,
    base_meta: Optional[Dict] = None,
    cfg: Optional[ChunkingConfig] = None
) -> List[Dict]:
    """
    Legislation-aware chunking driven by dashed headers (title/regulation/chunk_id...).
    For each dashed block:
      - Parse header meta (lowercased keys)
      - Semantic chunk the body with chunk_text_semantic using token-aware config
      - Attach header meta + base_meta into chunk_metadata
    Also includes any preface text before the first dashed header (useful if an upstream loader stripped the first header).
    If no dashed blocks found, returns an empty list (caller may fall back to chunk_document_semantic).
    """
    cfg = cfg or ChunkingConfig()
    dashed_blocks = parse_dashed_blocks(doc_text)
    if not dashed_blocks:
        return []

    out: List[Dict] = []
    sec_idx = 0
    # Include any preface text that appears before the first dashed header (e.g., body of an initial header removed upstream)
    try:
        if _re2 is not None and isinstance(_DASH_LINE_RE, _re2.Pattern):  # type: ignore[attr-defined]
            m_first = _DASH_LINE_RE.search(doc_text or "", timeout=_REGEX_TIMEOUT_S)  # type: ignore[arg-type]
        else:
            m_first = _DASH_LINE_RE.search(doc_text or "")
        if m_first:
            preface = (doc_text or "")[: m_first.start()].strip()
            if preface:
                chs0 = chunk_text_semantic(preface, cfg, section_title=None, section_idx=sec_idx)
                for bc in chs0:
                    md0 = dict(base_meta or {})
                    cm0 = dict(bc.get("chunk_metadata") or {})
                    md0.update(cm0)
                    out.append({
                        "text": bc["text"],
                        "chunk_metadata": md0
                    })
                sec_idx += 1
    except Exception:
        # Preface inclusion should never break processing
        pass
    for header_meta, body in dashed_blocks:
        section_title = header_meta.get("title") or header_meta.get("section_title") or header_meta.get("section")
        # Chunk body semantically with heading context
        chs = chunk_text_semantic(body, cfg, section_title=section_title, section_idx=sec_idx)
        for bc in chs:
            md = dict(base_meta or {})
            # carry over section-level metadata derived from dashed header,
            # but DO NOT overwrite document-level keys (e.g., 'title', 'year', etc.)
            safe_header = {k: v for k, v in (header_meta or {}).items() if k not in PROTECTED_DOC_KEYS and k != "title"}
            md.update(safe_header)
            # also merge per-chunk metadata (tokens_est, chunk_idx, section_title/idx)
            cm = dict(bc.get("chunk_metadata") or {})
            # Ensure section_title is present from header 'title' if not already set
            if not cm.get("section_title") and (header_meta.get("title")):
                cm["section_title"] = header_meta.get("title")
            md.update(cm)
            out.append({
                "text": bc["text"],
                "chunk_metadata": md
            })
        sec_idx += 1
    return out


# ---- Generic RCTS (Recursive Character/Text) fallback chunker (optional) ----

def _build_rcts_splitter(cfg: Optional[ChunkingConfig]):
    """
    Try to build a LangChain RecursiveCharacterTextSplitter.

    Preference order:
    1) Token-aware splitter via tiktoken (from_tiktoken_encoder) if available.
    2) Character-based splitter approximating token sizes (4 chars per token heuristic).

    Returns (splitter_or_none, is_token_mode: bool)
    """
    cfg = cfg or ChunkingConfig()
    try:
        # Newer modular package name (if installed)
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter as _RCTS
        except Exception:
            # Fallback to classic import
            from langchain.text_splitter import RecursiveCharacterTextSplitter as _RCTS  # type: ignore

        # Try token-aware mode first (requires tiktoken or supported tokenizer)
        try:
            splitter = _RCTS.from_tiktoken_encoder(
                chunk_size=int(cfg.target_tokens),
                chunk_overlap=int(cfg.overlap_tokens),
            )
            return splitter, True
        except Exception:
            # Fallback to character-based approximating token budgets
            chunk_size_chars = max(1, int(cfg.target_tokens * 4))
            overlap_chars = max(0, int(cfg.overlap_tokens * 4))
            splitter = _RCTS(
                chunk_size=chunk_size_chars,
                chunk_overlap=overlap_chars,
                separators=["\n\n", "\n", " ", ""],
                length_function=len,
            )
            return splitter, False
    except Exception:
        # LangChain not installed or failed to import
        return None, False


def chunk_generic_rcts(
    doc_text: str,
    base_meta: Optional[Dict] = None,
    cfg: Optional[ChunkingConfig] = None
) -> List[Dict]:
    """
    Generic fallback chunker using LangChain RecursiveCharacterTextSplitter (if available).

    Behavior:
    - If an initial dashed header block exists, parse it and treat its key:value pairs
      as file-level header metadata to attach to ALL produced chunks.
    - RCTS splitting is applied to the body text after the first dashed header.
      If no dashed header exists, RCTS is applied to the entire document.
    - Token budgets:
        * If tiktoken is available: chunk_size/overlap are in tokens (cfg.target_tokens/overlap_tokens).
        * Else: approximate via ~4 chars/token for chunk_size/overlap.

    Returns empty list if LangChain is not available, so caller can fall back to
    chunk_document_semantic or other strategies.
    """
    cfg = cfg or ChunkingConfig()
    splitter, is_token_mode = _build_rcts_splitter(cfg)
    if splitter is None:
        # Signal to caller that RCTS is unavailable
        return []

    text = (doc_text or "").strip()
    if not text:
        return []

    # Attempt to parse a single initial dashed header block for metadata
    header_meta: Dict[str, str] = {}
    body_text = text
    try:
        blocks = parse_dashed_blocks(text)
        if blocks:
            # Take the first dashed block as the file-level header
            header_meta = dict(blocks[0][0] or {})
            body_text = (blocks[0][1] or "").strip()
            if not body_text:
                # If empty body for some reason, revert to full text
                body_text = text
    except Exception:
        # Non-critical; continue without header meta
        header_meta = {}
        body_text = text

    # Split using RCTS
    pieces = splitter.split_text(body_text or "")
    out: List[Dict] = []
    sec_title = header_meta.get("title") or header_meta.get("section_title") or header_meta.get("section")
    for idx, piece in enumerate(pieces):
        p = (piece or "").strip()
        if not p:
            continue
        md = dict(base_meta or {})
        if header_meta:
            # do not clobber document-level keys; map header 'title' into section context only
            safe_header = {k: v for k, v in (header_meta or {}).items() if k not in PROTECTED_DOC_KEYS and k != "title"}
            md.update(safe_header)
        md.update({
            "section_title": sec_title,
            "section_idx": 0,
            "chunk_idx": idx,
            "tokens_est": count_tokens(p),
            "rcts_token_mode": bool(is_token_mode),
            "strategy": "rcts-generic"
        })
        out.append({
            "text": p,
            "chunk_metadata": md
        })
    return out
