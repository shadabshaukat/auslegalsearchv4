"""
Loader for auslegalsearchv3: Nomic 768D Embedding, 1500-char Chunking
- All legal/journal/case/generic chunking uses ~1500-char target.
- If any chunk >1500 chars after paragraph chunking, further split chunk at sentence/end boundary or substring.

Chunking rules:
- Prefer not to split paragraphs, but will split internally if a single paragraph chunk exceeds 1500 chars.
- Absolute guarantee: every chunk ≤1500 characters.
"""

import os
import re
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple

from bs4 import BeautifulSoup
import numpy as np
from ast import literal_eval

SUPPORTED_EXTS = {'.txt', '.html'}

def walk_legal_files(root_dirs: list[str]) -> Iterator[str]:
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_EXTS:
                    yield os.path.join(dirpath, fname)

def extract_metadata_block(text: str) -> tuple[dict, str]:
    lines = text.lstrip().splitlines()
    chunk_metadata = {}
    if len(lines) >= 3 and lines[0].startswith("-") and lines[1].find(":") != -1:
        try:
            start_idx = 0
            while start_idx < len(lines) and not lines[start_idx].startswith("-"):
                start_idx += 1
            end_idx = start_idx + 1
            while end_idx < len(lines) and not lines[end_idx].startswith("-"):
                end_idx += 1
            for l in lines[start_idx+1:end_idx]:
                if ":" in l:
                    k, v = l.split(":", 1)
                    k, v = k.strip(), v.strip()
                    # Safe parsing (no eval):
                    # - Try Python literals (ints, floats, lists, dicts, booleans, None)
                    # - If that fails, keep the original string
                    try:
                        val = literal_eval(v)
                    except Exception:
                        val = v
                    chunk_metadata[k] = val
            body_text = "\n".join(lines[end_idx+1:]).strip()
            return chunk_metadata, body_text
        except Exception:
            return {}, text
    return {}, text

def parse_txt(filepath: str) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        meta, body = extract_metadata_block(text)
        detected_type = (meta.get("type", "") or "").strip().lower()
        return {
            "text": body,
            "source": filepath,
            "format": detected_type if detected_type else "txt",
            "chunk_metadata": meta or None
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def parse_html(filepath: str) -> dict:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        raw_text = soup.get_text(separator='\n', strip=True)
        meta, body = extract_metadata_block(raw_text)
        detected_type = (meta.get("type", "") or "").strip().lower()
        return {
            "text": body,
            "source": filepath,
            "format": detected_type if detected_type else "html",
            "chunk_metadata": meta or None
        }
    except Exception as e:
        print(f"Error parsing HTML {filepath}: {e}")
        return {}

def split_chunk_hard(chunk: str, max_length: int = 1500) -> list:
    """If any chunk > max_length, break it up further—prefer periods, else break every max_length."""
    if len(chunk) <= max_length:
        return [chunk]
    sentences = re.split(r'(?<=[.?!]) +', chunk)
    out = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_length:
            current = f"{current} {sent}".strip() if current else sent
        else:
            if current:
                out.append(current.strip())
            if len(sent) > max_length:
                for i in range(0, len(sent), max_length):
                    out.append(sent[i : i + max_length].strip())
                current = ""
            else:
                current = sent
    if current.strip():
        out.append(current.strip())
    return out

def chunk_by_paragraphs(text: str, target_chars: int = 1500) -> list[str]:
    paras = [p for p in re.split(r'(?:\r?\n){2,}', text) if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paras:
        if not current_chunk:
            current_chunk = para.strip()
        elif len(current_chunk) + len(para) + 2 <= target_chars:
            current_chunk = f"{current_chunk}\n\n{para.strip()}"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para.strip()
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    # HARD SPLIT step: if any chunk > 1500 chars, break up at sentence/end
    final = []
    for chunk in chunks:
        final.extend(split_chunk_hard(chunk, max_length=1500))
    return final

def chunk_legislation(doc: dict, min_chunk_length: int = 150) -> list[dict]:
    text = doc.get("text", "")
    meta = doc.get("chunk_metadata", {})
    srcpath = doc.get("source", "")
    section_pattern = re.compile(
        r"-{5,}\s*\n\s*section:\s*([^\n]+)\n\s*title:\s*([^\n]+)\n-{5,}\s*\n(.*?)(?=(?:-{5,}\s*\n\s*section:)|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = section_pattern.finditer(text)
    sections = []
    for idx, match in enumerate(matches):
        sec_num = match.group(1).strip()
        sec_title = match.group(2).strip()
        sec_text = match.group(3).strip()
        for subchunk in chunk_by_paragraphs(sec_text, target_chars=1500):
            if len(subchunk.strip()) >= min_chunk_length:
                section_meta = {**meta, "section": sec_num, "section_title": sec_title, "section_idx": idx}
                sections.append({
                    "text": subchunk,
                    "source": srcpath,
                    "format": "legislation",
                    "chunk_metadata": section_meta,
                    "legal_section": sec_title or sec_num,
                })
    # Fallback: If any chunk >1500 chars, or if no chunks, do paragraph chunking (with hard split) for the whole doc
    if not sections or any(len(chunk['text']) > 1500 for chunk in sections):
        para_chunks = chunk_by_paragraphs(text, target_chars=1500)
        out = []
        for idx, ch in enumerate(para_chunks):
            if len(ch.strip()) >= min_chunk_length:
                out.append({
                    "text": ch.strip(),
                    "source": srcpath,
                    "format": "legislation",
                    "chunk_metadata": meta,
                    "legal_section": meta.get("title", f"para_{idx}")
                })
        return out
    return sections

def chunk_journal(doc: dict, min_chunk_length: int = 150) -> list[dict]:
    meta = doc.get("chunk_metadata", {})
    srcpath = doc.get("source", "")
    text = doc.get("text", "")
    lines = text.splitlines()
    chunks = []
    buffer = ""
    last_heading = ""
    heading_pattern = re.compile(r"^([IVXLCDM]+\.|[0-9]+\.|[A-Z]\.)\s+.*")
    for line in lines:
        line_stripped = line.strip()
        if heading_pattern.match(line_stripped):
            if buffer.strip() and len(buffer.strip()) >= min_chunk_length:
                chunks.append({
                    "text": buffer.strip(),
                    "source": srcpath,
                    "format": "journal",
                    "chunk_metadata": meta,
                    "legal_section": last_heading,
                })
            last_heading = line_stripped
            buffer = last_heading + "\n"
        else:
            buffer += line.strip() + " "
    if buffer.strip() and len(buffer.strip()) >= min_chunk_length:
        chunks.append({
            "text": buffer.strip(),
            "source": srcpath,
            "format": "journal",
            "chunk_metadata": meta,
            "legal_section": last_heading,
        })
    # Force fallback if any chunk >1500 chars, or if no chunks
    if not chunks or any(len(chunk['text']) > 1500 for chunk in chunks):
        para_chunks = chunk_by_paragraphs(text, target_chars=1500)
        out = []
        for idx, ch in enumerate(para_chunks):
            if len(ch.strip()) >= min_chunk_length:
                out.append({
                    "text": ch.strip(),
                    "source": srcpath,
                    "format": "journal",
                    "chunk_metadata": meta,
                    "legal_section": f"para_{idx}"
                })
        return out
    return chunks

def chunk_case(doc: dict, min_chunk_length: int = 150) -> list[dict]:
    meta = doc.get("chunk_metadata", {})
    srcpath = doc.get("source", "")
    text = doc.get("text", "")
    para_chunks = chunk_by_paragraphs(text, target_chars=1500)
    chunks = []
    for idx, para in enumerate(para_chunks):
        if len(para.strip()) >= min_chunk_length:
            section_meta = {**meta, "section_idx": idx}
            chunks.append({
                "text": para,
                "source": srcpath,
                "format": "case",
                "chunk_metadata": section_meta,
                "legal_section": meta.get("title", f"para_{idx}")
            })
    # Fallback: If any chunk >1500 chars or no chunks
    if not chunks or any(len(chunk['text']) > 1500 for chunk in chunks):
        para_chunks_fallback = chunk_by_paragraphs(text, target_chars=1500)
        out = []
        for idx, ch in enumerate(para_chunks_fallback):
            if len(ch.strip()) >= min_chunk_length:
                out.append({
                    "text": ch.strip(),
                    "source": srcpath,
                    "format": "case",
                    "chunk_metadata": meta,
                    "legal_section": meta.get("title", f"para_{idx}")
                })
        return out
    return chunks

def chunk_generic(doc: dict, min_chunk_length: int = 50, target_chars: int = 1500) -> list[dict]:
    meta = doc.get("chunk_metadata", {})
    srcpath = doc.get("source", "")
    text = doc.get("text", "")
    para_chunks = chunk_by_paragraphs(text, target_chars=target_chars)
    out_chunks = []
    for idx, ch in enumerate(para_chunks):
        subchunks = split_chunk_hard(ch, max_length=1500)
        for subidx, sch in enumerate(subchunks):
            if len(sch.strip()) >= min_chunk_length:
                out_chunks.append({
                    "text": sch.strip(),
                    "source": srcpath,
                    "format": doc.get("format", "txt"),
                    "chunk_metadata": meta,
                    "chunk_idx": f"{idx}_{subidx}",
                })
    return out_chunks

def chunk_document(doc: dict, chunk_size: int = 1500, overlap: int = 200) -> list[dict]:
    dtype = (doc.get("format") or "").strip().lower()
    if dtype == "legislation":
        chunks = chunk_legislation(doc)
        if chunks:
            return chunks
    if dtype == "journal":
        chunks = chunk_journal(doc)
        if chunks:
            return chunks
    if dtype == "case":
        chunks = chunk_case(doc)
        if chunks:
            return chunks
    return chunk_generic(doc, min_chunk_length=50, target_chars=1500)

def embed_chunk(chunk: dict) -> dict:
    try:
        from embedding.embedder import Embedder
        embedding = Embedder().embed([chunk["text"]])[0]
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        chunk["embedding"] = embedding
        return chunk
    except Exception as e:
        print(f"Embedding error for chunk {chunk.get('source', '(unknown)')} : {e}")
        return {**chunk, "embedding": None}
