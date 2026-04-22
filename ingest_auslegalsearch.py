import os
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any

# If needed, install: pip install beautifulsoup4
from bs4 import BeautifulSoup

SUPPORTED_EXTS = {'.txt', '.html'}

def walk_legal_files(root_dirs):
    """
    Walks through the directories recursively, yielding supported filepaths.
    Skips system/metadata files like .DS_Store.
    """
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_EXTS:
                    yield os.path.join(dirpath, fname)

def parse_txt(filepath: str) -> Dict[str, Any]:
    """
    Reads and returns the raw content from a plaintext legal file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        # Extend here with smart chunking, legal section parsing etc.
        return {"text": text, "source": filepath, "format": "txt"}
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def parse_html(filepath: str) -> Dict[str, Any]:
    """
    Extracts and returns the main textual content from a legal HTML file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        # Heuristics: Remove nav, scripts etc
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        # Grab visible text
        raw_text = soup.get_text(separator='\n', strip=True)
        # Extend here with smarter chunking/sectioning if needed
        return {"text": raw_text, "source": filepath, "format": "html"}
    except Exception as e:
        print(f"Error parsing HTML {filepath}: {e}")
        return {}

def ingest_auslegalsearch():
    """
    Main entrypoint for ingest pipeline.
    """
    # Set these to absolute Austlii corpus dirs
    HOME = "/Users/shadab/Downloads/OracleContent/CoE/CoE-Projects/Austlii/home"
    HOME2 = "/Users/shadab/Downloads/OracleContent/CoE/CoE-Projects/Austlii/home2"
    for filepath in walk_legal_files([HOME, HOME2]):
        ext = Path(filepath).suffix.lower()
        doc = None
        if ext == ".txt":
            doc = parse_txt(filepath)
        elif ext == ".html":
            doc = parse_html(filepath)
        if doc:
            print(f"Ingested {filepath} ({ext}): {len(doc['text'])} chars")
            # TODO: Chunk, yield/store doc for embedding
            # yield doc
        else:
            print(f"Skipped {filepath}")

if __name__ == "__main__":
    ingest_auslegalsearch()

"""
Notes:
- Extensible, add parse_pdf/parse_docx for future types as needed.
- Next: Integrate chunking, embedding, storage in Postgres+pgvector, and search logic.
- For production: refactor to yield docs for pipeline step composition (embedding, DB, etc).
- Consider parallelization for large corpora.
"""
