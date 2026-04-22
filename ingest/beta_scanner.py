"""
Beta dataset scanner for AUSLegalSearch v3.

Goal:
- From a given root directory, pick at most ONE representative file from each folder, recursively.
- Skip ANY subdirectory whose name looks like a year (e.g., 1999, 2001, 2020, 2024) entirely.
- Only consider supported file types (.txt, .html) for ingestion (to match the current pipeline).
- Return a deterministic, naturally sorted list of filepaths.

Rationale:
- The new dataset may contain year-based subtrees (e.g., '/.../SomeCourt/2001', '/.../SomeCourt/2002', ...).
  These contain per-year document dumps we do not want to scan exhaustively.
- We only need one file per folder to build a representative index for beta launch.

Usage:
    from ingest.beta_scanner import find_sample_files

    files = find_sample_files("/abs/path/to/Data_for_Beta_Launch")
    for f in files:
        print(f)
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Set, Tuple

SUPPORTED_EXTS = {".txt", ".html"}

_YEAR_DIR_RE = re.compile(r"^(19|20)\d{2}$")  # 1900-2099

def _is_year_dir(name: str) -> bool:
    return bool(_YEAR_DIR_RE.match(name.strip()))

def _natural_sort_key(s: str):
    # Natural sort helper: split digits from text e.g., "file10" < "file2"
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

def find_sample_files(root_dir: str, skip_year_dirs: bool = True) -> List[str]:
    """
    Recursively traverses root_dir.
    - For each directory visited, selects at most one supported file (prefer .txt over .html if both exist).
    - If skip_year_dirs=True, does not descend into subdirectories that look like years.
    - Returns deterministic sorted list of absolute file paths.
    """
    if not os.path.isdir(root_dir):
        return []

    sample_files: List[str] = []
    # We need to control traversal: prune year-dirs in-place
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune year directories to avoid descending
        if skip_year_dirs:
            pruned: List[str] = []
            for d in dirnames:
                if _is_year_dir(d):
                    # Skip descending into this dir
                    continue
                pruned.append(d)
            # Modify dirnames in-place to control os.walk
            dirnames[:] = sorted(pruned, key=_natural_sort_key)
        else:
            dirnames[:] = sorted(dirnames, key=_natural_sort_key)

        # Pick one file from this directory if present
        supported = [f for f in filenames if Path(f).suffix.lower() in SUPPORTED_EXTS]
        if not supported:
            continue
        supported.sort(key=_natural_sort_key)

        # Preference: .txt first, else .html
        preferred = None
        txts = [f for f in supported if f.lower().endswith(".txt")]
        htmls = [f for f in supported if f.lower().endswith(".html")]
        if txts:
            preferred = txts[0]
        elif htmls:
            preferred = htmls[0]

        if preferred:
            abspath = os.path.abspath(os.path.join(dirpath, preferred))
            sample_files.append(abspath)

    # Deduplicate and deterministic sort
    dedup = sorted(list(dict.fromkeys(sample_files)), key=_natural_sort_key)
    return dedup

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Scan beta dataset and print 1 sample file per folder, skipping year dirs.")
    ap.add_argument("root", help="Root directory (absolute path) e.g. /Users/.../Data_for_Beta_Launch")
    ap.add_argument("--no-skip-year-dirs", action="store_true", help="Do not skip 4-digit year directories")
    args = ap.parse_args()
    files = find_sample_files(args.root, skip_year_dirs=not args.no_skip_year_dirs)
    print(json.dumps(files, indent=2))
