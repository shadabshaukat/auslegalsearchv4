#!/usr/bin/env python3
"""
Relational metadata loader for AUSLegalSearch v3

Purpose:
  - Parse files (txt/html) with dashed frontmatter and load normalized relational records
    into dedicated tables for cases and legislation (journals/treaties can be extended).
  - Complements beta_worker (which writes raw documents/embeddings) by providing a
    structured relational view driven by frontmatter/headers and dashed sections.

Schema used (defined in db/store.py):
  - cases(case_id, url, jurisdiction, case_date, court)
  - case_names(case_name_id, case_id, name)
  - case_citation_refs(citation_ref_id, case_id, citation)

  - legislation(legislation_id, url, jurisdiction, enacted_date, year, name, database)
  - legislation_sections(section_id, legislation_id, identifier, type, title, content)

Assumptions:
  - Frontmatter (dashed header) is present at file start for doc-level metadata:
      -----------------------------------
      title: Offshore Petroleum Act 2006 (No. 14, 2006)
      year: 2006
      date: 2006-01-01 00:00:00
      type: legislation
      data_quality: H
      subjurisdiction: cth
      jurisdiction: au
      database: num_act
      url: https://austlii.edu.au/cgi-bin/viewdoc/...
      -----------------------------------
  - Legislation files may contain repeated dashed headers for sections/parts; we parse them.

CLI:
  python -m ingest.relational_loader \
    --root "/abs/path/to/data" \
    --log_dir "./logs"

Or:
  python -m ingest.relational_loader \
    --partition_file ".beta-gpu-partition-SESSION_NAME-gpu0.txt" \
    --log_dir "./logs"

Notes:
  - Idempotent-ish by checking existing rows by URL and name (best effort).
  - Does not embed or write to documents/embeddings; only normalized relational tables.
  - Extendable to journal/treaty loaders by adding models and mappers similarly.
"""

from __future__ import annotations

import os
import re
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, OperationalError

from db.store import (
    engine, SessionLocal,
    create_all_tables,
    Case, CaseName, CaseCitationRef,
    Legislation, LegislationSection,
    Journal, JournalAuthor, JournalCitationRef,
    Treaty, TreatyCountry, TreatyCitationRef,
)
from ingest.loader import parse_txt, parse_html, extract_metadata_block
from ingest.semantic_chunker import parse_dashed_blocks, detect_doc_type

SUPPORTED_EXTS = {".txt", ".html"}

def _natural_sort_key(s: str):
    import re as _re
    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s or "")]

def find_all_supported_files(root_dir: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = sorted(dirnames, key=_natural_sort_key)
        files = [f for f in filenames if Path(f).suffix.lower() in SUPPORTED_EXTS]
        for f in sorted(files, key=_natural_sort_key):
            out.append(os.path.abspath(os.path.join(dirpath, f)))
    # de-duplicate and return in natural order
    return sorted(list(dict.fromkeys(out)), key=_natural_sort_key)

def read_partition_file(fname: str) -> List[str]:
    with open(fname, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def parse_file(filepath: str) -> Dict[str, Any]:
    ext = Path(filepath).suffix.lower()
    if ext == ".txt":
        return parse_txt(filepath)
    if ext == ".html":
        return parse_html(filepath)
    return {}

def parse_date(value: Any) -> Optional[str]:
    """
    Return ISO date (YYYY-MM-DD) if parsable, else None.
    Accepts strings like "2021-01-01 00:00:00" or "21-05-2003 00:00:00" (DD-MM-YYYY).
    """
    if not value:
        return None
    s = str(value).strip()
    # YYYY-MM-DD...
    if len(s) >= 10 and re.match(r"\d{4}-\d{2}-\d{2}", s[:10]):
        return s[:10]
    # DD-MM-YYYY...
    m = re.match(r"^\s*(\d{2})-(\d{2})-(\d{4})", s)
    if m:
        d, mth, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mth}-{d}"
    # Try YYYY only (store as Jan 1)
    m2 = re.match(r"^\s*(\d{4})\s*$", s)
    if m2:
        return f"{m2.group(1)}-01-01"
    return None

def as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None

def ensure_case(session, url: Optional[str], jurisdiction: Optional[str], subjurisdiction: Optional[str], date_iso: Optional[str], court: Optional[str]) -> Case:
    """
    Upsert-like: find case by URL if present, else by (jurisdiction, subjurisdiction, court, case_date) as heuristic.
    """
    # Try by URL
    if url:
        row = session.execute(select(Case).where(Case.url == url)).scalar_one_or_none()
        if row:
            # Update attrs if missing
            changed = False
            if jurisdiction and not row.jurisdiction:
                row.jurisdiction = jurisdiction; changed = True
            if subjurisdiction and not getattr(row, "subjurisdiction", None):
                row.subjurisdiction = subjurisdiction; changed = True
            if date_iso and not row.case_date:
                row.case_date = date_iso; changed = True
            if court and not row.court:
                row.court = court; changed = True
            if changed: session.commit()
            return row
    # Fallback heuristic: first matching jurisdiction/subjurisdiction
    q = select(Case)
    if jurisdiction:
        q = q.where(Case.jurisdiction == jurisdiction)
    if subjurisdiction:
        q = q.where(Case.subjurisdiction == subjurisdiction)
    row = session.execute(q).scalars().first()
    if row:
        return row
    obj = Case(url=url, jurisdiction=jurisdiction, subjurisdiction=subjurisdiction, case_date=date_iso, court=court)
    session.add(obj); session.commit(); session.refresh(obj)
    return obj

def add_case_name(session, case_id: int, name: str) -> None:
    name = (name or "").strip()
    if not name:
        return
    # avoid duplicates for same case/name
    existing = session.execute(
        select(CaseName).where(CaseName.case_id == case_id, CaseName.name == name)
    ).scalar_one_or_none()
    if existing:
        return
    session.add(CaseName(case_id=case_id, name=name))
    session.commit()

def add_case_citations(session, case_id: int, citations: List[str]) -> None:
    for c in citations:
        cc = (c or "").strip()
        if not cc:
            continue
        exists = session.execute(
            select(CaseCitationRef).where(CaseCitationRef.case_id == case_id, CaseCitationRef.citation == cc)
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(CaseCitationRef(case_id=case_id, citation=cc))
    session.commit()

def ensure_legislation(session, url: Optional[str], jurisdiction: Optional[str], subjurisdiction: Optional[str],
                       enacted_date_iso: Optional[str], year: Optional[int],
                       name: Optional[str], database: Optional[str]) -> Legislation:
    # Try by URL first
    if url:
        row = session.execute(select(Legislation).where(Legislation.url == url)).scalar_one_or_none()
        if row:
            # update missing attrs
            changed = False
            if jurisdiction and not row.jurisdiction: row.jurisdiction = jurisdiction; changed = True
            if subjurisdiction and not getattr(row, "subjurisdiction", None): row.subjurisdiction = subjurisdiction; changed = True
            if enacted_date_iso and not row.enacted_date: row.enacted_date = enacted_date_iso; changed = True
            if year and not row.year: row.year = year; changed = True
            if name and not row.name: row.name = name; changed = True
            if database and not row.database: row.database = database; changed = True
            if changed: session.commit()
            return row
    # Heuristic: match by (name, jurisdiction, year)
    filters = []
    if name: filters.append(Legislation.name == name)
    if jurisdiction: filters.append(Legislation.jurisdiction == jurisdiction)
    if subjurisdiction: filters.append(Legislation.subjurisdiction == subjurisdiction)
    if year: filters.append(Legislation.year == year)
    if filters:
        row = session.execute(select(Legislation).where(*filters)).scalar_one_or_none()
        if row:
            return row
    obj = Legislation(url=url, jurisdiction=jurisdiction, subjurisdiction=subjurisdiction,
                      enacted_date=enacted_date_iso, year=year, name=name, database=database)
    session.add(obj); session.commit(); session.refresh(obj)
    return obj

def add_legislation_sections_from_dashed(session, legislation_id: int, doc_text: str) -> int:
    """
    Parse dashed blocks and insert LegislationSection rows.
    """
    blocks = parse_dashed_blocks(doc_text or "")
    count = 0
    for header_meta, body in blocks:
        identifier = header_meta.get("section") or header_meta.get("identifier") or header_meta.get("chunk_id")
        sec_title = header_meta.get("title") or header_meta.get("section_title")
        # Infer type: explicit 'type' key or guess from title keywords
        sec_type = header_meta.get("type")
        if not sec_type:
            t = (sec_title or "").lower()
            if "schedule" in t:
                sec_type = "schedule"
            elif "regulation" in t or "reg." in t:
                sec_type = "regulation"
            else:
                sec_type = "section"
        content = (body or "").strip()
        if not content:
            continue
        # avoid duplicates for same legislation_id + identifier + title
        exists = session.execute(
            select(LegislationSection).where(
                LegislationSection.legislation_id == legislation_id,
                LegislationSection.identifier == identifier if identifier else (LegislationSection.identifier.is_(None)),
                LegislationSection.title == sec_title if sec_title else (LegislationSection.title.is_(None)),
            )
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(LegislationSection(
            legislation_id=legislation_id,
            identifier=identifier,
            type=sec_type,
            title=sec_title,
            content=content
        ))
        count += 1
    if count:
        session.commit()
    return count

def _extract_citations_from_meta(meta: Dict[str, Any]) -> List[str]:
    """
    Extract citations from common keys: citation, citations, md_citation, md_citations.
    """
    if not meta:
        return []
    out: List[str] = []
    for key in ("md_citations", "citations", "mdCitation", "md_citation", "citation"):
        if key in meta:
            val = meta.get(key)
            if isinstance(val, list):
                out.extend([str(x).strip() for x in val if str(x).strip()])
            elif isinstance(val, str):
                v = val.strip()
                if v:
                    out.append(v)
    # de-dup preserve order
    return list(dict.fromkeys(out))

def _split_authors(val: Any) -> List[str]:
    if not val:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val)
    # split by ';' or ',' conservatively
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]

def ensure_journal(session, url: Optional[str], jurisdiction: Optional[str], subjurisdiction: Optional[str],
                   published_date_iso: Optional[str], year: Optional[int],
                   title: Optional[str], database: Optional[str]) -> Journal:
    if url:
        row = session.execute(select(Journal).where(Journal.url == url)).scalar_one_or_none()
        if row:
            changed = False
            if jurisdiction and not row.jurisdiction: row.jurisdiction = jurisdiction; changed = True
            if subjurisdiction and not getattr(row, "subjurisdiction", None): row.subjurisdiction = subjurisdiction; changed = True
            if published_date_iso and not row.published_date: row.published_date = published_date_iso; changed = True
            if year and not row.year: row.year = year; changed = True
            if title and not row.title: row.title = title; changed = True
            if database and not row.database: row.database = database; changed = True
            if changed: session.commit()
            return row
    filters = []
    if title: filters.append(Journal.title == title)
    if jurisdiction: filters.append(Journal.jurisdiction == jurisdiction)
    if subjurisdiction: filters.append(Journal.subjurisdiction == subjurisdiction)
    if year: filters.append(Journal.year == year)
    if filters:
        row = session.execute(select(Journal).where(*filters)).scalar_one_or_none()
        if row:
            return row
    obj = Journal(url=url, jurisdiction=jurisdiction, subjurisdiction=subjurisdiction, published_date=published_date_iso,
                  year=year, title=title, database=database)
    session.add(obj); session.commit(); session.refresh(obj)
    return obj

def add_journal_authors(session, journal_id: int, authors: List[str]) -> None:
    for a in authors:
        aa = (a or "").strip()
        if not aa:
            continue
        exists = session.execute(
            select(JournalAuthor).where(JournalAuthor.journal_id == journal_id, JournalAuthor.name == aa)
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(JournalAuthor(journal_id=journal_id, name=aa))
    session.commit()

def add_journal_citations(session, journal_id: int, citations: List[str]) -> None:
    for c in citations:
        cc = (c or "").strip()
        if not cc:
            continue
        exists = session.execute(
            select(JournalCitationRef).where(JournalCitationRef.journal_id == journal_id, JournalCitationRef.citation == cc)
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(JournalCitationRef(journal_id=journal_id, citation=cc))
    session.commit()

def process_journal(session, meta: Dict[str, Any], text: str) -> None:
    url = (meta.get("url") or "").strip() or None
    jurisdiction = (meta.get("jurisdiction") or "").strip() or None
    subjurisdiction = (meta.get("subjurisdiction") or "").strip() or None
    published_date_iso = parse_date(meta.get("date")) or None
    year = as_int(meta.get("year"))
    title = (meta.get("title") or "").strip() or None
    database = (meta.get("database") or "").strip() or None
    j = ensure_journal(session, url=url, jurisdiction=jurisdiction, subjurisdiction=subjurisdiction,
                       published_date_iso=published_date_iso, year=year, title=title, database=database)
    # Authors
    authors = _split_authors(meta.get("author"))
    if authors:
        add_journal_authors(session, j.journal_id, authors)
    # Citation(s)
    cites = _extract_citations_from_meta(meta)
    if cites:
        add_journal_citations(session, j.journal_id, cites)

def ensure_treaty(session, url: Optional[str], jurisdiction: Optional[str], subjurisdiction: Optional[str],
                  signed_date_iso: Optional[str], year: Optional[int],
                  title: Optional[str], database: Optional[str]) -> Treaty:
    if url:
        row = session.execute(select(Treaty).where(Treaty.url == url)).scalar_one_or_none()
        if row:
            changed = False
            if jurisdiction and not row.jurisdiction: row.jurisdiction = jurisdiction; changed = True
            if subjurisdiction and not getattr(row, "subjurisdiction", None): row.subjurisdiction = subjurisdiction; changed = True
            if signed_date_iso and not row.signed_date: row.signed_date = signed_date_iso; changed = True
            if year and not row.year: row.year = year; changed = True
            if title and not row.title: row.title = title; changed = True
            if database and not row.database: row.database = database; changed = True
            if changed: session.commit()
            return row
    filters = []
    if title: filters.append(Treaty.title == title)
    if jurisdiction: filters.append(Treaty.jurisdiction == jurisdiction)
    if subjurisdiction: filters.append(Treaty.subjurisdiction == subjurisdiction)
    if year: filters.append(Treaty.year == year)
    if filters:
        row = session.execute(select(Treaty).where(*filters)).scalar_one_or_none()
        if row:
            return row
    obj = Treaty(url=url, jurisdiction=jurisdiction, subjurisdiction=subjurisdiction, signed_date=signed_date_iso,
                 year=year, title=title, database=database)
    session.add(obj); session.commit(); session.refresh(obj)
    return obj

def add_treaty_countries(session, treaty_id: int, countries: List[str]) -> None:
    for c in countries or []:
        cc = (str(c) or "").strip()
        if not cc:
            continue
        exists = session.execute(
            select(TreatyCountry).where(TreatyCountry.treaty_id == treaty_id, TreatyCountry.country == cc)
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(TreatyCountry(treaty_id=treaty_id, country=cc))
    session.commit()

def add_treaty_citations(session, treaty_id: int, citations: List[str]) -> None:
    for c in citations:
        cc = (c or "").strip()
        if not cc:
            continue
        exists = session.execute(
            select(TreatyCitationRef).where(TreatyCitationRef.treaty_id == treaty_id, TreatyCitationRef.citation == cc)
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(TreatyCitationRef(treaty_id=treaty_id, citation=cc))
    session.commit()

def process_treaty(session, meta: Dict[str, Any], text: str) -> None:
    url = (meta.get("url") or "").strip() or None
    jurisdiction = (meta.get("jurisdiction") or "").strip() or None
    subjurisdiction = (meta.get("subjurisdiction") or "").strip() or None
    signed_date_iso = parse_date(meta.get("date")) or None
    year = as_int(meta.get("year"))
    title = (meta.get("title") or "").strip() or None
    database = (meta.get("database") or "").strip() or None
    t = ensure_treaty(session, url=url, jurisdiction=jurisdiction, subjurisdiction=subjurisdiction,
                      signed_date_iso=signed_date_iso, year=year, title=title, database=database)
    # Countries
    countries = meta.get("countries") or []
    if isinstance(countries, list):
        add_treaty_countries(session, t.treaty_id, countries)
    # Citations
    cites = _extract_citations_from_meta(meta)
    if cites:
        add_treaty_citations(session, t.treaty_id, cites)

def process_case(session, meta: Dict[str, Any], text: str) -> None:
    url = (meta.get("url") or "").strip() or None
    jurisdiction = (meta.get("jurisdiction") or "").strip() or None
    subjurisdiction = (meta.get("subjurisdiction") or "").strip() or None
    court = (meta.get("database") or "").strip() or None  # database often maps to court for cases (e.g., HCA, NSWSC)
    date_iso = parse_date(meta.get("date")) or None
    case = ensure_case(session, url=url, jurisdiction=jurisdiction, subjurisdiction=subjurisdiction, date_iso=date_iso, court=court)
    # Add name(s)
    title = (meta.get("title") or "").strip()
    if title:
        add_case_name(session, case.case_id, title)
    # Add any alternate titles list
    try:
        titles = meta.get("titles")
        if isinstance(titles, list):
            for t in titles:
                tt = (str(t) or "").strip()
                if tt:
                    add_case_name(session, case.case_id, tt)
    except Exception:
        pass
    # Optionally, infer more names from headings (e.g., "v" patterns) — light heuristic
    try:
        m = re.search(r"([A-Z][A-Za-z0-9\.\-& ]+)\s+v\s+([A-Z][A-Za-z0-9\.\-& ]+)", title)
        if m:
            add_case_name(session, case.case_id, f"{m.group(1).strip()} v {m.group(2).strip()}")
    except Exception:
        pass
    # Citations from metadata
    citations = _extract_citations_from_meta(meta)
    if citations:
        add_case_citations(session, case.case_id, citations)

def process_legislation(session, meta: Dict[str, Any], text: str) -> None:
    url = (meta.get("url") or "").strip() or None
    jurisdiction = (meta.get("jurisdiction") or "").strip() or None
    subjurisdiction = (meta.get("subjurisdiction") or "").strip() or None
    enacted_date_iso = parse_date(meta.get("date")) or None
    year = as_int(meta.get("year"))
    name = (meta.get("title") or "").strip() or None
    database = (meta.get("database") or "").strip() or None
    leg = ensure_legislation(session, url=url, jurisdiction=jurisdiction, subjurisdiction=subjurisdiction,
                             enacted_date_iso=enacted_date_iso, year=year, name=name, database=database)
    # Add sections by parsing dashed blocks
    try:
        add_legislation_sections_from_dashed(session, leg.legislation_id, text or "")
    except Exception as e:
        print(f"[relational_loader] Warning: failed to parse sections for {url or name}: {e}")

def run_loader(root: Optional[str], partition_file: Optional[str], log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    files: List[str]
    if partition_file:
        files = read_partition_file(partition_file)
    else:
        if not root:
            raise SystemExit("Either --root or --partition_file must be provided")
        files = find_all_supported_files(root)

    print(f"[relational_loader] start files={len(files)}")
    create_all_tables()
    ok = 0
    err = 0
    skipped = 0

    with SessionLocal() as session:
        for idx, fp in enumerate(files, start=1):
            try:
                if idx % 100 == 0:
                    print(f"[relational_loader] progress {idx}/{len(files)}")
                doc = parse_file(fp)
                if not doc or not doc.get("text"):
                    skipped += 1
                    continue
                meta = dict(doc.get("chunk_metadata") or {})
                text = doc.get("text") or ""
                dtype = (meta.get("type") or "").strip().lower() or detect_doc_type(meta, text)
                if dtype == "case":
                    process_case(session, meta, text)
                    ok += 1
                elif dtype == "legislation":
                    process_legislation(session, meta, text)
                    ok += 1
                elif dtype == "journal":
                    process_journal(session, meta, text)
                    ok += 1
                elif dtype == "treaty":
                    process_treaty(session, meta, text)
                    ok += 1
                else:
                    skipped += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Ensure the session is usable for the next file after any failure
                try:
                    session.rollback()
                except Exception:
                    pass
                err += 1
                print(f"[relational_loader] error: {fp} :: {e}")

    print(f"[relational_loader] done. ok={ok} skipped={skipped} err={err}")

def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    import argparse
    ap = argparse.ArgumentParser(description="Relational metadata loader: normalize case/legislation into relational tables.")
    ap.add_argument("--root", default=None, help="Root directory (used if --partition_file not provided)")
    ap.add_argument("--partition_file", default=None, help="Text file listing file paths for this loader")
    ap.add_argument("--log_dir", default="./logs", help="Directory to write logs")
    return vars(ap.parse_args(argv))

if __name__ == "__main__":
    args = _parse_cli_args(sys.argv[1:])
    run_loader(
        root=args.get("root"),
        partition_file=args.get("partition_file"),
        log_dir=args.get("log_dir") or "./logs",
    )
