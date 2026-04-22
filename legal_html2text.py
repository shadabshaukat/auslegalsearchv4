"""
Legal HTML/text/pdf file conversion for batch Streamlit UI.
- Uses DB ConversionFile ORM from store.py.
- Always constructs AustLII URL using a configurable CLEAN_BASE (e.g., '/au/cases/cth/HCA' or '/nz/cases/cth/HCA'), not hardcoded, so works for other jurisdictions.
"""

import os
import re
import html
from pathlib import Path
from bs4 import BeautifulSoup
import html2text
from db.store import SessionLocal, add_conversion_file, update_conversion_file_status, create_all_tables
from datetime import datetime
import traceback

BASE_DOC_PATH = os.environ.get("DOC_PATH", "")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "../output")
BASE_URL_PREFIX = "https://www.austlii.edu.au/cgi-bin/viewdoc"

# The canonical target root for URL construction, e.g. '/au/cases/cth/HCA' or '/nz/cases/cth/HCA'
# This should be passed to functions based on output structure.
def generate_canonical_url(rel_path_for_url, clean_base):
    rel_url = os.path.join(clean_base.lstrip("/"), rel_path_for_url.replace("\\", "/"))
    rel_url = '/' + rel_url.replace("\\", "/")
    if rel_url.endswith('.txt'):
        rel_url = rel_url[:-4] + '.html'
    return BASE_URL_PREFIX + rel_url, rel_url

def extract_jurisdiction_and_court(rel_cases_path):
    parts = rel_cases_path.lstrip("/").split("/")
    if len(parts) >= 4 and parts[0] in ("au", "nz") and parts[1] == "cases":
        return parts[2], parts[3]
    return "", ""

def reformat_date(date_str):
    months = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04',
        'May': '05', 'June': '06', 'July': '07', 'August': '08',
        'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }
    try:
        day, month, year = date_str.split()
        formatted_day = day.zfill(2)
        month_number = months[month]
        return f"{year}-{month_number}-{formatted_day} 00:00:00"
    except Exception:
        return date_str

def parse_title(title_string):
    title_string = html.unescape(title_string)
    regex = r"^(.+)(\[\d{4}\].*)\((\d{1,2} \w+ \d{4})\)$"
    match = re.match(regex, title_string)
    if match:
        title = match.group(1).strip()
        citation_parts = match.group(2).split(";")
        citations = []
        for citation in citation_parts:
            citation = citation.strip()
            citations.append(citation)
            cleaned_citation = re.sub(r"^(\[\d{4}\]|\(\d{4}\))", "", citation).strip()
            if cleaned_citation != citation:
                citations.append(cleaned_citation)
        date = reformat_date(match.group(3))
        return {
            "titles": [t.strip() for t in title.split(';')],
            "citations": citations,
            "date": date
        }
    else:
        raise ValueError("Title wasn't in the expected format!")

def generate_doc_header(metadata):
    res = "-----------------------------------\n"
    for key, value in metadata.items():
        if isinstance(value, str):
            res += f"{key}: {value}\n"
        elif isinstance(value, list):
            res += f"{key}: {str(value)}\n"
    res += "-----------------------------------\n"
    return res

def parse_case(filepath, output_rel_path, html_string, clean_base):
    soup = BeautifulSoup(html_string, 'html.parser')
    body = soup.body
    text_test = body.get_text(separator="\n", strip=True)
    if "Neutral Citation has changed" in text_test:
        return {
            "meta": {
                "status": "reference_notice",
                "notice": "Neutral Citation has changed: This file is a reference or redirect.",
            },
            "text": text_test
        }
    if re.search(r"High Court of Australia decisions beginning with .", text_test):
        return {
            "meta": {
                "status": "case_list_notice",
                "notice": "This is a list-of-cases/index page.",
            },
            "text": text_test
        }
    hr_nodes = body.find_all('hr')
    if hr_nodes:
        first_hr = hr_nodes[0]
        current = body.contents[0]
        while current and current != first_hr:
            if hasattr(current, "decompose"):
                current.decompose()
            current = body.contents[0] if body.contents else None
    if len(hr_nodes) >= 2:
        second_hr = hr_nodes[1]
        current = second_hr.next_sibling
        while current:
            next_node = current.next_sibling
            if hasattr(current, "decompose"):
                current.decompose()
            current = next_node
    small_node = body.find('small')
    if small_node:
        small_node.decompose()
    h2_node = body.find('h2')
    if not h2_node:
        raise ValueError("HCA parsing: Missing <h2> title node")
    title = h2_node.get_text(strip=True)
    parsed = parse_title(title)
    case_url, rel_url = generate_canonical_url(output_rel_path, clean_base)
    jurisdiction, court = extract_jurisdiction_and_court(rel_url)
    parsed['type'] = 'case'
    parsed['url'] = case_url
    parsed['jurisdiction'] = jurisdiction
    parsed['court'] = court
    text = h2t.handle(str(body)).strip()
    return {"meta": parsed, "text": generate_doc_header(parsed) + "\n\n" + text}

# Configure HTML to text converter
h2t = html2text.HTML2Text()
h2t.body_width = 0
h2t.ignore_links = False
h2t.ignore_images = True
h2t.skip_internal_links = True
h2t.ignore_emphasis = False
h2t.strong_mark = "**"
h2t.emphasis_mark = "_"

def convert_html_file(src_path, rel_path, out_base, clean_base):
    out_txt_path = os.path.splitext(rel_path)[0] + ".txt"
    dst_fullpath = os.path.join(out_base, out_txt_path)
    os.makedirs(os.path.dirname(dst_fullpath), exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as htmlf:
        html_string = htmlf.read()
    try:
        output_rel_path_for_url = rel_path.replace("\\", "/")
        if output_rel_path_for_url.endswith('.txt'):
            output_rel_path_for_url = output_rel_path_for_url[:-4] + '.html'
        elif output_rel_path_for_url.endswith('.html'):
            pass  # leave as-is
        else:
            output_rel_path_for_url += '.html'
        parsed = parse_case(src_path, output_rel_path_for_url, html_string, clean_base)
        is_structured = "titles" in parsed.get("meta", {})
    except Exception:
        parsed = None
        is_structured = False
    with open(dst_fullpath, "w", encoding="utf-8") as outf:
        if parsed and is_structured:
            outf.write(parsed["text"])
        elif parsed and "reference_notice" in parsed.get("meta", {}).get("status", ""):
            outf.write(parsed["meta"]["notice"] + "\n" + parsed["text"])
        elif parsed and "case_list_notice" in parsed.get("meta", {}).get("status", ""):
            outf.write(parsed["meta"]["notice"] + "\n" + parsed["text"])
        else:
            soup = BeautifulSoup(html_string, "html.parser")
            outf.write(soup.get_text(separator="\n", strip=True))
    return dst_fullpath

def convert_pdf_file(src_path, rel_path, out_base):
    out_txt_path = os.path.splitext(rel_path)[0] + ".txt"
    dst_fullpath = os.path.join(out_base, out_txt_path)
    os.makedirs(os.path.dirname(dst_fullpath), exist_ok=True)
    try:
        import PyPDF2
        text_content = ""
        with open(src_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        with open(dst_fullpath, "w", encoding="utf-8") as outf:
            outf.write(text_content)
    except Exception as e:
        raise Exception(f"PDF conversion error: {e}")
    return dst_fullpath

def log_html_conversion(session_name, src_file, dst_file, status="pending", error_message=None):
    cf_id = add_conversion_file(session_name, src_file, dst_file, status=status)
    if status != "pending":
        update_conversion_file_status(cf_id, status, error_message=error_message, success=(status == "complete"))
    return cf_id

def streamlit_conversion_runner(input_dir, output_dir, session_name, num_gpus=1, clean_base="/au/cases/cth/HCA", status_write_func=None, stop_flag_func=None):
    """
    Batch convert all .html/.pdf in input_dir to .txt in output_dir, 
    preserving relative directory structure. URL in metadata is always BASE_URL_PREFIX + clean_base + relpath (with .html ext for the URL).
    clean_base should be set to match your intended canonical data location (e.g. /au/cases/cth/HCA or /nz/cases/cth/HCA).
    """
    create_all_tables()
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for fname in filenames:
            if fname.lower().endswith(".html") or fname.lower().endswith(".pdf"):
                fullpath = os.path.join(root, fname)
                relpath = os.path.relpath(fullpath, input_dir)
                files.append((fullpath, relpath))
    total = len(files)
    complete = 0
    for i, (src_path, rel_path) in enumerate(files, 1):
        if stop_flag_func is not None and stop_flag_func():
            if status_write_func:
                status_write_func(f"Stopped! {complete}/{total} files processed.")
            break
        fname = os.path.basename(src_path)
        ext = os.path.splitext(fname)[1].lower()
        try:
            if ext == ".html":
                dst_fullpath = convert_html_file(src_path, rel_path, output_dir, clean_base)
            elif ext == ".pdf":
                dst_fullpath = convert_pdf_file(src_path, rel_path, output_dir)
            log_html_conversion(session_name, src_path, dst_fullpath, "complete")
            complete += 1
            if status_write_func:
                status_write_func(f"✓ {rel_path} ==> {dst_fullpath} ({i}/{total})")
        except Exception as e:
            log_html_conversion(session_name, src_path, os.path.join(output_dir, rel_path), "error", error_message=str(e))
            if status_write_func:
                tb = traceback.format_exc()
                status_write_func(f"✗ {rel_path}: {e}\n{tb}")
    if status_write_func:
        status_write_func(f"Done! {complete}/{total} files converted.")
