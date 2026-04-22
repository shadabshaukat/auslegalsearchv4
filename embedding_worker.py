"""
embedding_worker.py

Parallel vectorization/embedding worker for auslegalsearchv2/v3.
Each file: all chunks embedded as a batch, and DB inserts (Documents and Embeddings) done per file.
ENHANCED: Passes chunk-level legal metadata from loader to embedding/store.
"""

import sys
import os
from pathlib import Path
import time
from db.store import (
    get_session, update_session_progress, complete_session, fail_session,
    EmbeddingSessionFile, SessionLocal, Document, Embedding, add_document, add_embedding
)
from ingest.loader import walk_legal_files, parse_txt, parse_html, chunk_document
from embedding.embedder import Embedder

def get_completed_files(session_name):
    with SessionLocal() as session:
        rows = session.query(EmbeddingSessionFile).filter_by(session_name=session_name, status="complete").all()
        return set(row.filepath for row in rows)

def mark_file_complete(session_name, filepath):
    with SessionLocal() as session:
        f = session.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        if f:
            f.status = "complete"
            f.completed_at = now
        else:
            f = EmbeddingSessionFile(session_name=session_name, filepath=filepath, status="complete", completed_at=now)
            session.add(f)
        session.commit()

def mark_file_error(session_name, filepath):
    with SessionLocal() as session:
        f = session.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        if f:
            f.status = "error"
            f.completed_at = now
        else:
            f = EmbeddingSessionFile(session_name=session_name, filepath=filepath, status="error", completed_at=now)
            session.add(f)
        session.commit()

def read_partition_file(fname):
    with open(fname, "r") as f:
        return [line.strip() for line in f if line.strip()]

def run_embedding_session(session_name, file_list=None, poll_interval=1.0, model_name=None):
    sess = get_session(session_name)
    if not sess:
        print(f"Session {session_name} not found in DB.")
        return
    target_dir = sess.directory
    if file_list is None:
        file_list = list(walk_legal_files([target_dir]))
    completed_files = get_completed_files(session_name)
    total_files = len(file_list)
    processed_chunks = sess.processed_chunks or 0
    embedder = Embedder(model_name) if model_name else Embedder()

    print(f"Embedding session {session_name} skipping {len(completed_files)} files. Total files: {total_files}")
    try:
        for f_idx, filepath in enumerate(file_list):
            if filepath in completed_files:
                print(f"Skipping completed file: {filepath}")
                continue
            ext = Path(filepath).suffix.lower()
            if ext not in [".txt", ".html"]:
                continue
            if ext == ".txt":
                docdata = parse_txt(filepath)
            elif ext == ".html":
                docdata = parse_html(filepath)
            else:
                continue
            if not docdata or not docdata.get("text"):
                continue
            chunks = chunk_document(docdata)
            chunk_texts = [c["text"] for c in chunks]
            metas = [c.get("chunk_metadata") for c in chunks]
            errored = False
            try:
                if not chunk_texts:
                    continue
                vectors = embedder.embed(chunk_texts)
                with SessionLocal() as session:
                    doc_objs = []
                    embed_objs = []
                    for idx, c in enumerate(chunks):
                        doc_obj = Document(
                            source=c.get("source", filepath),
                            content=c["text"],
                            format=c.get("format", ext.strip(".")),
                        )
                        session.add(doc_obj)
                        session.flush()
                        doc_objs.append(doc_obj)
                        embed_objs.append(
                            Embedding(
                                doc_id=doc_obj.id,
                                chunk_index=idx,
                                vector=vectors[idx],
                                chunk_metadata=c.get("chunk_metadata"),
                            )
                        )
                    session.add_all(embed_objs)
                    session.commit()
                    processed_chunks += len(chunks)
                update_session_progress(session_name, filepath, len(chunks) - 1, processed_chunks)
                mark_file_complete(session_name, filepath)
            except Exception as e:
                print(f"Batch error embedding file {filepath}: {e}")
                errored = True
                mark_file_error(session_name, filepath)
                fail_session(session_name)
                return
            sess = get_session(session_name)
            if sess and sess.status == "error":
                print(f"Ingestion interrupted by user or stop request in DB for session {session_name}.")
                return
            time.sleep(poll_interval)
        complete_session(session_name)
        print(f"Embedding session {session_name} complete ({processed_chunks} chunks embedded).")
    except Exception as e:
        print(f"Error in embedding session: {e}")
        fail_session(session_name)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("session_name")
    parser.add_argument("embedding_model")
    parser.add_argument("--partition_file", default=None)
    args = parser.parse_args()
    if args.partition_file:
        fnames = read_partition_file(args.partition_file)
        run_embedding_session(args.session_name, file_list=fnames, model_name=args.embedding_model)
    else:
        run_embedding_session(args.session_name, model_name=args.embedding_model)
