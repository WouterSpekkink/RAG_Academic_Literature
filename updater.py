#!/usr/bin/env python3
# updater.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple
import hashlib

import bibtexparser
import tiktoken
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --------------------
# Config
# --------------------
SOURCE_PATH = Path(os.getenv("RAG_SOURCE_PATH", "./data/src/"))
STORE_PATH = Path(os.getenv("RAG_STORE_PATH", "./vectorstore/"))
INGEST_LOG = Path(os.getenv("RAG_INGEST_LOG", "./data/ingested.txt"))
BIBTEX_FILE = Path(os.getenv("RAG_BIBTEX_PATH", "/home/wouter/Tools/Zotero/bibtex/library.bib"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "papers")

# Embeddings / tokenization
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")
ENCODING_NAME = "cl100k_base"
TOKEN_LIMIT = 8192
SAFETY_MARGIN = 200
MAX_CHUNK_TOKENS = 1200
OVERLAP_TOKENS = 120

EMBED_CHUNK_SIZE = 96      # caps total tokens per request
ADD_BATCH_DOCS = 200       # docs per add_documents

# --------------------
# Helpers
# --------------------
def load_existing_ingested(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def load_docs(src_dir: Path) -> List[Document]:
    loader = DirectoryLoader(
        str(src_dir),
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    return loader.load()

def read_bibtex(bib_path: Path):
    with open(bib_path, "r", encoding="utf-8") as f:
        return bibtexparser.load(f)

def attach_bibtex_metadata(docs: List[Document], bib_db) -> None:
    txt_names = {p.name for p in SOURCE_PATH.iterdir() if p.is_file()}
    metadata_store = []
    for entry in bib_db.entries:
        if "file" not in entry:
            continue
        pdf_base = Path(entry["file"]).name.replace(".pdf", "")
        if f"{pdf_base}.txt" in txt_names:
            e = entry.copy()
            if "year" in e and not isinstance(e["year"], int):
                try:
                    e["year"] = int(e["year"])
                except Exception:
                    pass
            metadata_store.append(e)
    for d in docs:
        doc_base = Path(d.metadata.get("source", "")).name.replace(".txt", "")
        for e in metadata_store:
            ent_base = Path(e["file"]).name.replace(".pdf", "")
            if doc_base == ent_base:
                d.metadata.update(e)
                break

def hard_token_chunker(
    docs: List[Document],
    enc,
    max_chunk_tokens: int,
    overlap_tokens: int,
    cap_tokens: int,
) -> List[Document]:
    out: List[Document] = []

    def split_text(text: str) -> List[str]:
        ids = enc.encode(text)
        parts = []
        start = 0
        n = len(ids)
        while start < n:
            end = min(start + max_chunk_tokens, n)
            parts.append(enc.decode(ids[start:end]))
            if end == n:
                break
            start = max(0, end - overlap_tokens)
        return parts

    for d in docs:
        src = d.metadata.get("source", "unknown")
        title = d.metadata.get("title") or Path(src).stem
        header = f"{title}\n\n"
        for ch in split_text(d.page_content):
            if len(enc.encode(ch)) > cap_tokens:
                # Should not happen with hard chunker; skip if it does
                continue
            out.append(Document(page_content=header + ch, metadata=d.metadata.copy()))
    return out

def deterministic_id(doc: Document, idx: int) -> str:
    """Stable ID: source | (start_index or idx) | hash(content[:200])."""
    src = doc.metadata.get("source", "unknown")
    start = str(doc.metadata.get("start_index", idx))
    h = hashlib.sha1((doc.page_content[:200]).encode("utf-8")).hexdigest()
    return hashlib.sha1(f"{src}|{start}|{h}".encode("utf-8")).hexdigest()

# --------------------
# Main
# --------------------
def main():
    load_dotenv()
    # pick up API key from constants if present
    try:
        import constants
        if getattr(constants, "APIKEY", None):
            os.environ["OPENAI_API_KEY"] = constants.APIKEY
    except Exception:
        pass

    print("=== Loading documents ===")
    all_docs = load_docs(SOURCE_PATH)
    if not all_docs:
        print("No documents found. Exiting.")
        return

    # Filter to new files only
    already = load_existing_ingested(INGEST_LOG)
    new_docs = [d for d in all_docs if Path(d.metadata.get("source", "")).name not in already]
    if not new_docs:
        print("No new documents to ingest. Exiting.")
        return
    print(f"Found {len(new_docs)} new file(s).")

    # Attach BibTeX metadata if available
    if BIBTEX_FILE.exists():
        print("=== Attaching BibTeX metadata ===")
        bib_db = read_bibtex(BIBTEX_FILE)
        attach_bibtex_metadata(new_docs, bib_db)
    else:
        print(f"Note: BibTeX file not found at {BIBTEX_FILE}; continuing without enrichment.")

    # Split with hard token cap
    print("=== Splitting into token-capped chunks ===")
    enc = tiktoken.get_encoding(ENCODING_NAME)
    chunks = hard_token_chunker(
        new_docs,
        enc=enc,
        max_chunk_tokens=min(MAX_CHUNK_TOKENS, TOKEN_LIMIT - SAFETY_MARGIN),
        overlap_tokens=OVERLAP_TOKENS,
        cap_tokens=TOKEN_LIMIT - SAFETY_MARGIN,
    )
    if not chunks:
        print("No chunks produced; nothing to embed.")
        return

    lens = [len(enc.encode(d.page_content)) for d in chunks]
    avg = sum(lens) / len(lens)
    print(f"Chunks: {len(chunks)} | max_toks={max(lens)} | p95={sorted(lens)[int(0.95*len(lens))]} | avg≈{avg:.0f}")
    print(f"Embeddings chunk_size={EMBED_CHUNK_SIZE} → ~{int(avg)*EMBED_CHUNK_SIZE} tokens/request (< 300k cap)")

    # Embeddings with capped request size
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        request_timeout=90,
        show_progress_bar=True,
        chunk_size=EMBED_CHUNK_SIZE,
    )

    # Open collection and upsert in batches with deterministic IDs
    print("=== Embedding and updating Chroma collection ===")
    db = Chroma(
        persist_directory=str(STORE_PATH),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Build ids once so retries don't duplicate
    ids = [deterministic_id(doc, i) for i, doc in enumerate(chunks)]

    from time import sleep
    for i in range(0, len(chunks), ADD_BATCH_DOCS):
        batch_docs = chunks[i : i + ADD_BATCH_DOCS]
        batch_ids = ids[i : i + ADD_BATCH_DOCS]
        for attempt in range(3):
            try:
                # langchain_chroma supports passing ids to avoid duplicates
                db.add_documents(batch_docs, ids=batch_ids)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"add_documents failed (attempt {attempt+1}/3): {e}; retrying in 2s")
                sleep(2)

    # Append newly ingested filenames to the log
    INGEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(INGEST_LOG, "a", encoding="utf-8") as f:
        for d in new_docs:
            f.write(Path(d.metadata.get("source", "unknown")).name + "\n")

    # Basic sanity print
    try:
        count = db._collection.count()
        print(f"✅ Update complete. Collection '{COLLECTION_NAME}' doc count: {count}")
    except Exception:
        print("✅ Update complete.")

if __name__ == "__main__":
    main()
