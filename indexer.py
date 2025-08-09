#!/usr/bin/env python3
# indexer.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

import bibtexparser
import tiktoken
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from time import sleep

# --------------------
# Config
# --------------------
# Paths (override via env if you like)
SOURCE_PATH = Path(os.getenv("RAG_SOURCE_PATH", "./data/src/"))
STORE_PATH = Path(os.getenv("RAG_STORE_PATH", "./vectorstore/"))
INGEST_LOG = Path(os.getenv("RAG_INGEST_LOG", "./data/ingested.txt"))
BIBTEX_FILE = Path(os.getenv("RAG_BIBTEX_PATH", "/home/wouter/Tools/Zotero/bibtex/library.bib"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "papers")

# Embeddings / tokenization
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")
ENCODING_NAME = "cl100k_base"  # safest for OpenAI embedding models
TOKEN_LIMIT = 8192
SAFETY_MARGIN = 200
MAX_CHUNK_TOKENS = 1200
OVERLAP_TOKENS = 120

OVERSIZED_LOG = Path("./data/oversized_chunks.log")

# --------------------
# Helpers
# --------------------
def confirm_overwrite(persist_dir: Path) -> bool:
    """Ask the user before wiping/rebuilding an existing vector store."""
    if not persist_dir.exists():
        return True
    ans = input(f"⚠️  This will DELETE and rebuild '{persist_dir}'. Continue? (yes/no): ").strip().lower()
    while ans not in {"yes", "no"}:
        ans = input("Please enter 'yes' or 'no': ").strip().lower()
    return ans == "yes"

def wipe_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)

def load_docs(src_dir: Path) -> List[Document]:
    print("=== Loading documents ===")
    loader = DirectoryLoader(
        str(src_dir),
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    docs = loader.load()
    if not docs:
        print("No documents found. Exiting.")
        raise SystemExit(0)
    return docs

def read_bibtex(bib_path: Path):
    with open(bib_path, "r", encoding="utf-8") as f:
        return bibtexparser.load(f)

def attach_bibtex_metadata(docs: List[Document], bib_db) -> None:
    """Match docs to BibTeX entries via filename (pdf base name ↔ txt base name)."""
    print("=== Attaching BibTeX metadata ===")
    txt_names = {p.name for p in SOURCE_PATH.iterdir() if p.is_file()}
    metadata_store = []

    for entry in bib_db.entries:
        if "file" not in entry:
            continue
        pdf_base = Path(entry["file"]).name.replace(".pdf", "")
        if f"{pdf_base}.txt" in txt_names:
            e = entry.copy()
            # normalize year to int if possible
            if "year" in e and not isinstance(e["year"], int):
                try:
                    e["year"] = int(e["year"])
                except Exception:
                    print(f"Warning: could not normalize year for {pdf_base}")
            metadata_store.append(e)

    for d in docs:
        doc_base = Path(d.metadata.get("source", "")).name.replace(".txt", "")
        for e in metadata_store:
            ent_base = Path(e["file"]).name.replace(".pdf", "")
            if doc_base == ent_base:
                d.metadata.update(e)
                break  # assume single match

def hard_token_chunker(
    docs: List[Document],
    enc,
    max_chunk_tokens: int,
    overlap_tokens: int,
    cap_tokens: int,
) -> Tuple[List[Document], List[Tuple[str, int]]]:
    """Split by tokens with a hard cap; never emits > cap_tokens chunks."""
    print("=== Splitting into token-capped chunks ===")
    out: List[Document] = []
    oversized: List[Tuple[str, int]] = []

    def token_len(text: str) -> int:
        return len(enc.encode(text))

    def split_text(text: str) -> List[str]:
        ids = enc.encode(text)
        n = len(ids)
        parts = []
        start = 0
        while start < n:
            end = min(start + max_chunk_tokens, n)
            parts.append(enc.decode(ids[start:end]))
            if end == n:
                break
            start = max(0, end - overlap_tokens)
        return parts

    for d in docs:
        chunks = split_text(d.page_content)
        for ch in chunks:
            tks = token_len(ch)
            if tks > cap_tokens:
                # extremely pathological input; skip
                oversized.append((d.metadata.get("source", "unknown"), tks))
                continue
            meta = d.metadata.copy()
            # optional: add lightweight structural header to aid retrieval
            title = meta.get("title") or Path(meta.get("source", "unknown")).stem
            header = f"{title}\n\n"
            out.append(Document(page_content=header + ch, metadata=meta))

    # Stats
    lens = [len(enc.encode(d.page_content)) for d in out]
    if lens:
        p95 = sorted(lens)[int(0.95 * len(lens))]
        print(f"Chunks: {len(lens)} | max_toks={max(lens)} | p95={p95}")
    return out, oversized

def write_oversized_log(entries: List[Tuple[str, int]]):
    if not entries:
        return
    OVERSIZED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OVERSIZED_LOG, "w", encoding="utf-8") as f:
        for src, tks in entries:
            f.write(f"{src}\t{tks}\n")
    print(f"⚠️  Skipped {len(entries)} oversized chunks. See {OVERSIZED_LOG}")

# --------------------
# Main
# --------------------
def main():
    load_dotenv()
    # API key expected in env or constants
    try:
        import constants  # your existing file
        if constants.APIKEY:
            os.environ["OPENAI_API_KEY"] = constants.APIKEY
    except Exception:
        pass

    if not confirm_overwrite(STORE_PATH):
        print("User aborted.")
        return
    wipe_dir(STORE_PATH)

    # Load & enrich
    documents = load_docs(SOURCE_PATH)
    if BIBTEX_FILE.exists():
        bib_db = read_bibtex(BIBTEX_FILE)
        attach_bibtex_metadata(documents, bib_db)
    else:
        print(f"Note: BibTeX file not found at {BIBTEX_FILE}; continuing without metadata enrichment.")

    # Split by tokens with hard cap
    enc = tiktoken.get_encoding(ENCODING_NAME)
    clean_chunks, oversized = hard_token_chunker(
        documents,
        enc=enc,
        max_chunk_tokens=min(MAX_CHUNK_TOKENS, TOKEN_LIMIT - SAFETY_MARGIN),
        overlap_tokens=OVERLAP_TOKENS,
        cap_tokens=TOKEN_LIMIT - SAFETY_MARGIN,
    )
    write_oversized_log(oversized)

    # Final assert—fail fast if any chunk exceeds cap (shouldn’t happen)
    assert all(len(enc.encode(d.page_content)) <= (TOKEN_LIMIT - SAFETY_MARGIN) for d in clean_chunks), \
        "A chunk exceeded the token cap—check logs and splitter."

    # Embed & persist
    print("=== Embedding and creating Chroma collection ===")
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        request_timeout=90,
        show_progress_bar=True,
        chunk_size=96,   # <= important: caps tokens per request
    )

    db = Chroma(
        persist_directory=str(STORE_PATH),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    BATCH_DOCS = 200  # number of docs per add, separate from embeddings chunk_size
    for i in range(0, len(clean_chunks), BATCH_DOCS):
        batch = clean_chunks[i : i + BATCH_DOCS]
        for attempt in range(3):
            try:
                db.add_documents(batch)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"add_documents failed (attempt {attempt+1}/3): {e}; retrying in 2s")
                sleep(2)

    # Log ingested files
    INGEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(INGEST_LOG, "w", encoding="utf-8") as f:
        for d in documents:
            f.write(Path(d.metadata.get("source", "unknown")).name + "\n")

    # Basic sanity print
    try:
        count = db._collection.count()  # chroma internal
        print(f"✅ Ingestion complete. Collection '{COLLECTION_NAME}' doc count: {count}")
    except Exception:
        print("✅ Ingestion complete.")

if __name__ == "__main__":
    main()
