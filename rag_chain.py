# rag_chain.py
from __future__ import annotations

import hashlib
import os
from typing import List, Tuple, Iterable

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Dedupe/compress utilities
from langchain_community.document_transformers import LongContextReorder, EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# ----------------------------
# Config
# ----------------------------
PERSIST_DIR = "./vectorstore"
COLLECTION_NAME = "papers"

# Embeddings
EMBED_MODEL = "text-embedding-3-large"

# LLMs
ANSWER_MODEL = "gpt-4o"        # main answering model
UTILITY_MODEL = "gpt-4o-mini"  # cheap/fast for query expansion + rerank

# Retrieval knobs
K_FINAL = 8                 # number of chunks to keep after rerank/compress
K_MMR = 40                  # wide fetch for MMR recall
MMR_LAMBDA = 0.1            # lower -> more diversity
N_EXPANSIONS = 4            # multi-query variants to union with original

# ----------------------------
# Env / clients
# ----------------------------
load_dotenv()
try:
    import constants
    if getattr(constants, "APIKEY", None):
        os.environ["OPENAI_API_KEY"] = constants.APIKEY
except Exception:
    pass

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
db = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Base retriever (MMR for diverse recall)
base_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": K_FINAL,          # returned by retriever itself (we'll widen below manually)
        "fetch_k": K_MMR,      # pool to choose from
        "lambda_mult": MMR_LAMBDA,
    },
)

# Utility & answer models
llm_util = ChatOpenAI(model=UTILITY_MODEL, temperature=0)
llm_answer = ChatOpenAI(model=ANSWER_MODEL, temperature=0)

# Light compression (after rerank)
redundancy_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
reorderer = LongContextReorder()
compressor = DocumentCompressorPipeline(transformers=[redundancy_filter, reorderer])

# ----------------------------
# Query expansion (multi-query)
# ----------------------------
EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite search queries to improve document retrieval. "
     "Return exactly {n} diverse reformulations, one per line, no numbering."),
    ("human", "Original query:\n{q}")
])

def expand_queries(q: str, n: int = N_EXPANSIONS) -> List[str]:
    prompt = EXPAND_PROMPT.format_messages(q=q, n=n)
    resp = llm_util.invoke(prompt)
    lines = [ln.strip() for ln in resp.content.splitlines() if ln.strip()]
    # Keep at most n and ensure uniqueness & not identical to q
    uniq = []
    seen = set([q.strip().lower()])
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(ln)
        if len(uniq) >= n:
            break
    return uniq

# ----------------------------
# Wide retrieval (original + expansions) with union
# ----------------------------
def _dedupe_docs(docs: Iterable[Document]) -> List[Document]:
    """Dedupe by stable content hash + source + start index if present."""
    seen = set()
    out = []
    for d in docs:
        # Prefer a robust key: content hash + source path + (optional) start_index
        start_idx = d.metadata.get("start_index", "")
        src = d.metadata.get("source", "")
        key = hashlib.sha256((d.page_content + "|" + src + "|" + str(start_idx)).encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def retrieve_wide(question: str) -> List[Document]:
    # 1) Original query
    docs = base_retriever.invoke(question)

    # 2) Multi-query expansion + union
    try:
        variants = expand_queries(question, N_EXPANSIONS)
    except Exception:
        variants = []

    for v in variants:
        try:
            docs += base_retriever.invoke(v)
        except Exception:
            pass

    docs = _dedupe_docs(docs)

    # 3) Optional: soft LLM re-rank (fast numeric scorer)
    docs = llm_rerank(question, docs, top_n=max(K_FINAL * 2, 12))

    # 4) Light compression (removes near-duplicates, reorders for coherence)
    docs = compressor.compress_documents(docs, question)

    # 5) Keep final K
    return docs[:K_FINAL]

# ----------------------------
# LLM re-ranking (compact numeric)
# ----------------------------
RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Score the relevance of the passage to the question on a 0–10 scale. "
     "Return only a number."),
    ("human", "Question:\n{q}\n\nPassage:\n{p}")
])

def _score_passage(q: str, passage: str) -> float:
    msgs = RERANK_PROMPT.format_messages(q=q, p=passage[:1200])  # cap per doc to keep it fast
    try:
        resp = llm_util.invoke(msgs)
        txt = resp.content.strip()
        # Extract first float-like number
        num = ""
        for ch in txt:
            if ch in "0123456789.":
                num += ch
            elif num:
                break
        return float(num) if num else 0.0
    except Exception:
        return 0.0

def llm_rerank(question: str, docs: List[Document], top_n: int = 12) -> List[Document]:
    scored: List[Tuple[float, Document]] = []
    for d in docs:
        s = _score_passage(question, d.page_content)
        scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_n]]

# ----------------------------
# Formatting and sources
# ----------------------------
def format_docs(docs: List[Document]) -> str:
    """Join chunks with minimal separators. We keep source info only for display."""
    return "\n\n".join(d.page_content for d in docs)

def _cleanup(s: str) -> str:
    return (s or "").replace("{", "").replace("}", "").replace("\\", "").replace("/", "")

def get_sources(docs: List[Document]) -> List[dict]:
    """Return structured source info for UI."""
    out = []
    for doc in docs:
        m = doc.metadata
        ref = "INVALID REF"
        if m.get("ENTRYTYPE") == "article":
            ref = f"{_cleanup(m.get('author',''))} ({m.get('year','')}). {_cleanup(m.get('title',''))}. {_cleanup(m.get('journal',''))}, {_cleanup(m.get('volume',''))}({_cleanup(m.get('number',''))}): {_cleanup(m.get('pages',''))}."
        elif m.get("ENTRYTYPE") == "book":
            author = _cleanup(m.get("author", m.get("editor", "NA")))
            ref = f"{author} ({m.get('year','')}). {_cleanup(m.get('title',''))}. {_cleanup(m.get('address',''))}: {_cleanup(m.get('publisher',''))}."
        elif m.get("ENTRYTYPE") == "incollection":
            ref = f"{_cleanup(m.get('author',''))} ({m.get('year','')}). {_cleanup(m.get('title',''))}. In: {_cleanup(m.get('editor',''))} (Eds.), {_cleanup(m.get('booktitle',''))}, {_cleanup(m.get('pages',''))}."
        else:
            ref = f"{_cleanup(m.get('author',''))} ({m.get('year','')}). {_cleanup(m.get('title',''))}."

        out.append({
            "ref": ref,
            "filename": os.path.basename(m.get("source", "unknown")),
            "content": doc.page_content.replace("\n", " "),
        })
    return out

# ----------------------------
# Prompt & LCEL answer chain
# ----------------------------
SYSTEM_TEMPLATE = """
You are a knowledgeable professor working in academia.
Answer ONLY from the provided context. If the answer is not in the context, say you don't know.

Chat history (may be empty):
{chat_history}

Context:
{context}

Write as an academic text. Use APA-style in-text citations and then provide a short APA-style bibliography.
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate.from_template(SYSTEM_TEMPLATE)),
    HumanMessagePromptTemplate.from_template("{question}")
])

# LCEL chain that expects {"question","context","chat_history"} and returns a string
rag_chain = (prompt | llm_answer | StrOutputParser())

# ----------------------------
# Public helpers used by app.py
# ----------------------------
def retrieve_docs(question: str) -> List[Document]:
    """Full retrieval pipeline returning final chunks for display & context."""
    return retrieve_wide(question)

def build_context(question: str) -> Tuple[str, List[Document]]:
    """Fetch docs and build the context string."""
    docs = retrieve_docs(question)
    return format_docs(docs), docs
