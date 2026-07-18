"""
common.py — shared utilities for the RAG scripts.

Provides:
    - load_groq_key()          portable GROQ_API_KEY loader (env / .env / Colab / prompt)
    - make_llm_client()        thin wrapper returning a Groq client + call_llm(prompt)
    - download_pdf(url, path)  idempotent PDF download
    - load_and_chunk_pdf(...)  PyMuPDF loader + RecursiveCharacterTextSplitter
    - build_faiss(docs)        FAISS vector store from HuggingFace MiniLM embeddings
    - preprocess(text)         lowercase + strip punctuation → tokens (for BM25)
    - RAG_PROMPT               canonical grounded-answer prompt template

Runnable smoke test:
    python common.py
"""

from __future__ import annotations

import os
import re
import tempfile
import urllib.request
from getpass import getpass
from typing import Callable, List, Tuple

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Constants ────────────────────────────────────────────────────────
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"
DEFAULT_PDF_NAME = "Attention is all you need.pdf"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"

RAG_PROMPT = """You are a precise assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "Not found in document".
Cite pages like [Page X].

Context:
{context}

Question: {question}

Answer:"""


# ── API-key loader ───────────────────────────────────────────────────
def load_groq_key() -> str:
    """Load GROQ_API_KEY from env / Colab secrets / .env / interactive prompt.

    Sets os.environ["GROQ_API_KEY"] as a side effect.
    Returns the source it was loaded from.
    """
    if os.environ.get("GROQ_API_KEY"):
        return "environment"
    try:
        from google.colab import userdata  # type: ignore
        os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
        return "colab-secrets"
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        if load_dotenv() and os.environ.get("GROQ_API_KEY"):
            return "dotenv"
    except ImportError:
        pass
    os.environ["GROQ_API_KEY"] = getpass("Enter your GROQ_API_KEY: ").strip()
    return "prompt"


# ── LLM client factory ───────────────────────────────────────────────
def make_llm_client(model: str = DEFAULT_LLM_MODEL, temperature: float = 0.2):
    """Return (client, call_llm) — call_llm(prompt) → str."""
    from groq import Groq

    if not os.environ.get("GROQ_API_KEY"):
        load_groq_key()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def call_llm(prompt: str) -> str:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return res.choices[0].message.content

    return client, call_llm


# ── PDF download ─────────────────────────────────────────────────────
def download_pdf(url: str = DEFAULT_PDF_URL, name: str = DEFAULT_PDF_NAME) -> str:
    """Download `url` to a stable OS temp location if missing. Return the file path."""
    if os.path.isdir("/content"):  # Colab
        pdf_dir = "/content"
    else:
        pdf_dir = os.path.join(tempfile.gettempdir(), "rag_demo")
        os.makedirs(pdf_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, name)
    if not os.path.exists(pdf_path):
        print(f"Downloading PDF → {pdf_path} …")
        urllib.request.urlretrieve(url, pdf_path)
    return pdf_path


# ── PDF loading + chunking ───────────────────────────────────────────
def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Tuple[List[Document], List[Document]]:
    """Return (pages, chunks). Chunks get chunk_id + 1-indexed page metadata."""
    pages = PyMuPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(pages)
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i
        d.metadata["page"] = d.metadata.get("page", 0) + 1
    return pages, docs


# ── Dense embeddings + FAISS ─────────────────────────────────────────
def build_faiss(
    docs: List[Document],
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Tuple[HuggingFaceEmbeddings, FAISS]:
    """Return (embeddings, vectorstore)."""
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return embeddings, vectorstore


# ── BM25 preprocessing ───────────────────────────────────────────────
def preprocess(text: str) -> List[str]:
    """Lowercase, drop punctuation, split on whitespace. Used for BM25 tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


# ── Context builder for prompts ──────────────────────────────────────
def build_context(chunks: List[Document]) -> str:
    """Concatenate chunk texts with page-tagged headers."""
    return "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in chunks
    )


# ── CLI smoke test ───────────────────────────────────────────────────
def _smoke_test() -> None:
    print(f"✅ GROQ_API_KEY loaded from: {load_groq_key()}")
    pdf_path = download_pdf()
    pages, chunks = load_and_chunk_pdf(pdf_path)
    print(f"✅ Loaded {len(pages)} pages · {len(chunks)} chunks")
    _, vs = build_faiss(chunks)
    hits = vs.similarity_search("What is attention?", k=3)
    print(f"✅ FAISS returned {len(hits)} chunks for sample query")
    _, call_llm = make_llm_client()
    reply = call_llm("Say 'ready' and nothing else.")
    print(f"✅ Groq LLM reply: {reply!r}")


if __name__ == "__main__":
    _smoke_test()
