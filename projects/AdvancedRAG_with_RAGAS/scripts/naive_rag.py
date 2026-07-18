"""
naive_rag.py — Naive RAG pipeline (PDF → chunk → embed → FAISS → LLM).

Exposes:
    NaiveRAG.build()      one-shot factory: downloads PDF, chunks, indexes
    NaiveRAG.query(q, k)  returns {"answer", "retrieved_chunks", "pages_used"}

Run as a script:
    python naive_rag.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from common import (
    RAG_PROMPT,
    build_context,
    build_faiss,
    download_pdf,
    load_and_chunk_pdf,
    make_llm_client,
)


@dataclass
class NaiveRAG:
    """PDF → FAISS (dense) → LLM. The baseline pipeline."""

    docs: List[Document]
    vectorstore: FAISS
    call_llm: Callable[[str], str]

    @classmethod
    def build(cls, pdf_path: str | None = None) -> "NaiveRAG":
        """One-shot factory: downloads the PDF if needed, chunks, and indexes."""
        pdf_path = pdf_path or download_pdf()
        _, docs = load_and_chunk_pdf(pdf_path)
        _, vectorstore = build_faiss(docs)
        _, call_llm = make_llm_client()
        return cls(docs=docs, vectorstore=vectorstore, call_llm=call_llm)

    def query(self, question: str, k: int = 5) -> Dict:
        retrieved = self.vectorstore.similarity_search(question, k=k)
        context = build_context(retrieved)
        answer = self.call_llm(RAG_PROMPT.format(context=context, question=question))
        return {
            "answer": answer,
            "retrieved_chunks": retrieved,
            "pages_used": sorted({d.metadata.get("page", "?") for d in retrieved}),
        }


# ── CLI demo ─────────────────────────────────────────────────────────
DEMO_QUERIES = [
    "What is the main idea of this paper?",
    "What is the Transformer architecture?",
    "What datasets were used in the experiments?",
    "Who are the authors of this paper?",  # expected to fail — motivates BM25
]


def main() -> None:
    print("🔧 Building Naive RAG pipeline …")
    rag = NaiveRAG.build()
    print(f"✅ Ready — {len(rag.docs)} chunks indexed\n")

    for q in DEMO_QUERIES:
        print("═" * 70)
        print(f" Question: {q}")
        print("═" * 70)
        out = rag.query(q)
        print(f" Pages used: {out['pages_used']}")
        print(f"\n Answer:\n{out['answer']}\n")


if __name__ == "__main__":
    main()
