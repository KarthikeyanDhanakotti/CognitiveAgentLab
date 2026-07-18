"""
advanced_rag.py — Hybrid retrieval (FAISS + BM25) → RRF fusion → Cross-Encoder rerank → LLM.

Exposes:
    AdvancedRAG.build()      one-shot factory
    AdvancedRAG.query(q, top_k, fetch_k)
    rrf(result_lists, k)     reusable Reciprocal Rank Fusion

Run as a script:
    python advanced_rag.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from common import (
    RAG_PROMPT,
    build_context,
    build_faiss,
    download_pdf,
    load_and_chunk_pdf,
    make_llm_client,
    preprocess,
)


# ── Reciprocal Rank Fusion ───────────────────────────────────────────
def rrf(result_lists: List[List[Document]], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank + 1). Returns descending list."""
    scores: Dict[int, Dict] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = hash(doc.page_content)
            scores.setdefault(key, {"doc": doc, "score": 0.0})
            scores[key]["score"] += 1.0 / (rank + k + 1)
    return sorted(scores.values(), key=lambda x: x["score"], reverse=True)


@dataclass
class AdvancedRAG:
    """Hybrid retrieval + RRF + Cross-Encoder rerank → LLM."""

    docs: List[Document]
    vectorstore: FAISS
    bm25: BM25Okapi
    reranker: CrossEncoder
    call_llm: Callable[[str], str]

    @classmethod
    def build(
        cls,
        pdf_path: str | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_device: str = "cpu",
    ) -> "AdvancedRAG":
        pdf_path = pdf_path or download_pdf()
        _, docs = load_and_chunk_pdf(pdf_path)
        _, vectorstore = build_faiss(docs)
        bm25 = BM25Okapi([preprocess(d.page_content) for d in docs])
        reranker = CrossEncoder(reranker_model, device=reranker_device)
        _, call_llm = make_llm_client()
        return cls(
            docs=docs,
            vectorstore=vectorstore,
            bm25=bm25,
            reranker=reranker,
            call_llm=call_llm,
        )

    def query(self, question: str, top_k: int = 5, fetch_k: int = 20) -> Dict:
        # 1. DENSE retrieval
        sem_docs = self.vectorstore.similarity_search(question, k=fetch_k)

        # 2. SPARSE retrieval (BM25)
        bm_scores = self.bm25.get_scores(preprocess(question))
        bm_idx = np.argsort(bm_scores)[::-1][:fetch_k]
        kw_docs = [self.docs[i] for i in bm_idx]

        # 3. RRF fusion
        fused = [x["doc"] for x in rrf([sem_docs, kw_docs])]

        # 4. Cross-encoder rerank
        pairs = [(question, d.page_content) for d in fused]
        ce_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(fused, ce_scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:top_k]]

        # 5. Build prompt & call LLM
        context = build_context(top_docs)
        answer = self.call_llm(RAG_PROMPT.format(context=context, question=question))
        return {
            "answer": answer,
            "retrieved_chunks": top_docs,
            "pages_used": sorted({d.metadata.get("page", "?") for d in top_docs}),
        }


# ── CLI demo ─────────────────────────────────────────────────────────
DEMO_QUERIES = [
    "What is the main idea of this paper?",
    "What is the Transformer architecture?",
    "What datasets were used in the experiments?",
    "Who are the authors of this paper?",  # now succeeds thanks to BM25
]


def main() -> None:
    print("🔧 Building Advanced RAG pipeline …")
    rag = AdvancedRAG.build()
    print(f"✅ Ready — {len(rag.docs)} chunks · FAISS + BM25 + Cross-Encoder\n")

    for q in DEMO_QUERIES:
        print("═" * 70)
        print(f" Question: {q}")
        print("═" * 70)
        out = rag.query(q)
        print(f" Pages used: {out['pages_used']}")
        print(f"\n Answer:\n{out['answer']}\n")


if __name__ == "__main__":
    main()
