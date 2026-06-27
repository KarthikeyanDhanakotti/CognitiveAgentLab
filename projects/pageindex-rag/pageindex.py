"""
PageIndex vs Chunked RAG — Employee Policy Q&A
===============================================
LLM   : Groq (free tier) — llama-3.3-70b-versatile
Embed : sentence-transformers (local, fully free — no API needed)

──────────────────────────────────────────────────────────────────────
HOW TO RUN (PowerShell)
──────────────────────────────────────────────────────────────────────
Step 1 — Get a free Groq API key:
    https://console.groq.com/keys

Step 2 — Open PowerShell and run:

    cd "C:\Users\kartdh\OneDrive - Microsoft\Desktop\Model\Domain_Intent"

    $env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

    # Full demo: PageIndex vs ChunkedRAG side-by-side
    .\.venv\Scripts\python.exe page_index_groq.py

    # Interactive mode: ask your own questions
    .\.venv\Scripts\python.exe page_index_groq.py interactive

    # Use with a real PDF (pass the path as argument)
    .\.venv\Scripts\python.exe page_index_groq.py path\to\your.pdf

Note: First run downloads the embedding model (~22 MB) automatically.
──────────────────────────────────────────────────────────────────────

Install:
    pip install groq sentence-transformers scikit-learn numpy PyMuPDF

Get a free Groq API key at: https://console.groq.com/keys

Set env variable:
    $env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

Scenario
--------
A 6-page HR handbook where each page is one complete policy topic
(parental leave, expenses, remote work, etc.).

Problem with Chunked RAG:
  - A 300-char chunk may cut a table in half or split a numbered list mid-step.
  - GPT / Llama then gets fragments and has to guess the missing context.

Solution with PageIndex:
  - Each full page is embedded as one unit.
  - Retrieval returns complete pages — tables and steps are always intact.
  - Llama reads the whole page exactly as a human would.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Clients ──────────────────────────────────────────────────────────────────

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Local embedding model — downloaded once, runs offline after that.
# all-MiniLM-L6-v2  : 22 MB, fast, good for semantic search
# all-mpnet-base-v2  : 420 MB, higher quality (swap if you want better accuracy)
print("Loading embedding model (first run downloads ~22 MB) …")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.\n")

CHAT_MODEL = "llama-3.3-70b-versatile"   # best free Llama on Groq (128k context)


# ── Synthetic HR handbook (simulates pages parsed from a PDF) ────────────────

HANDBOOK_PAGES: List[dict] = [
    {
        "page": 1,
        "title": "Welcome & Company Overview",
        "content": textwrap.dedent("""\
            Welcome to Contoso Corp!
            We are a technology company founded in 2005. Our mission is to empower
            every person on the planet to achieve more. This handbook covers all
            policies effective January 2025. For questions contact hr@contoso.com.
        """),
    },
    {
        "page": 2,
        "title": "Parental Leave Policy",
        "content": textwrap.dedent("""\
            PARENTAL LEAVE POLICY

            Eligibility: Employees who have completed 6 months of continuous service.

            Entitlements:
            ┌─────────────────────────────┬──────────────┐
            │ Leave Type                  │ Duration     │
            ├─────────────────────────────┼──────────────┤
            │ Maternity (birth/adoption)  │ 26 weeks     │
            │ Paternity                   │ 6 weeks      │
            │ Shared parental leave       │ Up to 37 wks │
            └─────────────────────────────┴──────────────┘

            How to apply:
            1. Notify your manager at least 8 weeks before the start date.
            2. Submit form HR-PL-01 to the HR portal.
            3. HR will confirm entitlement within 5 business days.
            4. Payroll adjustments are processed automatically.

            Pay during leave: 100% for the first 16 weeks, 50% thereafter.
        """),
    },
    {
        "page": 3,
        "title": "Expense Reimbursement",
        "content": textwrap.dedent("""\
            EXPENSE REIMBURSEMENT POLICY

            Eligible expenses: Travel, accommodation, client meals, and training costs
            pre-approved by your manager.

            Limits:
            • Hotel      : up to $250/night (major cities) or $180/night (other).
            • Meals      : up to $75/day (no alcohol on company account).
            • Air travel : economy class unless flight > 6 hours.

            Steps to claim:
            1. Collect all original receipts (photos accepted for items < $25).
            2. Log into the Expense Portal (portal.contoso.com/expenses).
            3. Submit the claim within 30 days of the expense date.
            4. Your manager approves within 5 business days.
            5. Finance processes payment in the next payroll cycle.

            Non-reimbursable: Fines, personal entertainment, mini-bar charges.
        """),
    },
    {
        "page": 4,
        "title": "Remote Work Policy",
        "content": textwrap.dedent("""\
            REMOTE WORK POLICY

            Contoso supports a hybrid work model. Employees may work remotely up to
            3 days per week with manager approval.

            Requirements:
            • Maintain a distraction-free, ergonomic home workspace.
            • Be available on Teams during core hours: 10 AM – 3 PM local time.
            • Attend all mandatory on-site days (currently Tuesday and Thursday).
            • Use a company-approved VPN when accessing internal systems.

            Equipment: Contoso provides a laptop and one monitor. Additional peripherals
            are the employee's responsibility unless approved as an accessibility need.

            Approval: Submit remote work agreement (form HR-RW-05) annually.
        """),
    },
    {
        "page": 5,
        "title": "Code of Conduct",
        "content": textwrap.dedent("""\
            CODE OF CONDUCT

            All employees must uphold Contoso's values: Respect, Integrity, Innovation.

            Key expectations:
            • Treat colleagues, customers, and partners with dignity.
            • Report conflicts of interest to your manager or Ethics Hotline.
            • Protect confidential information — do not share externally without approval.
            • Zero tolerance for harassment, discrimination, or retaliation.

            Violations:
            Violations are investigated by HR and may result in disciplinary action
            up to and including termination. Reports can be made anonymously via
            ethics.contoso.com or by calling 1-800-555-0199 (24/7).
        """),
    },
    {
        "page": 6,
        "title": "Annual Leave & Public Holidays",
        "content": textwrap.dedent("""\
            ANNUAL LEAVE

            Accrual:
            • Years 0–2:  18 days / year
            • Years 3–5:  22 days / year
            • Years 6+:   26 days / year

            Rules:
            1. Leave must be approved by your manager at least 2 weeks in advance.
            2. Maximum carry-over is 5 days into the next calendar year.
            3. Unused leave beyond carry-over is forfeited (not paid out).
            4. Sick leave (10 days/year) is separate and does not accrue.

            Public Holidays: 11 federal holidays; employees in states with additional
            mandated holidays receive those too. See the HR portal for the full list.
        """),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: local embeddings via sentence-transformers
# ═══════════════════════════════════════════════════════════════════════════════

def embed(texts: List[str]) -> np.ndarray:
    """Return a (N, D) float32 embedding matrix using the local sentence model."""
    vectors = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: call Groq Llama
# ═══════════════════════════════════════════════════════════════════════════════

def llama_chat(system: str, user: str) -> str:
    """Send a chat request to Groq and return the reply text."""
    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PageIndex  — one embedding per page
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageNode:
    page_num  : int
    title     : str
    content   : str
    embedding : np.ndarray = field(default_factory=lambda: np.array([]))


class PageIndex:
    """
    Index where each document page is a single retrieval unit.

    Benefits over chunked RAG:
    ✔ Tables / numbered steps on the same page stay together.
    ✔ No chunk-boundary artefacts (mid-sentence / mid-table splits).
    ✔ Each retrieved unit is human-readable without extra stitching.
    ✔ Natural citation: "See Page 3" rather than "chunk_47".
    ✔ Ideal for structured docs: policies, contracts, technical manuals.
    """

    def __init__(self) -> None:
        self.nodes: List[PageNode] = []

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, pages: List[dict]) -> None:
        """
        Embed every page and store PageNodes.
        pages: list of {"page": int, "title": str, "content": str}
        """
        print(f"[PageIndex] Building index for {len(pages)} pages …")
        texts      = [f"{p['title']}\n\n{p['content']}" for p in pages]
        embeddings = embed(texts)

        for page_dict, vec in zip(pages, embeddings):
            self.nodes.append(
                PageNode(
                    page_num  = page_dict["page"],
                    title     = page_dict["title"],
                    content   = page_dict["content"],
                    embedding = vec,
                )
            )
        print(f"[PageIndex] Ready — {len(self.nodes)} pages indexed.\n")

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[PageNode, float]]:
        """Return the top-k most relevant pages with cosine similarity scores."""
        query_vec  = embed([query])                              # (1, D)
        index_vecs = np.stack([n.embedding for n in self.nodes]) # (N, D)
        scores     = cosine_similarity(query_vec, index_vecs)[0] # (N,)

        ranked = sorted(
            zip(self.nodes, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    # ── Generate answer ──────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 2) -> str:
        """Retrieve relevant pages, then generate an answer with Llama via Groq."""
        hits = self.retrieve(question, top_k=top_k)

        context_parts = []
        for node, score in hits:
            context_parts.append(
                f"=== Page {node.page_num}: {node.title} (score={score:.3f}) ===\n"
                f"{node.content}"
            )
        context = "\n\n".join(context_parts)

        print(f"  [PageIndex]  Retrieved pages : {[n.page_num for n, _ in hits]}")
        print(f"               Similarity scores: {[round(s, 3) for _, s in hits]}")

        system = (
            "You are an HR assistant. Answer the employee's question using ONLY "
            "the policy pages provided. If a table is present, preserve its structure "
            "in your answer. Be concise and cite the page number at the end."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"

        return llama_chat(system, user)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ChunkedRAG  — for side-by-side comparison
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    page_num  : int
    chunk_idx : int
    text      : str
    embedding : np.ndarray = field(default_factory=lambda: np.array([]))


def _split_into_chunks(text: str, max_chars: int = 300) -> List[str]:
    """Naive word-boundary chunking that simulates token-level chunking."""
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(" ".join(current)) >= max_chars:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


class ChunkedRAG:
    """Traditional RAG: pages split into fixed-size chunks — included for comparison."""

    def __init__(self, chunk_size_chars: int = 300) -> None:
        self.chunks: List[Chunk] = []
        self.chunk_size = chunk_size_chars

    def build(self, pages: List[dict]) -> None:
        print(f"[ChunkedRAG] Splitting {len(pages)} pages into ~{self.chunk_size}-char chunks …")
        all_texts, meta = [], []

        for page in pages:
            full_text   = f"{page['title']}\n\n{page['content']}"
            page_chunks = _split_into_chunks(full_text, self.chunk_size)
            for idx, chunk_text in enumerate(page_chunks):
                meta.append((page["page"], idx))
                all_texts.append(chunk_text)

        embeddings = embed(all_texts)
        for (page_num, chunk_idx), text, vec in zip(meta, all_texts, embeddings):
            self.chunks.append(
                Chunk(page_num=page_num, chunk_idx=chunk_idx, text=text, embedding=vec)
            )

        print(f"[ChunkedRAG] Ready — {len(self.chunks)} chunks from {len(pages)} pages.\n")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        query_vec  = embed([query])
        index_vecs = np.stack([c.embedding for c in self.chunks])
        scores     = cosine_similarity(query_vec, index_vecs)[0]
        ranked = sorted(
            zip(self.chunks, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def query(self, question: str, top_k: int = 3) -> str:
        hits = self.retrieve(question, top_k=top_k)

        context_parts = []
        for chunk, score in hits:
            context_parts.append(
                f"--- Page {chunk.page_num}, chunk {chunk.chunk_idx} "
                f"(score={score:.3f}) ---\n{chunk.text}"
            )
        context = "\n\n".join(context_parts)

        print(f"  [ChunkedRAG] Retrieved chunks from pages: "
              f"{[c.page_num for c, _ in hits]}")
        print(f"               Similarity scores: {[round(s, 3) for _, s in hits]}")

        system = (
            "You are an HR assistant. Answer using ONLY the provided text snippets. "
            "Be concise."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"

        return llama_chat(system, user)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Demo — compare PageIndex vs ChunkedRAG on HR questions
# ═══════════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    "What is the parental leave entitlement and how do I apply?",
    "How do I claim expenses for a business trip and what are the hotel limits?",
    "How many annual leave days do I get after 4 years of service?",
]

SEP = "═" * 72


def run_demo() -> None:
    # Build both indexes
    page_index  = PageIndex()
    chunked_rag = ChunkedRAG(chunk_size_chars=300)

    page_index.build(HANDBOOK_PAGES)
    chunked_rag.build(HANDBOOK_PAGES)

    for q in QUESTIONS:
        print(f"\n{SEP}")
        print(f"  QUESTION: {q}")
        print(SEP)

        print("\n[PageIndex Answer]")
        pi_answer = page_index.query(q, top_k=2)
        print(textwrap.indent(pi_answer.strip(), "  "))

        print("\n[ChunkedRAG Answer]")
        rag_answer = chunked_rag.query(q, top_k=3)
        print(textwrap.indent(rag_answer.strip(), "  "))

    print(f"\n{SEP}")
    print("  SUMMARY: PageIndex vs ChunkedRAG")
    print(SEP)
    print(textwrap.dedent("""\
      PageIndex wins when:
        • Pages contain tables, numbered steps, or structured lists.
        • Each page naturally covers one complete topic.
        • You want clean, citable answers ("See Page 3").

      ChunkedRAG wins when:
        • Pages are very long (>2 000 tokens) — exceed the LLM context window.
        • Free-form prose where any 500-token window is self-contained.
        • You need sub-page precision (e.g., one clause from a 40-clause page).
    """))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Bonus: Drop-in PDF loader  (replaces HANDBOOK_PAGES with a real PDF)
# ═══════════════════════════════════════════════════════════════════════════════

def load_pdf_pages(pdf_path: str) -> List[dict]:
    """
    Parse a real PDF into page-dicts for PageIndex.

    pip install PyMuPDF
    Usage:
        pages = load_pdf_pages("employee_handbook.pdf")
        pi = PageIndex()
        pi.build(pages)
        print(pi.query("What is the parental leave policy?"))
    """
    import fitz  # PyMuPDF

    pages = []
    doc   = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:           # skip blank / image-only pages
            continue
        title = text.splitlines()[0][:60]   # first line as title proxy
        pages.append({"page": i, "title": title, "content": text})
    doc.close()
    print(f"Loaded {len(pages)} text pages from {pdf_path}")
    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Interactive Q&A (run from terminal)
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_qa() -> None:
    """
    After building the index, let you ask custom questions in a loop.
    Type 'exit' to quit.
    """
    pi = PageIndex()
    pi.build(HANDBOOK_PAGES)

    print("\nInteractive Q&A — PageIndex + Groq Llama")
    print("Type your question or 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = pi.query(question, top_k=2)
        print(f"\nLlama: {answer}\n")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    # ── Usage ──────────────────────────────────────────────────────────────────
    # python page_index_groq.py                     → full demo (default)
    # python page_index_groq.py interactive          → interactive Q&A loop
    # python page_index_groq.py path\to\file.pdf    → Q&A on a real PDF
    # ──────────────────────────────────────────────────────────────────────────

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "interactive":
            interactive_qa()
        elif arg.endswith(".pdf"):
            # PDF mode: build PageIndex from real PDF, then go interactive
            import os
            if not os.path.exists(arg):
                print(f"File not found: {arg}")
                sys.exit(1)
            pdf_pages = load_pdf_pages(arg)
            pi = PageIndex()
            pi.build(pdf_pages)
            print(f"\nPDF loaded — {len(pdf_pages)} pages indexed.")
            print("Type your question or 'exit' to quit.\n")
            while True:
                question = input("You: ").strip()
                if question.lower() in {"exit", "quit", "q"}:
                    print("Goodbye!")
                    break
                if not question:
                    continue
                answer = pi.query(question, top_k=2)
                print(f"\nLlama: {answer}\n")
        else:
            print(f"Unknown argument: {arg}")
            print("Usage:")
            print("  python page_index_groq.py                  # demo")
            print("  python page_index_groq.py interactive       # Q&A loop")
            print("  python page_index_groq.py file.pdf          # Q&A on PDF")
    else:
        run_demo()
