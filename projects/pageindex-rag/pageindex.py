"""
pageindex.py — Page-Level Retrieval for RAG Systems
=====================================================
Index full document pages (not chunks) so tables, numbered steps,
and structured content are never split across retrieval boundaries.

LLM   : Groq  — llama-3.3-70b-versatile  (free tier)
Embed : sentence-transformers             (local, no API cost)

Quick start:
    pip install -r requirements.txt
    $env:GROQ_API_KEY = "gsk_xxxx"
    python pageindex.py                   # demo
    python pageindex.py interactive       # Q&A loop
    python pageindex.py your_doc.pdf      # use your own PDF
"""

from __future__ import annotations

import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
CHAT_MODEL    = "llama-3.3-70b-versatile"   # best free Llama on Groq (128k ctx)
EMBED_MODEL   = "all-MiniLM-L6-v2"          # 22 MB local model, no API needed
DEFAULT_TOP_K = 2                            # pages to retrieve per query


# ── Clients (lazy-loaded) ─────────────────────────────────────────────────────

_groq_client: Groq | None = None
_embedder: SentenceTransformer | None = None


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY not set.\n"
                "Get a free key at https://console.groq.com/keys\n"
                "Then run:  $env:GROQ_API_KEY = 'gsk_xxxx'"
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading embedding model (downloads ~22 MB on first run) ...")
        _embedder = SentenceTransformer(EMBED_MODEL)
        print("Embedding model ready.\n")
    return _embedder


# ── Core types ────────────────────────────────────────────────────────────────

@dataclass
class PageNode:
    """Represents one indexed page."""
    page_num  : int
    title     : str
    content   : str
    embedding : np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class RetrievalResult:
    """A retrieved page with its similarity score."""
    node  : PageNode
    score : float

    def __str__(self) -> str:
        return (
            f"=== Page {self.node.page_num}: {self.node.title} "
            f"(score={self.score:.3f}) ===\n{self.node.content}"
        )


# ── PageIndex ─────────────────────────────────────────────────────────────────

class PageIndex:
    """
    Indexes documents at the page level.

    Each page becomes one vector in the index. Retrieval returns complete
    pages — preserving tables, numbered steps, and structured content that
    chunked RAG would split across multiple fragments.

    Usage:
        pi = PageIndex()
        pi.build(pages)            # pages = list of {"page", "title", "content"}
        answer = pi.query("...")
    """

    def __init__(self) -> None:
        self.nodes: List[PageNode] = []

    # ── Index building ────────────────────────────────────────────────────────

    def build(self, pages: List[dict]) -> "PageIndex":
        """
        Embed every page and build the index.

        Args:
            pages: list of dicts with keys: page (int), title (str), content (str)

        Returns:
            self  (for chaining)
        """
        if not pages:
            raise ValueError("pages list is empty")

        print(f"[PageIndex] Building index for {len(pages)} page(s) ...")
        texts = [f"{p['title']}\n\n{p['content']}" for p in pages]
        vecs  = get_embedder().encode(texts, show_progress_bar=False,
                                      convert_to_numpy=True).astype(np.float32)

        self.nodes = [
            PageNode(
                page_num  = p["page"],
                title     = p["title"],
                content   = p["content"],
                embedding = v,
            )
            for p, v in zip(pages, vecs)
        ]
        print(f"[PageIndex] Ready — {len(self.nodes)} pages indexed.\n")
        return self

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievalResult]:
        """
        Find the most relevant pages for a query.

        Args:
            query : natural language question
            top_k : number of pages to return

        Returns:
            list of RetrievalResult sorted by similarity (highest first)
        """
        if not self.nodes:
            raise RuntimeError("Index is empty. Call build() first.")

        qvec   = get_embedder().encode([query], convert_to_numpy=True).astype(np.float32)
        ivecs  = np.stack([n.embedding for n in self.nodes])
        scores = cosine_similarity(qvec, ivecs)[0]

        ranked = sorted(
            zip(self.nodes, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [RetrievalResult(node=n, score=s) for n, s in ranked[:top_k]]

    # ── Answer generation ─────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = DEFAULT_TOP_K,
              verbose: bool = True) -> str:
        """
        Retrieve relevant pages and generate an answer using Groq Llama.

        Args:
            question : user's question
            top_k    : number of pages to retrieve
            verbose  : print retrieved page numbers and scores

        Returns:
            answer string from Llama
        """
        hits = self.retrieve(question, top_k=top_k)

        if verbose:
            print(f"  Retrieved pages : {[h.node.page_num for h in hits]}")
            print(f"  Scores          : {[round(h.score, 3) for h in hits]}")

        context = "\n\n".join(str(h) for h in hits)

        response = get_groq().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question "
                        "using ONLY the provided document pages. If a table or "
                        "numbered list is present, preserve its structure. "
                        "End your answer by citing the page number(s) used."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document pages:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ── PDF loader ────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> List[dict]:
    """
    Parse a PDF into page dicts for PageIndex.

    Args:
        pdf_path : path to the PDF file

    Returns:
        list of {"page", "title", "content"} dicts (blank pages skipped)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF required:  pip install PyMuPDF")

    pages, doc = [], fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        title = text.splitlines()[0][:60]
        pages.append({"page": i, "title": title, "content": text})
    doc.close()

    print(f"Loaded {len(pages)} text page(s) from: {pdf_path}\n")
    return pages


# ── Sample data (HR handbook) ─────────────────────────────────────────────────

SAMPLE_PAGES: List[dict] = [
    {
        "page": 1,
        "title": "Welcome & Company Overview",
        "content": textwrap.dedent("""\
            Welcome to Contoso Corp!
            Founded in 2005. Our mission: empower every person to achieve more.
            This handbook covers all policies effective January 2025.
            Questions? Contact hr@contoso.com
        """),
    },
    {
        "page": 2,
        "title": "Parental Leave Policy",
        "content": textwrap.dedent("""\
            PARENTAL LEAVE POLICY

            Eligibility: Employees who have completed 6 months of continuous service.

            Entitlements:
            | Leave Type                  | Duration     |
            |-----------------------------|--------------|
            | Maternity (birth/adoption)  | 26 weeks     |
            | Paternity                   | 6 weeks      |
            | Shared parental leave       | Up to 37 wks |

            How to apply:
            1. Notify your manager at least 8 weeks before the start date.
            2. Submit form HR-PL-01 to the HR portal.
            3. HR confirms entitlement within 5 business days.
            4. Payroll adjustments are processed automatically.

            Pay during leave: 100% first 16 weeks, 50% thereafter.
        """),
    },
    {
        "page": 3,
        "title": "Expense Reimbursement",
        "content": textwrap.dedent("""\
            EXPENSE REIMBURSEMENT POLICY

            Eligible: Travel, accommodation, client meals, pre-approved training.

            Limits:
            | Category    | Limit                                    |
            |-------------|------------------------------------------|
            | Hotel       | $250/night (major cities), $180 (other)  |
            | Meals       | $75/day (no alcohol)                     |
            | Air travel  | Economy (business if flight > 6 hours)   |

            Steps to claim:
            1. Collect all receipts (photos ok for items < $25).
            2. Log in at portal.contoso.com/expenses.
            3. Submit within 30 days of the expense date.
            4. Manager approves within 5 business days.
            5. Finance pays in the next payroll cycle.

            Non-reimbursable: Fines, personal entertainment, mini-bar charges.
        """),
    },
    {
        "page": 4,
        "title": "Remote Work Policy",
        "content": textwrap.dedent("""\
            REMOTE WORK POLICY

            Hybrid model: Up to 3 remote days/week with manager approval.

            Requirements:
            - Distraction-free, ergonomic home workspace.
            - Available on Teams: 10 AM – 3 PM local time (core hours).
            - Mandatory on-site days: Tuesday and Thursday.
            - VPN required for all internal systems.

            Equipment: Contoso provides laptop + one monitor.
            Additional peripherals are employee's responsibility unless
            approved as an accessibility need.

            Approval: Submit form HR-RW-05 annually.
        """),
    },
    {
        "page": 5,
        "title": "Annual Leave",
        "content": textwrap.dedent("""\
            ANNUAL LEAVE

            Accrual by tenure:
            | Years of Service | Days / Year |
            |------------------|-------------|
            | 0 – 2 years      | 18 days     |
            | 3 – 5 years      | 22 days     |
            | 6+ years         | 26 days     |

            Rules:
            1. Request approval at least 2 weeks in advance.
            2. Max carry-over: 5 days into next calendar year.
            3. Unused leave beyond carry-over is forfeited (not paid out).
            4. Sick leave (10 days/year) is separate and does not accrue.

            Public Holidays: 11 federal holidays + state-mandated extras.
        """),
    },
]


# ── Demo ──────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "What is the parental leave entitlement and how do I apply?",
    "What are the hotel expense limits and how do I claim a business trip?",
    "How many annual leave days do I get after 4 years of service?",
]

SEP = "=" * 68


def run_demo(pages: List[dict] = None) -> None:
    """Run PageIndex on sample (or provided) pages and print answers."""
    pages = pages or SAMPLE_PAGES

    pi = PageIndex()
    pi.build(pages)

    for q in DEMO_QUESTIONS:
        print(f"\n{SEP}")
        print(f"  Q: {q}")
        print(SEP)
        answer = pi.query(q)
        print(f"\n  A: {textwrap.indent(answer.strip(), '     ')}\n")


def run_interactive(pages: List[dict] = None) -> None:
    """Interactive Q&A loop — type questions, get answers, type 'exit' to quit."""
    pages = pages or SAMPLE_PAGES

    pi = PageIndex()
    pi.build(pages)

    print("PageIndex Interactive Q&A  |  Powered by Groq Llama-3.3-70B")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        answer = pi.query(question)
        print(f"\nLlama: {answer}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Usage:
        python pageindex.py                   # demo with sample HR handbook
        python pageindex.py interactive       # interactive Q&A with sample data
        python pageindex.py path/to/file.pdf  # Q&A on your own PDF
    """
    args = sys.argv[1:]

    if not args:
        run_demo()

    elif args[0] == "interactive":
        run_interactive()

    elif args[0].lower().endswith(".pdf"):
        pdf_path = args[0]
        if not os.path.exists(pdf_path):
            print(f"Error: file not found — {pdf_path}")
            sys.exit(1)
        pdf_pages = load_pdf(pdf_path)
        run_interactive(pages=pdf_pages)

    else:
        print(__doc__)
        print("Unknown argument:", args[0])
        sys.exit(1)


if __name__ == "__main__":
    main()
