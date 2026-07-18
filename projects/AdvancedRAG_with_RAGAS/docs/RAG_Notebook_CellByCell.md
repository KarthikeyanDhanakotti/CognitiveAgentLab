# RAG_AdvancedRAG — Cell-by-Cell Walkthrough

This document explains **every cell** in `RAG_AdvancedRAG (2).ipynb`: what the code does, **why** it is needed, and **why this choice and not another**.

The notebook builds two pipelines on the *"Attention Is All You Need"* PDF and grades both with RAGAS:

1. **Naive RAG** – PDF → chunk → embed → FAISS → LLM
2. **Advanced RAG** – Hybrid retrieval (FAISS + BM25) → RRF fusion → Cross-Encoder rerank → LLM
3. **RAGAS evaluation** – faithfulness, answer relevancy, context precision, context recall

**New here?** Start with [RAG_Algorithms_DeepDive.md § 0](RAG_Algorithms_DeepDive.md#0-end-to-end-what-happens-when-you-query-an-llm) — it walks through the full end-to-end flow of what happens when a user's query hits the LLM, for a plain LLM, naive RAG, and advanced RAG. Then come back here for the cell-by-cell detail.

For deep algorithmic reasoning (BM25 vs TF-IDF, cosine vs Euclidean, RRF vs weighted sum, etc.) see [RAG_Algorithms_DeepDive.md](RAG_Algorithms_DeepDive.md).

---

## Cell 1 — Nuclear dependency fix

```python
!pip uninstall -y -q ragas langchain langchain-core langchain-community ...
!pip install -q "ragas>=0.2.10,<0.3" "langchain>=0.3,<0.4" ...
```

**What it does:** Uninstalls whatever Colab ships preinstalled, then installs a compatible matrix of `ragas` + `langchain*`.

**Why it is required:** Colab preloads an **old `ragas 0.1.x`** that hard-imports `langchain_community.chat_models.ChatVertexAI`. That symbol was removed in `langchain-community 0.3+`, so `import ragas` throws `ImportError`. Uninstall + reinstall is the only reliable fix inside a running kernel.

**Why version *ranges* not exact pins:** Pins (`==0.2.10`) break the day a bug-fix release ships. Ranges (`>=0.2.10,<0.3`) stay compatible while still preventing a major-version jump.

**Why these packages:**
| Package | Purpose |
|---|---|
| `ragas` | LLM-as-judge evaluation framework |
| `langchain*` | Document loaders, splitters, vector-store wrappers |
| `langchain-groq` | LLM client for the free Groq API |
| `sentence-transformers` | Runs the MiniLM embedder locally |
| `faiss-cpu` | Dense vector index (Facebook AI Similarity Search) |
| `rank_bm25` | Sparse keyword ranker |
| `pymupdf` | Fast, accurate PDF text extraction |
| `datasets` | HuggingFace `Dataset` — required input format for RAGAS |
| `gradio`, `matplotlib` | Optional UI + charts |

---

## Cell 2 — Markdown intro & prerequisites

Documents Python version, RAM, `GROQ_API_KEY` sources, and quick-start steps. No code impact.

---

## Cell 3 — Second install cell (idempotent safety net)

Reinstalls the same set with `-q`. Safe to re-run after a runtime restart without redoing Cell 1's uninstall. Also pins `requests>=2.32.4,<2.33` to avoid a Colab-side conflict warning.

---

## Cell 4 — Imports

```python
import gradio, os, re, numpy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq
```

**Why `PyMuPDFLoader` and not `PyPDFLoader`:** PyMuPDF (`fitz`) preserves layout better on multi-column academic PDFs (this notebook uses one). PyPDF often merges columns line-by-line, producing garbled text.

**Why `RecursiveCharacterTextSplitter`:** It splits on a hierarchy of separators (`\n\n → \n → " " → ""`) instead of blindly cutting mid-word. This preserves paragraphs and sentences, which are the natural semantic units.

**Why import from `langchain_huggingface` and not `langchain_community`:** In LangChain ≥ 0.3, HF integrations moved to their own package. The old import still works but throws a `DeprecationWarning`.

---

## Cell 5 — Portable API-key loader

Tries four sources in order:
1. `os.environ["GROQ_API_KEY"]` — already exported
2. Google Colab secrets (`google.colab.userdata`)
3. `.env` file via `python-dotenv`
4. Interactive `getpass()` prompt

**Why the fallback chain:** the notebook must run identically on Colab, local Jupyter, VS Code, and Docker without editing code.

**Why `getpass` and not `input`:** `getpass` hides the key from the terminal echo and Jupyter cell output — the key never lands in a saved `.ipynb`.

---

## Cell 6 — LLM client wrapper

```python
client = Groq(api_key=os.environ["GROQ_API_KEY"])
def call_llm(prompt): ...
    model="llama-3.1-8b-instant", temperature=0.2
```

**Why Groq:** Free tier with sub-second latency (LPU inference). No OpenAI key needed.

**Why `llama-3.1-8b-instant` for generation:** Fastest model on Groq, ample free quota. Big enough (8B) to synthesize retrieved chunks into a coherent answer. The heavier `llama-3.3-70b` is reserved for the RAGAS judge where reasoning quality matters more.

**Why `temperature=0.2` and not 0:** Slight non-determinism reduces "stuck-in-a-loop" phrasing on repeated calls, but stays close to greedy for reproducibility.

---

## Cell 7 — Load PDF (auto-download)

Downloads `arxiv.org/pdf/1706.03762.pdf` to `/content` (Colab) or a tmp dir (local). Runs `PyMuPDFLoader(path).load()`.

**Why load per page (not one giant string):** Each `Document` object keeps `metadata["page"]`, which powers the `[Page X]` citations in the final answer.

---

## Cell 8 — Chunking

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)
```

**Why chunk at all:** LLM context windows are limited and expensive. Retrieval quality also drops when the "unit of meaning" is too big — the top-k hit becomes noisy.

**Why 1000 chars / 200 overlap:** Empirically the sweet spot for research papers:
- 1000 chars ≈ 200–250 tokens ≈ 1–2 paragraphs → dense enough to answer factoid questions
- 20% overlap (200 chars) prevents an answer sentence from being *split* across chunks — a common cause of "Not found in document" errors

**Why not fixed 512 tokens like BERT:** BERT's limit does not apply here — the embedder (MiniLM) and the LLM (Llama) both accept much more. 1000-char chunks yield better recall.

**Why store `chunk_id` and `page` in metadata:** Enables citations and later debug/diagnostic cells.

---

## Cell 9 — Chunking trade-off demo

Splits the *same* page at `(200/20)`, `(1000/200)`, `(3000/300)` to visualise:
- Too small → context bleeds across boundaries → **recall drops**
- Too large → signal drowned by noise → **precision drops**

Purely instructive — output helps students *see* why 1000/200 is the default recommendation.

---

## Cell 10 — Embed + store in FAISS

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
```

**Why `all-MiniLM-L6-v2`:**
| Property | Value |
|---|---|
| Dimensions | 384 |
| Size on disk | ~90 MB |
| Speed | ~14 000 sentences/sec on CPU |
| Quality | Within ~3% of models 10× larger on MTEB retrieval |

**What is an embedding, really?** A neural-network function that turns any text into a fixed-length list of 384 numbers (a **vector**) such that texts with *similar meaning* end up as vectors that are *close together* in space. The model was fine-tuned with contrastive learning on millions of `(query, relevant_passage)` pairs so that paraphrases move a tiny distance and unrelated topics move far apart. **Why RAG needs this:** it lets the retriever match a question like *"How do I cancel my subscription?"* to a passage titled *"Steps to terminate your plan"* even though they share no words. Pure keyword search cannot. Full explanation — what embeddings are, how they're trained, the anatomy of a vector, why 384 dims, symmetric vs asymmetric models, and their systematic weaknesses — in [RAG_Algorithms_DeepDive.md § 2](RAG_Algorithms_DeepDive.md#2-embeddings--why-all-minilm-l6-v2).

**Why not OpenAI `text-embedding-3-small`:** Requires a paid API key and network round-trip per query. MiniLM is free, local, and fast enough that the workshop runs offline after first download.

**Why FAISS:** Facebook's C++ library — fastest single-machine ANN index available (billions of vectors in production at Meta). For this 200-chunk demo it uses an exact flat index; for millions of vectors you'd switch to `IndexIVFFlat` or `IndexHNSW`.

**Why not Chroma / Pinecone / Weaviate:** All three are excellent, but FAISS has zero server/setup, zero cost, and is embedded in-process — perfect for a notebook demo.

**Uses cosine similarity under the hood.** See [RAG_Algorithms_DeepDive.md#cosine-vs-euclidean-vs-dot-product](RAG_Algorithms_DeepDive.md#cosine-similarity-vs-euclidean-vs-dot-product) for why.

**Which ANN algorithm does FAISS use here?** For ~200 chunks it uses the exact **Flat** index (brute-force but instant). At scale you'd swap to **HNSW** (graph-based, sub-ms latency), **IVF** (inverted file, partition-and-search), or **IVF-PQ** (compressed for billions of vectors). See [RAG_Algorithms_DeepDive.md § 4a](RAG_Algorithms_DeepDive.md#4a-ann-algorithms--the-engines-inside-every-vector-db) for a full tour of ANN algorithms (Flat, IVF, HNSW, PQ, LSH, ScaNN, DiskANN) and which vector DB uses which.

---

## Cell 11 — Naive RAG function

```python
def naive_rag(query, k=5):
    retrieved = vectorstore.similarity_search(query, k=k)
    context = "\n\n---\n\n".join(f"[Page {d.metadata['page']}]\n{d.page_content}" for d in retrieved)
    prompt = NAIVE_PROMPT.format(context=context, question=query)
    return call_llm(prompt)
```

**Why `k=5`:** Enough context for the LLM to synthesize, small enough to stay within the token budget and avoid "lost-in-the-middle" attention decay.

**Why the "Not found in document" instruction:** Anti-hallucination guardrail. Without it, the LLM will invent an answer from parametric memory when retrieval misses.

**Why `[Page X]` citations in the prompt:** Forces grounded, verifiable answers. If a user disputes the answer, they can jump to that page.

---

## Cell 12 — Demo the 4 questions

Runs `naive_rag` on:
- "What is the main idea of this paper?" → works (semantically rich query)
- "What is the Transformer architecture?" → works
- "What datasets were used?" → mostly works
- **"Who are the authors of this paper?" → FAILS ("Not found in document")**

The failure is intentional — it motivates Part 2.

---

## Cell 13 — Markdown: "Naive RAG hit a wall"

Explains that MiniLM encodes **meaning**, not **exact tokens**. Proper nouns ("Vaswani", "Shazeer") have no semantic neighborhood in an attention paper, so pure vector search misses them.

---

## Cell 14 — Diagnostic cell

Prints FAISS's actual top-5 for the authors query and checks whether the correct chunk appears. It won't. This is the pedagogical "aha" moment before adding BM25.

---

## Cell 15 — Markdown: Part 2 header

Introduces the three upgrades: hybrid retrieval, RRF fusion, cross-encoder rerank.

---

## Cell 16 — Advanced RAG imports (already imported, no-op)

Kept so the cell can be run standalone after a kernel restart.

---

## Cell 17 — LLM client (re-declared)

Same wrapper as Cell 6, re-declared so Part 2 runs independently.

---

## Cell 18 — Load PDF (Part 2)

Reuses `PDF_PATH`. Same idea as Cell 7.

---

## Cell 19 — Chunk (Part 2)

Identical to Cell 8 — ensures the docs list matches the BM25 index built next.

---

## Cell 20 — Build **three** indexes

```python
# 3a. DENSE
vectorstore = FAISS.from_documents(docs, embeddings)

# 3b. SPARSE
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()
bm25 = BM25Okapi([preprocess(d.page_content) for d in docs])

# 3c. RERANKER
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
```

**Why BM25 (and not TF-IDF or plain keyword match):** BM25 fixes two flaws of TF-IDF — (1) term-frequency saturation so a word appearing 100× doesn't dominate, and (2) document-length normalization so short chunks aren't unfairly penalized. Details in [RAG_Algorithms_DeepDive.md#bm25-vs-tf-idf](RAG_Algorithms_DeepDive.md#why-bm25-and-not-tf-idf).

**Why the preprocess function (lowercase + strip punctuation):** BM25 is a bag-of-words model — case and punctuation add noise. Lowercasing gives `Vaswani` = `vaswani`; stripping punctuation prevents `"attention,"` and `"attention"` from being treated as different tokens.

**Why a cross-encoder for rerank (not another bi-encoder):**
- **Bi-encoder** (used for initial retrieval): encodes query and doc *separately*, compares vectors. Fast, scalable, but loses interaction signal.
- **Cross-encoder** (used for rerank): concatenates `[query, doc]` into one BERT input and outputs a relevance score. Much more accurate but O(N) per query — that's why we only rerank the top ~20, not the whole corpus.

**Why `ms-marco-MiniLM-L-6-v2` specifically:** Trained on the MS-MARCO passage-ranking dataset — the standard benchmark for query-passage relevance. L-6 keeps it lightweight enough for CPU inference.

**Why `device="cpu"`:** Guaranteed to work everywhere. On a GPU box you'd set `device="cuda"` for ~10× speedup.

---

## Cell 21 — RRF fusion

```python
def rrf(result_lists, k=60):
    scores = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = hash(doc.page_content)
            scores.setdefault(key, {"doc": doc, "score": 0.0})
            scores[key]["score"] += 1.0 / (rank + k + 1)
    return sorted(scores.values(), key=lambda x: x["score"], reverse=True)
```

**Why RRF and not a weighted score sum (e.g. `0.5·cos + 0.5·bm25`):** BM25 and cosine live on totally different scales (BM25 is unbounded positive, cosine is [-1, 1]). Combining them requires per-corpus tuning of weights and normalization. RRF ignores the raw scores entirely and only looks at **rank position**, so it needs zero tuning and works across arbitrary rankers.

**Why `k=60`:** The value from the original Cormack et al. 2009 paper. It de-emphasizes the very top rank slightly so a rank-1 in one ranker doesn't automatically win against a rank-2 in both.

**Why `hash(doc.page_content)`:** Deduplicates the same chunk showing up in both lists — you want its scores *added*, not counted twice as separate entries.

Details: [RAG_Algorithms_DeepDive.md#why-rrf](RAG_Algorithms_DeepDive.md#why-rrf-reciprocal-rank-fusion).

---

## Cell 22 — Advanced RAG pipeline

```python
def advanced_rag(query, top_k=5, fetch_k=20):
    sem_docs = vectorstore.similarity_search(query, k=fetch_k)      # dense
    bm_idx   = np.argsort(bm25.get_scores(preprocess(query)))[::-1][:fetch_k]
    kw_docs  = [docs[i] for i in bm_idx]                            # sparse
    fused    = [x["doc"] for x in rrf([sem_docs, kw_docs])]         # RRF
    pairs    = [(query, d.page_content) for d in fused]
    ce_scores = reranker.predict(pairs)                             # rerank
    ranked   = sorted(zip(fused, ce_scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:top_k]]
    ...
```

**Why the two-stage funnel (`fetch_k=20`, `top_k=5`):**
- Stage 1: cheap retrievers cast a wide net (20 candidates each)
- Stage 2: expensive cross-encoder only scores those ~40 unique candidates
- Final: LLM only sees the 5 most relevant

This is the standard **retrieve-and-rerank** pattern used in Google, Bing, and modern RAG systems.

**Why not skip fusion and just concat the two lists:** Concatenation loses the ordering signal — BM25's rank-1 result deserves more weight than its rank-15. RRF preserves that.

**Why rerank after fusion (not before):** The cross-encoder is the most expensive step. Running it on the pre-fusion candidates would double the work.

---

## Cell 23 — Demo Part 2

Re-runs the same 4 queries. The authors query now **succeeds** because BM25 caught the exact name tokens.

---

## Cell 24 — Markdown: "The Reveal"

Frames the next diagnostic cell.

---

## Cell 25 — Side-by-side diagnostic

Prints the top-5 from **Dense only**, **Sparse only**, and **Advanced (Hybrid + Rerank)** for the authors query. Visually proves:
- Dense misses (as before)
- Sparse finds page 1 immediately (exact-token match)
- Advanced returns the correct chunk at rank 1 (rerank promoted it)

---

## Cell 26 — Markdown: Part 3 header

Announces RAGAS evaluation.

---

## Cell 27 — Golden test set

Hand-authored `ground_truth` for each of the 4 questions. **Required** — RAGAS's `context_recall` and `answer_correctness` are supervised metrics; they compare against these gold answers.

**Why only 4 questions:** Free-tier judge quota. In production you'd have 50–500.

---

## Cell 28 — Run both pipelines on the test set

Builds two dicts (`naive_rows`, `adv_rows`) each shaped like `{question, answer, contexts, ground_truth}` — the exact schema RAGAS expects.

**Why `contexts` is a list of strings:** RAGAS scores each retrieved chunk individually for `context_precision` (was this chunk useful?) and aggregates.

---

## Cell 29 — Configure the RAGAS judge

```python
JUDGE_MODEL = "llama-3.1-8b-instant"
judge_llm = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(embeddings)  # reuse MiniLM
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
```

**Why `temperature=0` on the judge:** Reproducible scoring. Any drift here corrupts the comparison between pipelines.

**Why reuse MiniLM for judge embeddings:** `answer_relevancy` needs to embed the generated answer against synthetic questions — MiniLM is enough and free.

**Why the 4 metrics chosen:**
| Metric | Question it answers | Formula (intuition) |
|---|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? (detects hallucination) | `# claims in answer supported by context / # total claims` |
| **Answer relevancy** | Does the answer address the question? (detects off-topic responses) | mean cosine of `question` vs `N` questions the judge reverse-generates from the answer |
| **Context precision** | Are the retrieved chunks relevant? (retriever quality — signal) | rank-weighted average of precision@k using judge's per-chunk relevance |
| **Context recall** | Did we retrieve all chunks needed for the ground truth? (retriever quality — coverage) | `# statements in ground_truth supported by contexts / # total statements` |

Together they cover both **retriever** and **generator** quality — see [RAG_Algorithms_DeepDive.md § 9](RAG_Algorithms_DeepDive.md#9-ragas-metrics) for worked examples, other RAGAS metrics (answer correctness, semantic similarity, entities recall, noise sensitivity, aspect critique), score-interpretation thresholds, and a symptom → root-cause table for diagnosing which stage of the pipeline is failing.

---

## Cell 30 — Full RAGAS evaluation (self-contained)

Same as Cell 29 but with guards (`raise RuntimeError` if prerequisites missing) so it's safe to run after a kernel restart. Runs `evaluate(...)` on both pipelines and concatenates into `df_all`.

**Why guards:** Common Colab UX issue — users skip cells and get cryptic `NameError`. Guards give an actionable message instead.

---

## Cell 31 — Averaged summary + improvement table

Aggregates per-pipeline means and prints:
- Absolute lift (`Advanced − Naive`)
- Percentage lift (`(Advanced − Naive) / Naive`)

Typical result: **+50–150 % on faithfulness and precision**.

---

## Cell 32 — Bar chart

Matplotlib comparison chart. `axhline(y=0.7)` draws a "production threshold" — a common empirical bar for shippable RAG quality.

---

## Cells 33–35 — Per-question breakdown

Detects column names (RAGAS 0.1 used `question`, 0.2+ uses `user_input`), then pivots `df_all` into a per-question × per-metric table so you can see *which* question drove the biggest gain (usually the authors one).

---

## Cell 36 — Markdown recap + troubleshooting

Summary table + fixes for the top 5 common failure modes.

---

## Summary — why the Advanced pipeline wins

| Weakness in Naive RAG | Fix in Advanced RAG |
|---|---|
| Semantic embeddings miss exact tokens (names, IDs, code) | **BM25** as a second retriever |
| Two rankers give incompatible scores | **RRF** merges by rank, not raw score |
| Top-k from retrieval is still noisy | **Cross-encoder** rerank as precision filter |
| No way to prove the pipeline is better | **RAGAS** measures the lift on 4 axes |

The measurable improvement is what makes the Advanced pipeline production-worthy — not just "it feels better".
