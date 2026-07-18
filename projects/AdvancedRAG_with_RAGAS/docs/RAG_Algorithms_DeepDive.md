# RAG Algorithms — Deep Dive & Design Choices

Companion to [RAG_Notebook_CellByCell.md](RAG_Notebook_CellByCell.md).
This document answers the "**why this and not that**" questions for every algorithm used in the notebook.

## Contents
0. [End-to-end: what happens when you query an LLM (with & without RAG)](#0-end-to-end-what-happens-when-you-query-an-llm)
1. [Chunking strategy](#1-chunking-strategy)
2. [Embeddings — why MiniLM](#2-embeddings--why-all-minilm-l6-v2)
   - [2a. What is an embedding?](#2a-what-is-an-embedding)
   - [2b. Why are embeddings required for RAG?](#2b-why-are-embeddings-required-for-rag)
   - [2c. How does an embedding model actually work?](#2c-how-does-an-embedding-model-actually-work)
   - [2d. Anatomy of an embedding vector](#2d-anatomy-of-an-embedding-vector)
   - [2e. Bigger dimensions = better embeddings?](#2e-bigger-dimensions--better-embeddings)
   - [2f. Different flavors of embedding models](#2f-different-flavors-of-embedding-models)
   - [2g. The concrete flow inside `HuggingFaceEmbeddings`](#2g-the-concrete-flow-inside-huggingfaceembeddings)
   - [2h. What embeddings do *not* do well (and why hybrid is needed)](#2h-what-embeddings-do-not-do-well-and-why-hybrid-is-needed)
   - [2i. Model comparison table (retrieval-focused)](#2i-model-comparison-table-retrieval-focused)
   - [2j. Practical checklist when picking an embedder](#2j-practical-checklist-when-picking-an-embedder)
3. [Cosine similarity vs Euclidean vs Dot product](#3-cosine-similarity-vs-euclidean-vs-dot-product)
4. [FAISS vs other vector stores](#4-faiss-vs-chroma-vs-pinecone-vs-weaviate)
   - [4a. ANN algorithms — the engines inside every vector DB](#4a-ann-algorithms--the-engines-inside-every-vector-db) *(Flat · IVF · HNSW · PQ · LSH · ScaNN · DiskANN)*
   - [4b. Which algorithm does each vector DB actually use?](#4b-which-algorithm-does-each-vector-db-actually-use)
   - [4c. How to pick an index — decision table](#4c-how-to-pick-an-index--decision-table)
   - [4d. Distance metric ↔ index compatibility](#4d-distance-metric--index-compatibility)
   - [4e. Filtering: pre-filter vs post-filter](#4e-filtering-pre-filter-vs-post-filter)
   - [4f. What about the vector *database* features (beyond the index)?](#4f-what-about-the-vector-database-features-beyond-the-index)
5. [Why BM25 and not TF-IDF](#5-why-bm25-and-not-tf-idf)
6. [Why hybrid (dense + sparse) retrieval](#6-why-hybrid-dense--sparse)
7. [Why RRF (Reciprocal Rank Fusion)](#7-why-rrf-reciprocal-rank-fusion)
8. [Cross-encoder reranking](#8-cross-encoder-reranking-vs-bi-encoder)
9. [RAGAS metrics explained](#9-ragas-metrics)
10. [Why Groq + Llama and not OpenAI](#10-why-groq--llama)

---

## 0. End-to-end: what happens when you query an LLM

Before diving into the individual algorithms, it helps to see the full path a user's question takes through the system — first with a **plain LLM** (no retrieval), then with the **naive RAG** pipeline from the notebook, then with the **advanced RAG** pipeline. Each level adds one problem-solving layer on top of the previous.

---

### 0.1 Plain LLM (no retrieval) — the baseline

This is what happens if you paste a question directly into ChatGPT / Llama / Claude with no external data.

```
    ┌────────────┐
    │  User Q    │  "Who are the authors of the Attention Is All You Need paper?"
    └─────┬──────┘
          │
          ▼
    ┌────────────────────────────┐
    │ 1. Tokenizer               │  Q → [15496, 546, 262, 6156, ...]
    │    (BPE / SentencePiece)   │  splits text into ~50k-vocab integer IDs
    └─────────────┬──────────────┘
                  │
                  ▼
    ┌────────────────────────────┐
    │ 2. Embedding layer         │  each token ID → learned vector (d≈4096)
    └─────────────┬──────────────┘
                  │
                  ▼
    ┌────────────────────────────┐
    │ 3. Transformer stack       │  32–120 layers of
    │    (self-attention + MLP)  │    self-attention → feed-forward
    │                            │  every token "sees" every earlier token
    └─────────────┬──────────────┘
                  │
                  ▼
    ┌────────────────────────────┐
    │ 4. LM head                 │  final hidden state → 50k-way softmax
    │    (next-token probability)│  → probability distribution over vocab
    └─────────────┬──────────────┘
                  │
                  ▼
    ┌────────────────────────────┐
    │ 5. Sampling                │  argmax / top-p / temperature → 1 token
    └─────────────┬──────────────┘
                  │
                  ▼
    ┌────────────────────────────┐
    │ 6. Autoregressive loop     │  append token, feed back, generate next...
    │    until <eos> or max_tok  │  until stop or length limit
    └─────────────┬──────────────┘
                  │
                  ▼
              "The authors are Vaswani, Shazeer, ...  ← may be fact or hallucination"
```

**Where the answer comes from:** the model's **parametric memory** — the ~8–70 billion weights learned during training. It never sees the actual PDF; it only has whatever it memorized.

**What can go wrong:**

| Failure mode | Cause |
|---|---|
| **Hallucination** — invents authors that don't exist | Training data didn't contain the fact, but the model guesses fluently |
| **Stale data** — quotes an old CEO / deprecated API | Training cutoff was months/years ago |
| **No source attribution** — can't be verified | Answer is dissolved into weights, no citation possible |
| **No private data** — can't answer about your internal wiki | Never saw it in pre-training |

This is exactly the problem RAG is designed to fix.

---

### 0.2 Naive RAG (the notebook's Part 1) — retrieve, then generate

Now the exact same question with a retrieval step bolted on. This is what the notebook builds in Cells 1–14.

```
    ┌────────────┐
    │  User Q    │  "Who are the authors?"
    └─────┬──────┘
          │
          ▼
 ┌─────────────────────────┐
 │ A. EMBED THE QUERY      │  MiniLM(Q) → [0.13, -0.22, ..., 0.09]  (384-dim vector)
 │    (same encoder used   │  same model that embedded the chunks
 │     to build the index) │
 └───────────┬─────────────┘
             │
             ▼
 ┌─────────────────────────┐
 │ B. SEARCH THE VECTOR DB │  FAISS: compute cosine(q, every chunk)
 │    (FAISS similarity_   │           → sort → return top-k = 5
 │     search, k=5)        │  ~200 chunks × 384 dims = a matrix multiply
 └───────────┬─────────────┘
             │
             ▼
 ┌─────────────────────────┐  chunk 47 (Page 3) — attention formula
 │ C. TOP-k CHUNKS         │  chunk 12 (Page 2) — encoder-decoder overview
 │    (raw text + page #)  │  chunk 88 (Page 5) — training details
 │                         │  chunk 03 (Page 1) — abstract       ← close, still misses authors
 │                         │  chunk 71 (Page 4) — attention math
 └───────────┬─────────────┘
             │
             ▼
 ┌────────────────────────────────────────────────┐
 │ D. BUILD THE PROMPT (template + variables)     │
 │                                                │
 │  "You are a precise assistant. Answer using    │
 │   ONLY the context below. If not found,        │
 │   say 'Not found in document'. Cite [Page X]." │
 │                                                │
 │  Context:                                      │
 │    [Page 3] <chunk 47 text>                    │
 │    [Page 2] <chunk 12 text>                    │
 │    ...                                         │
 │                                                │
 │  Question: Who are the authors?                │
 │  Answer:                                       │
 └───────────────────────┬────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────┐
 │ E. SEND TO LLM          │  Groq API → llama-3.1-8b-instant
 │    (same 6 steps as 0.1,│  the ENTIRE prompt (system + context +
 │     but with a much     │   question) is tokenized and processed
 │     bigger prompt now)  │
 └───────────┬─────────────┘
             │
             ▼
 ┌─────────────────────────┐
 │ F. STREAMED ANSWER      │  "Not found in document."   ← honest failure
 │    + page citations     │  (retrieval missed the authors chunk)
 └─────────────────────────┘
```

**What changed vs 0.1:**
- The model no longer relies on memory — it reads a **fresh, curated context** every time.
- If the retriever is right → the answer is **grounded + citable**.
- If the retriever is wrong (as here) → the model **admits it** instead of hallucinating (thanks to the "Not found" instruction).

**But there's still a failure:** the naive retriever couldn't find the authors chunk because semantic embeddings don't neighbor proper nouns well. This is exactly why the notebook adds Part 2.

---

### 0.3 Advanced RAG (the notebook's Part 2) — hybrid + rerank

Same question, richer pipeline. Cells 15–25.

```
    ┌────────────┐
    │  User Q    │
    └─────┬──────┘
          │
          ├──────────────────────────┐
          │                          │
          ▼                          ▼
 ┌─────────────────┐        ┌─────────────────┐
 │ A1. DENSE       │        │ A2. SPARSE      │
 │   MiniLM(Q)     │        │   preprocess(Q) │
 │        │        │        │        │        │
 │        ▼        │        │        ▼        │
 │   FAISS top-20  │        │   BM25 top-20   │
 │   (semantic)    │        │   (keyword)     │
 └────────┬────────┘        └────────┬────────┘
          │                          │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ B. RRF FUSION            │  score(d) = Σ 1 / (60 + rank_r(d))
          │    (merge two ranked     │  no normalization needed —
          │     lists by rank only)  │  ranks are scale-free
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ C. FUSED LIST (~40)      │  union of both, dedup'd,
          │                          │  ordered by RRF score
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ D. CROSS-ENCODER RERANK  │  for each (Q, chunk):
          │    ms-marco-MiniLM-L-6   │    joint BERT forward pass
          │    (precision filter)    │    → single relevance score
          │                          │  sort by score
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ E. TOP-5 CHUNKS          │  Page 1 (authors!) ← now rank 1
          │                          │  Page 3 (attention)
          │                          │  ...
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ F. BUILD PROMPT + LLM    │  same as 0.1 steps D–F
          └────────────┬─────────────┘
                       │
                       ▼
          "The authors are Ashish Vaswani, Noam Shazeer, Niki Parmar,
           Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser,
           and Illia Polosukhin. [Page 1]"                       ← success
```

**The three added layers, each fixing a specific failure mode:**

| Layer | Fixes |
|---|---|
| **BM25** in parallel with FAISS | Semantic embeddings miss exact tokens (proper nouns, IDs, code) |
| **RRF fusion** | The two rankers produce incompatible scores; RRF ignores scores and only uses ranks |
| **Cross-encoder rerank** | Even after fusion, top-k is noisy; joint encoding boosts precision at position 1 |

Notice the **LLM step is identical** in all three pipelines — the model isn't smarter, the *input* is smarter. This is the core RAG insight: **improving the retrieved context improves the answer more reliably than improving the model**.

---

### 0.4 Where the time actually goes

Approximate breakdown for a single query on this notebook (CPU, ~200 chunks, Groq API for the LLM):

| Stage | Naive RAG | Advanced RAG |
|---|---|---|
| Embed query (MiniLM) | ~15 ms | ~15 ms |
| Vector search (FAISS Flat) | <1 ms | <1 ms |
| BM25 scan | — | ~5 ms |
| RRF merge | — | <1 ms |
| Cross-encoder rerank (~40 pairs) | — | ~1 000 ms *(CPU, biggest cost)* |
| LLM generation (Groq) | ~500 ms | ~500 ms |
| **Total** | **~0.5 s** | **~1.5 s** |

**Takeaway:** Advanced RAG is ~3× slower per query — almost entirely because of the cross-encoder. On GPU that drops to ~50 ms and the difference vanishes. The retrieval side (dense + sparse + fusion) is essentially free.

---

### 0.5 Six things to remember

1. **The LLM does not "know" your data.** It only sees the tokens you paste into the prompt.
2. **Retrieval quality caps answer quality.** If the right chunk isn't in the top-k, the LLM cannot recover it — and if instructed correctly it will admit "not found" rather than invent one.
3. **Same query, different question at each stage.** The embedder answers "which chunks are semantically near?", BM25 answers "which chunks share tokens?", the reranker answers "which chunks actually answer this question?", the LLM answers "given these chunks, what should I say?".
4. **Every layer is optional but cumulative.** You can ship Naive RAG and get 60–70 % of the value. Hybrid + rerank push you to 90 %+.
5. **You need RAGAS (or equivalent) to know it works.** Without measurement you're just guessing which pipeline is better. That's Part 3 of the notebook.
6. **Latency vs quality is a knob you tune per use case.** Chat/copilot → prefer fast (Naive or Hybrid without cross-encoder). Legal / medical / finance → prefer accurate (full Advanced + big LLM judge).

---

## 1. Chunking strategy

**The problem:** an embedding vector represents an *entire* chunk of text as one point. If the chunk is too big, the vector averages many concepts and matches nothing well. If it's too small, the answer sentence gets cut across two chunks and neither is a strong match.

**The trade-off curve:**

| Chunk size | Recall | Precision | Notes |
|---|---|---|---|
| ~200 chars | Low | High | Sentence-level; answer often split across boundary |
| **~1000 chars** | **Balanced** | **Balanced** | **Notebook default** — paragraph-sized |
| ~3000 chars | High | Low | Whole-section; vector meaning gets diluted |

**Why `RecursiveCharacterTextSplitter` and not fixed-size splitting:**

| Splitter | Behaviour |
|---|---|
| `CharacterTextSplitter` | Cuts at exactly N chars — often mid-word |
| **`RecursiveCharacterTextSplitter`** | Tries `\n\n` first, then `\n`, then `" "`, then `""`. Preserves paragraphs and sentences. |
| `TokenTextSplitter` | Splits by tokenizer output — accurate for LLM budget but slower and requires a tokenizer |
| `SemanticChunker` | Splits at points where consecutive-sentence embeddings diverge. Higher quality but 100× slower |

Recursive is the pragmatic default. Semantic chunking is worth trying once your MVP is working.

**Why 200 chars of overlap:** ~20% overlap prevents "answer split at boundary" while only inflating the index by 20%. Below 10% recall drops noticeably; above 30% you're just paying storage for duplicates.

---

## 2. Embeddings — why `all-MiniLM-L6-v2`

### 2a. What is an embedding?

An **embedding** is a fixed-length list of numbers (a **vector**) that represents a piece of text (or an image, or audio, or code) in a way that captures its **meaning**. Formally, an embedding model is a function

$$f : \text{text} \rightarrow \mathbb{R}^d$$

that turns any input string into a point in a $d$-dimensional space (e.g. $d = 384$ for MiniLM). The critical property is:

> **Semantically similar texts are mapped to points that are close together in the vector space; unrelated texts are mapped far apart.**

**Toy example (imagine $d = 3$ for illustration):**

| Sentence | Embedding | Distance |
|---|---|---|
| "The cat sat on the mat" | `[0.82, 0.11, 0.55]` | — |
| "A feline rested on the rug" | `[0.79, 0.14, 0.58]` | 0.04 ✅ close |
| "The stock market crashed today" | `[-0.31, 0.68, -0.22]` | 1.42 ❌ far |

Notice the paraphrase lands *nearly on top of* the original even though it shares **no words** — that's the magic. A keyword search would score it 0 (no overlap); the embedding score is essentially 1.

**Real embeddings are just bigger.** MiniLM outputs 384 numbers per input, OpenAI's `text-embedding-3-large` outputs 3072. The principle is identical.

### 2b. Why are embeddings required for RAG?

Without embeddings, retrieval is limited to **lexical matching** (exact words). That fails on:

| User query | Passage in the docs | Lexical match? | Embedding match? |
|---|---|:---:|:---:|
| "How do I cancel my subscription?" | "Steps to terminate your plan…" | ❌ | ✅ |
| "What's the CEO's compensation?" | "Executive salary disclosure…" | ❌ | ✅ |
| "Fix broken login" | "Authentication troubleshooting" | ❌ | ✅ |
| "Vaswani" | "Vaswani et al., 2017" | ✅ | ✅ |

The first three would leave a keyword-only system blind. Embeddings bridge the vocabulary mismatch between how users phrase questions and how documents phrase answers — often called the **"lexical gap"** or **"vocabulary mismatch problem"**.

Concretely, the RAG pipeline needs embeddings for **three separate reasons**:

1. **Retrieval.** Index every chunk as a vector; embed the user's query; return the $k$ nearest chunks by cosine similarity. This is what FAISS does.
2. **Reranking (bi-encoder variant).** The cross-encoder we use is *not* an embedding model — it's a joint encoder. But other rerankers (ColBERT, Cohere Rerank) use late-interaction embeddings.
3. **Evaluation.** RAGAS's `answer_relevancy` metric embeds generated & synthetic questions and takes cosine similarity — so the evaluator itself is embedding-powered.

Everything downstream — the LLM's context window, the citations, the RAGAS scores — flows from the fact that the embedder puts the right chunk near the query in vector space.

### 2c. How does an embedding model actually work?

Modern text embedders are **transformer neural networks** trained with a **contrastive objective**. Here's the recipe used for models like MiniLM, MPNet, and BGE:

1. **Start with a pre-trained BERT-family transformer.** It already understands English structure from masked-language-model pre-training on billions of tokens.
2. **Add a pooling head.** BERT outputs one vector per token; you need one vector per input. Common choice: mean-pool the token vectors, then L2-normalize.
3. **Fine-tune with contrastive learning.** Show the model millions of pairs:
   - **Positives:** `(query, relevant_passage)`, `(sentence, paraphrase)`, `(question, its_answer)` from datasets like MS MARCO, NLI, SNLI, Reddit, StackExchange.
   - **Negatives:** random passages, or (much better) **hard negatives** — passages that look similar but are wrong.
4. **Loss:** typically **InfoNCE** / **multiple-negatives-ranking loss**:

   $$L = -\log \frac{\exp(\text{sim}(q, p^+) / \tau)}{\sum_{p \in \text{batch}} \exp(\text{sim}(q, p) / \tau)}$$

   Read as: "make positives *more similar* than every negative in the batch, scaled by temperature $\tau$."
5. **Result:** the network learns a projection where meaning-preserving transformations (paraphrase, translation, synonym swap) move you a tiny distance, while topic changes move you far.

The trained model runs **fast at inference** — one forward pass per input — because the network is small (MiniLM has 22M parameters vs GPT-4's ~1.8 trillion).

**Why this matters:** because embeddings are trained with cosine similarity as the objective, **cosine is the mathematically correct metric to compare them at query time** (see § 3).

### 2d. Anatomy of an embedding vector

A single embedding is just a numpy array, but the following properties are worth understanding:

| Property | MiniLM value | Meaning |
|---|---|---|
| **Dimension** ($d$) | 384 | Number of floats per vector. Higher = more expressive but more RAM & slower search. |
| **Data type** | float32 | 4 bytes per number → 384 × 4 = 1536 B per chunk. Some DBs quantize to int8 (4× smaller). |
| **Norm** | 1.0 (unit length) | MiniLM L2-normalizes internally → cosine ≡ dot product. |
| **Interpretability** | ~none | Individual dimensions have no human meaning. The information lives in the *direction* of the whole vector. |
| **Determinism** | fully | Same input → same vector (assuming same model version). |

**Storage math:** 200 chunks × 384 dims × 4 bytes = **~300 KB** for the whole demo notebook. 1M chunks × 1536 dims × 4 bytes = **~6 GB** — this is where compression (PQ, quantization) starts to matter.

### 2e. Bigger dimensions = better embeddings?

Not linearly. There are diminishing returns and real costs:

| $d$ | Typical model | Pros | Cons |
|---|---|---|---|
| 128–384 | MiniLM, small BGE | Fast, small, CPU-friendly | Struggles on nuanced semantic tasks |
| 512–768 | MPNet, BGE-base | Sweet spot for most RAG | 2× slower, 2× RAM |
| 1024–1536 | BGE-large, OpenAI-3-small | High MTEB scores | Heavy on CPU, ANN indexes get slower |
| 3072+ | OpenAI-3-large, Cohere-v3 | Best quality, multilingual | Expensive, requires GPU or API |

**Why not just always pick the largest?**
- **Latency:** ANN indexes scan a vector at O($d$) per hop — 3072 is 8× slower than 384.
- **Storage:** 8× more RAM/disk.
- **Curse of dimensionality:** in very high $d$, the *ratio* between nearest and average distance shrinks, making rankings less discriminative if the model isn't well-trained.
- **Marginal quality gap:** on chunk-retrieval MTEB tasks, going from 384 → 3072 typically buys +15 points but costs 8× more.

For a workshop demo, 384 (MiniLM) is more than enough.

### 2f. Different flavors of embedding models

Not every model is trained the same way. You'll see these terms:

| Flavor | What it optimizes | Example | Use for |
|---|---|---|---|
| **Symmetric** | `sim(A, B) = sim(B, A)` — two sentences of the same kind | `all-mpnet-base-v2` | Sentence similarity, clustering |
| **Asymmetric** | `sim(short_query, long_passage)` | `msmarco-*`, BGE `-en-v1.5` | **RAG retrieval** — query and doc are shaped differently |
| **Multilingual** | Same vector space across 50+ languages | `paraphrase-multilingual-*`, BGE-M3 | Cross-lingual retrieval |
| **Code embeddings** | Understands syntax + semantics of code | `text-embedding-ada-002`, `codesage` | Code search, semantic diff |
| **Instruction-tuned** | Follows prompts like `"Represent this for search: ..."` | Instructor, E5-mistral | Task-specific retrieval |
| **Multi-vector (late interaction)** | One vector per token, not per doc | ColBERT | Precision reranking |
| **Sparse learned** | Outputs sparse dictionary of terms | SPLADE, uniCOIL | Best of both worlds w/ inverted index |
| **Multimodal** | Same space for images + text | CLIP, SigLIP | Image search, cross-modal RAG |

**MiniLM is symmetric** — technically not optimal for query→passage retrieval, but its quality is high enough that the difference is invisible in most workshops. In production RAG, an **asymmetric** model like BGE or E5 is a better default.

### 2g. The concrete flow inside `HuggingFaceEmbeddings`

When the notebook runs:

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
```

…this is what happens under the hood, per chunk:

```
"Attention is all you need. We propose the Transformer..."
        │
        ▼  tokenizer (WordPiece)
[101, 3086, 2003, 2035, 2017, 2342, 1012, ...]      # token IDs
        │
        ▼  6-layer transformer
[[0.11, ...], [-0.03, ...], ...]                    # one 384-dim vector per token
        │
        ▼  mean pooling over tokens
[0.09, -0.15, 0.22, ..., 0.07]                      # single 384-dim vector
        │
        ▼  L2 normalize
[0.14, -0.23, 0.34, ..., 0.11]                      # unit-length vector → stored in FAISS
```

At query time the identical pipeline runs on the user's question. FAISS then compares the query vector against every stored chunk vector via cosine, returns the top-k.

### 2h. What embeddings do *not* do well (and why hybrid is needed)

Embeddings are not magic — they have systematic weaknesses that motivate the notebook's Advanced pipeline:

| Weakness | Example | Fix in the notebook |
|---|---|---|
| **Rare proper nouns** | "Vaswani", "Shazeer" — no training signal to build a meaningful direction | **BM25** (Cell 20) |
| **Codes / IDs / SKUs** | "ERR-4291", "SKU-88712-A" — treated as noise | BM25 |
| **Exact quotes** | Legal, medical — wording matters, not paraphrase | BM25 |
| **Negation** | "not effective" vs "effective" often embed close | Hard to fix; needs stronger model + rerank |
| **Long documents** | Averaging a 3000-token doc → mush | Better chunking |
| **Domain drift** | Model trained on Wikipedia, deployed on medical/legal jargon | Fine-tune or use domain model |
| **Numbers** | "$5 million" vs "$5 billion" often score similar | Extract entities separately |

**Bottom line:** embeddings solve the *paraphrase* problem beautifully but not the *exact-match* problem. That's why every serious RAG system runs a **dense embedder + a sparse retriever (BM25) together** — exactly what Part 2 of the notebook demonstrates.

### 2i. Model comparison table (retrieval-focused)

| Model | Dim | Size | CPU speed | MTEB retrieval avg |
|---|---|---|---|---|
| **`all-MiniLM-L6-v2`** | **384** | **90 MB** | **~14k sent/s** | **~41.9** |
| `all-mpnet-base-v2` | 768 | 420 MB | ~2.8k sent/s | ~43.8 |
| `BAAI/bge-small-en-v1.5` | 384 | 130 MB | ~10k sent/s | ~51.7 |
| `BAAI/bge-base-en-v1.5` | 768 | 440 MB | ~2.5k sent/s | ~53.3 |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3 GB | ~700 sent/s | ~54.3 |
| `intfloat/e5-large-v2` | 1024 | 1.3 GB | ~700 sent/s | ~50.6 |
| `text-embedding-3-small` (OpenAI) | 1536 | API | API | ~62.3 |
| `text-embedding-3-large` (OpenAI) | 3072 | API | API | ~64.6 |
| `voyage-3` (Voyage AI) | 1024 | API | API | ~63.0 |
| `cohere embed-v3` (Cohere) | 1024 | API | API | ~64.5 |

**Why MiniLM for this notebook:**
- Runs on any laptop (CPU-only)
- Free (no API cost)
- Downloads once (90 MB), cached forever
- Quality gap vs bigger models is small on the *chunk-retrieval* task (unlike, say, semantic-similarity benchmarks where BGE crushes MiniLM)

**When to upgrade:** if RAGAS `context_precision` plateaus below 0.7 despite good chunks, swap to `bge-small-en-v1.5` (same dim as MiniLM, ~10 pts higher MTEB) or `bge-base-en-v1.5` — both are drop-in replacements. Only reach for OpenAI/Voyage/Cohere if you need multilingual or top-of-benchmark quality and can afford the API cost.

### 2j. Practical checklist when picking an embedder

1. **Language(s):** English-only → BGE / MiniLM. Multilingual → BGE-M3 or `paraphrase-multilingual-mpnet-base-v2`.
2. **Asymmetric task?** Query short, doc long → BGE / E5 (they have an `instruction` prefix trick for this).
3. **Domain:** general → BGE. Code → CodeSage. Medical → PubMedBERT-embed.
4. **Budget:** free/local → MiniLM/BGE. API OK → OpenAI/Voyage.
5. **Latency requirement:** <10ms → 384-dim. <100ms → 768-dim. Batch overnight → any size.
6. **Data volume:** millions of chunks → smaller dims + PQ compression.
7. **Sanity check:** always run `context_recall` from RAGAS on a small labeled set before committing.

---

## 3. Cosine similarity vs Euclidean vs Dot product

FAISS uses similarity to rank vectors. The three common choices:

| Metric | Formula | Range | What it measures |
|---|---|---|---|
| **Cosine** | $\cos(u,v) = \frac{u \cdot v}{\lVert u \rVert \lVert v \rVert}$ | [-1, 1] | **Angle** between vectors — direction only |
| **Dot product** | $u \cdot v = \sum u_i v_i$ | (-∞, ∞) | Direction *and* magnitude |
| **Euclidean (L2)** | $\lVert u - v \rVert = \sqrt{\sum (u_i - v_i)^2}$ | [0, ∞) | **Straight-line distance** |

**Why cosine wins for text embeddings:**

1. **Magnitude carries no semantic meaning.** A long document and a short paraphrase should be close in *direction* even if their vector norms differ. Cosine ignores magnitude; Euclidean doesn't.

2. **Sentence-transformer models are trained with cosine.** MiniLM, MPNet, BGE — all trained with contrastive losses that push similar-meaning pairs to high *cosine* similarity. Using a different metric at inference throws away that training signal.

3. **Cosine and dot product are equivalent when vectors are L2-normalized.** `sentence-transformers` normalizes by default, so `dot` and `cosine` produce the same ranking — but `cosine` is the semantically correct name.

**Why not Euclidean:**
- A vector at `2 × u` is *identical in meaning* to `u` but Euclidean says it's far away.
- Empirically ~2–5% worse on retrieval benchmarks with normalized embeddings.

**Why not Manhattan / L1:** Same problem as Euclidean — sensitive to magnitude. Only useful for high-dimensional sparse data (which is what BM25 handles anyway).

**Why not Jaccard / Hamming:** Designed for sets/binary vectors, not dense floats.

**Bottom line:** for dense text embeddings, always use cosine (or equivalently, dot product on normalized vectors).

---

## 4. FAISS vs Chroma vs Pinecone vs Weaviate

| Store | Type | Setup | Cost | Best for |
|---|---|---|---|---|
| **FAISS** | In-process C++ lib | `pip install faiss-cpu` | Free | **Notebooks, single-machine, <10M vectors** |
| Chroma | Local server (or in-process) | `pip install chromadb` | Free | Small apps with metadata filtering |
| Pinecone | Managed cloud | API key | $$$ | Production, multi-tenant, global |
| Weaviate | Server (self-hosted or cloud) | Docker or cloud | Free/$$$ | Hybrid search built-in, GraphQL API |
| pgvector | Postgres extension | SQL | Free | Teams already on Postgres |

**Why FAISS for this notebook:**
- Zero infra — imports and runs
- Fastest single-machine ANN library (built by Meta, powers FB search)
- With ~200 chunks we use the exact **`IndexFlatL2`** — perfect accuracy, no approximation

**When to switch:**
- 1M+ vectors → FAISS `IndexHNSW` or `IndexIVFFlat` (approximate but ~50× faster)
- Multi-user prod → Pinecone / Qdrant / Weaviate
- Filter-heavy workloads → Chroma or Weaviate (native metadata filtering)

### 4a. ANN algorithms — the engines inside every vector DB

A "vector database" is really a **thin metadata + persistence layer wrapped around one or more Approximate Nearest Neighbor (ANN) indexes**. The choice of index dominates latency, RAM cost, and recall. All the vector DBs above (FAISS, Chroma, Pinecone, Qdrant, Weaviate, Milvus, pgvector) implement some subset of the algorithms below.

The core problem: given a query vector $q \in \mathbb{R}^d$ and $N$ stored vectors, return the top-$k$ nearest. Doing it exactly is $O(N \cdot d)$ per query → too slow past ~100k vectors. ANN algorithms trade a tiny bit of **recall** (do we return the *actual* nearest neighbors?) for enormous **speedup**.

Every ANN algorithm sits somewhere on this triangle:

```
              ACCURACY (recall)
                    ▲
                   ╱ ╲
                  ╱   ╲
                 ╱     ╲
                ╱       ╲
               ╱         ╲
              ╱           ╲
    SPEED ◄──╱─────────────╲──► RAM / DISK
```

You can pick any two. The right index for you depends on which corner you can compromise on.

---

#### (i) Flat / Brute Force — `IndexFlatL2`, `IndexFlatIP`

**How it works:** compute distance from query to *every* stored vector. Sort. Return top-k.

- **Recall:** 1.0 (exact — this is the ground truth every other index is measured against)
- **Query time:** $O(N \cdot d)$
- **Build time:** 0 (just stores the raw matrix)
- **RAM:** $N \cdot d \cdot 4$ bytes (float32)

**Use when:** $N < 100{,}000$ or as the accuracy baseline for benchmarking approximate indexes. **This is what the notebook uses** (~200 chunks → flat is instant and exact).

**Don't use when:** you have millions of vectors and per-query latency matters.

---

#### (ii) IVF — Inverted File Index (`IndexIVFFlat`)

Inspired by classical information-retrieval inverted indexes.

**How it works:**
1. **Training:** run k-means on a sample to find $n_{\text{list}}$ centroids (typically $\sqrt{N}$).
2. **Indexing:** assign each vector to its nearest centroid → bucket ("posting list").
3. **Query:** find the $n_{\text{probe}}$ centroids nearest the query, then brute-force search *only* inside those buckets.

- **Recall:** tunable via $n_{\text{probe}}$ (probe 1 → fast, low recall; probe all → equals flat)
- **Query time:** $O((n_{\text{probe}} / n_{\text{list}}) \cdot N \cdot d)$
- **Build time:** k-means over a sample — minutes for millions of vectors
- **RAM:** $N \cdot d \cdot 4$ bytes + tiny centroid overhead

**Trade-off knob:** $n_{\text{probe}}$ directly trades recall for latency. Typical value: 8–32 out of $n_{\text{list}} = 1000$.

**Use when:** 1M–100M vectors, RAM is fine, you want simple tuning.

**Don't use when:** high recall (>95 %) matters more than latency — HNSW does better there.

---

#### (iii) HNSW — Hierarchical Navigable Small World (`IndexHNSW`)

The **industry default** for high-recall in-RAM ANN. Used by Qdrant, Weaviate, Elasticsearch, OpenSearch, Milvus, pgvector, and Redis Vector.

**How it works:** builds a multi-layer graph:
- **Bottom layer** contains every vector, each connected to its ~$M$ nearest neighbors.
- **Higher layers** are progressively sparser (like an express highway).
- **Query:** start at the top layer, greedily hop toward the query, descend a layer, repeat. In the bottom layer do a small local search of size `ef_search`.

Think "skip list, but for vectors in high-dim space".

- **Recall:** 0.95–0.99 with default params — the highest of any pure ANN algorithm at reasonable speed
- **Query time:** $O(\log N)$ typically — sub-millisecond even at 100M vectors
- **Build time:** slow — often 10× longer than IVF (each insert walks the graph)
- **RAM:** ~1.5–2× the raw vectors (graph edges)
- **No training** required — supports live inserts

**Parameters:**
- **`M`** (edges per node, typical 16–48): higher → better recall, more RAM
- **`ef_construction`** (search width during build, typical 100–400): higher → better graph quality, slower build
- **`ef_search`** (search width per query, typical 50–200): higher → better recall, higher latency

**Use when:** you can fit the vectors + graph in RAM and want the best speed/recall combo. Sweet spot: 100k–100M vectors.

**Don't use when:** RAM is the bottleneck (billions of vectors, single machine) — use IVF+PQ or DiskANN.

---

#### (iv) PQ — Product Quantization (`IndexPQ`, `IndexIVFPQ`)

A **compression** technique, not a search structure — usually combined with IVF or HNSW.

**How it works:**
1. Split each $d$-dim vector into $m$ sub-vectors (e.g. 768 dims → 8 sub-vectors of 96 dims each).
2. For *each* sub-vector position, run k-means with 256 centroids → build a codebook.
3. Each vector is stored as $m$ centroid IDs (1 byte each) instead of $d \times 4$ bytes.

Example: a 768-dim float vector (3072 B) becomes 8 bytes → **384× compression**.

- **Recall:** slight loss (a few %) that's often invisible in end-to-end RAG quality
- **Query time:** faster than flat because distance is computed via a small precomputed lookup table
- **RAM:** massive reduction (10–100×)
- **Build:** needs training (k-means per sub-vector position)

**Common combo:** `IndexIVFPQ` = IVF for pruning + PQ for compression → billion-vector search in ~1 GB RAM.

**Use when:** RAM/disk cost is the limiting factor. Web-scale (>100M vectors).

**Don't use when:** you need near-exact recall — PQ always loses a bit.

**Variants worth knowing:** **OPQ** (Optimized PQ) rotates the vector space before splitting for better recall. **ScaNN** below uses a related "anisotropic" quantization that's even better.

---

#### (v) LSH — Locality-Sensitive Hashing

**How it works:** design hash functions such that similar vectors are *more likely* to land in the same bucket. Query: hash $q$, look only in matching buckets.

For cosine similarity, the classic scheme is **random hyperplane hashing**: pick $k$ random hyperplanes; each vector's hash is a $k$-bit string of "which side of each plane" it fell on.

- **Recall:** modest (typically 0.7–0.9) unless you use many hash tables
- **Query time:** very fast for exact-duplicate / near-duplicate detection
- **RAM:** low
- **Build:** fast, embarrassingly parallel, supports easy sharding

**Use when:** deduplication, plagiarism detection, or streaming systems where you need cheap hash-table inserts. Rarely the best choice for RAG.

**Don't use when:** you need high recall — HNSW dominates on quality per RAM byte.

---

#### (vi) ScaNN — Google's asymmetric quantizer

Google's ANN library ([research paper](https://arxiv.org/abs/1908.10396)). Powers Google Search embeddings retrieval.

**How it works:** combines partitioning (like IVF) with a **learned anisotropic quantization** that puts more precision along directions that matter most for inner-product ranking. Better recall-per-byte than PQ.

- **Recall:** 0.95+ at very small memory footprint
- **Query time:** among the fastest for large-scale
- **Complexity:** harder to tune, less community familiarity

**Use when:** you're doing web-scale ANN and RAM is precious. Backend of Vertex Matching Engine.

---

#### (vii) DiskANN / Vamana — SSD-resident ANN

Microsoft Research index designed for **billion-scale on a single machine** by keeping most vectors on SSD.

**How it works:** builds a Vamana graph (similar to HNSW but single-layer, optimized for random SSD reads). Query traverses graph, reading a small number of SSD blocks per hop.

- **Recall:** ~0.95 comparable to HNSW
- **Query time:** ~5–10 ms (limited by SSD latency, not compute)
- **RAM:** only ~5 % of the raw vector size (holds compressed vectors + graph "hot set")
- **Cost:** dramatically cheaper — SSD is 10–100× cheaper per GB than RAM

**Use when:** billions of vectors, single/few-machine budget. This is what **Pinecone p2 pods**, **Milvus DiskANN**, and **Turbopuffer** are built on.

---

#### (viii) SPTAG, NGT, Annoy, FLANN — historical & niche

- **Annoy** (Spotify) — forest of random-projection trees. Simple, memory-mapped, no updates. Good for read-only recommendation indexes.
- **NGT** (Yahoo Japan) — graph-based, similar territory as HNSW.
- **SPTAG** (Microsoft) — hybrid tree + graph, powers Bing.
- **FLANN** — classical, largely superseded by HNSW.

Mostly encountered when reading older systems; new deployments almost never pick these over HNSW / IVF-PQ / DiskANN.

---

### 4b. Which algorithm does each vector DB actually use?

| Vector DB | Default index | Also supports |
|---|---|---|
| **FAISS** | Flat | IVF, HNSW, PQ, IVFPQ, OPQ (choose per-index) |
| **Chroma** | HNSW (via hnswlib) | — |
| **Qdrant** | HNSW | Quantization overlay (scalar, PQ, binary) |
| **Weaviate** | HNSW | Flat, PQ |
| **Milvus** | HNSW | IVF variants, DiskANN, GPU indexes |
| **Pinecone** | Proprietary (HNSW-like on s1, DiskANN-like on p2) | — |
| **pgvector** | HNSW (v0.5+) | IVFFlat |
| **Elasticsearch / OpenSearch** | HNSW | — |
| **Redis Vector** | HNSW | Flat |
| **Vertex Matching Engine** | ScaNN | — |
| **Vespa** | HNSW | — |

**Take-away:** HNSW has become the de-facto default. Learn its 3 knobs (`M`, `ef_construction`, `ef_search`) and you can tune 80 % of the market.

---

### 4c. How to pick an index — decision table

| Situation | Recommended index |
|---|---|
| <100k vectors, notebook / prototype | **Flat** (exact, zero setup) |
| 100k–10M, latency-sensitive, RAM fine | **HNSW** |
| 10M–100M, cost-sensitive, RAM constrained | **IVF-PQ** or **IVF-OPQ** |
| 100M–10B, single machine | **DiskANN** |
| Streaming inserts, no rebuild | **HNSW** (supports live inserts) |
| Highest possible recall, RAM no object | **HNSW** with high `M`/`ef` (or Flat) |
| Web-scale, Google infra | **ScaNN** |
| Dedup / near-duplicate | **LSH** |

---

### 4d. Distance metric ↔ index compatibility

Not every metric works with every index. Cheat sheet:

| Metric | Flat | IVF | HNSW | PQ | LSH |
|---|:---:|:---:|:---:|:---:|:---:|
| Cosine (L2-normalized dot) | ✅ | ✅ | ✅ | ✅ | ✅ (random hyperplane) |
| L2 (Euclidean) | ✅ | ✅ | ✅ | ✅ | ✅ (p-stable) |
| Inner Product | ✅ | ✅ | ✅ | ✅ | ⚠️ (needs asymmetric) |
| Hamming (binary) | ✅ | ❌ | ✅ | — | ✅ |
| Manhattan (L1) | ✅ | ⚠️ | ⚠️ | ❌ | ⚠️ |

The notebook uses **cosine on Flat** — the most compatible combination.

---

### 4e. Filtering: pre-filter vs post-filter

Real-world queries often need metadata filters (`WHERE user_id = 42 AND date > ...`). Two strategies:

- **Post-filter:** run ANN → filter results. Fast but can return <k results if filter is selective.
- **Pre-filter:** compute the filter, then restrict ANN to matching vectors. Complex — most graph indexes (HNSW) don't natively support it without special tricks.
- **Hybrid:** Qdrant and Weaviate implement "filterable HNSW" — annotate graph nodes with metadata and prune during traversal.

If your workload is filter-heavy (multi-tenant, RBAC), pick a DB with native filtered ANN: **Qdrant**, **Weaviate**, **Milvus**, **pgvector**.

---

### 4f. What about the vector *database* features (beyond the index)?

A production vector DB adds these on top of the ANN algorithm:

| Feature | Why it matters |
|---|---|
| **Persistence** (WAL, snapshots) | Survive restarts without re-indexing |
| **Metadata store** (JSON per vector) | Attach tenant IDs, timestamps, tags |
| **Hybrid search** (dense + sparse) | Built-in BM25 alongside ANN — Qdrant, Weaviate, Elasticsearch |
| **Sharding / replication** | Horizontal scale, HA |
| **Multi-tenancy / RBAC** | SaaS use cases |
| **API + client SDKs** | Python, JS, Go, etc. |
| **Live updates** | Add/delete without full rebuild |

**FAISS gives you the index only** — everything above is your responsibility. That's exactly why it's perfect for a notebook (nothing to run) and wrong for production (you'd rebuild all of the above).

---

## 5. Why BM25 and not TF-IDF

Both are **sparse** (bag-of-words) retrievers based on term frequency and inverse document frequency. BM25 is TF-IDF's successor, fixing its two biggest flaws.

### TF-IDF formula
$$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log\frac{N}{\text{df}(t)}$$

Two problems:
1. **Linear TF.** A word appearing 100 times contributes 100× as much as appearing once. Real relevance saturates — after 5–10 mentions, extra ones add little info.
2. **No length normalization.** A 5000-word doc that mentions "attention" 10 times looks equally relevant as a 50-word doc mentioning it 10 times. Intuitively the short doc is far more focused.

### BM25 formula
$$\text{BM25}(t, d) = \text{IDF}(t) \cdot \frac{\text{tf}(t,d) \cdot (k_1 + 1)}{\text{tf}(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Where:
- $k_1 \approx 1.5$ → controls **TF saturation** (denominator caps the effect of extra occurrences)
- $b \approx 0.75$ → controls **length normalization** (0 = ignore length, 1 = full normalization)

**Why BM25 wins:**

| Concern | TF-IDF | BM25 |
|---|---|---|
| TF saturation | Linear — 100× hits dominate | Asymptotic — extra hits barely help |
| Length bias | Punishes short docs | Balanced via `b` parameter |
| Tuning knobs | None | `k1`, `b` |
| Real-world use | Legacy | **Default in Elasticsearch, Lucene, Solr, Vespa** |

**Why not just embed queries into MiniLM and skip sparse:** Dense retrievers *cannot* reliably match rare tokens (proper nouns, product SKUs, error codes, function names) because those tokens have no training co-occurrence to build a meaningful embedding around. The notebook proves this: the authors query fails with FAISS alone but succeeds when BM25 is added.

**Why not use both TF-IDF and BM25:** They rank the same way for most queries — no diversity gain. Combining a dense (semantic) and a sparse (lexical) retriever gives complementary signals.

**Alternatives to BM25:**
- **SPLADE / uniCOIL** — learned sparse retrievers, better quality but need a GPU
- **BM25F** — BM25 that weights document *fields* differently (e.g. title × 3, body × 1)
- **BM25+** — small tweak that removes a corner-case bug

For a workshop demo, `rank_bm25`'s `BM25Okapi` is the classic textbook implementation and stays fast on pure Python.

---

## 6. Why hybrid (dense + sparse)

The two retriever families have **orthogonal failure modes**:

| Query type | Dense (FAISS) | Sparse (BM25) |
|---|---|---|
| "What is the paper about?" (conceptual) | ✅ | ⚠️ |
| "Explain self-attention" (paraphrasable) | ✅ | ⚠️ |
| "Who is Ashish Vaswani?" (proper noun) | ❌ | ✅ |
| "Error code E_ACCESS_DENIED" (rare token) | ❌ | ✅ |
| Synonyms / non-English paraphrase | ✅ | ❌ |
| Exact quote lookup | ⚠️ | ✅ |

Hybrid gives the union of both — you cover both the semantic and the lexical case. Almost every serious production RAG (Perplexity, Bing Chat, Cohere Rerank, etc.) is hybrid.

**Cost:** ~2× retrieval work (one BM25 scan + one FAISS lookup). Both are fast enough that this is negligible compared to the LLM call.

---

## 7. Why RRF (Reciprocal Rank Fusion)

You now have two ranked lists — how do you merge them into one?

### Option A: Weighted score sum

```
final_score = 0.5 × cosine + 0.5 × bm25_score
```

**Problem:** BM25 scores are unbounded positives (5–30 typical), cosine is [-1, 1]. You must normalize. Normalization method (min-max? z-score? global? per-query?) affects results. Tuning the weights requires a labeled dev set. **A lot of work for a demo.**

### Option B: Reciprocal Rank Fusion (Cormack et al., 2009)

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

With $k=60$ (canonical value).

**Why it works:**
- **Score-agnostic.** Uses only rank position, so no normalization needed.
- **Zero tuning.** No weights, no data required. Works out of the box.
- **Robust.** Diminishing returns per rank (rank 1 → 1/61, rank 10 → 1/70) so top hits dominate but a document ranked highly in *both* lists beats one ranked #1 in only one.
- **Empirically strong.** Frequently beats tuned weighted schemes on TREC benchmarks.

### Option C: Learned fusion (e.g. LambdaMART)

Trains a gradient-boosted model on labeled (query, doc) → relevance data. Best quality but needs hundreds of labeled queries. Overkill for a workshop.

**Why RRF for this notebook:** it just works, needs no training data, and generalizes to more than 2 rankers (you could add colbert, splade, etc. later without re-tuning).

**Why `k=60`:** the original paper's value, chosen so that a rank-1 doc in one list roughly ties with a rank-2 doc in both lists — a good balance between "trust the top hit" and "trust agreement".

---

## 8. Cross-encoder reranking vs bi-encoder

**Bi-encoder** (what FAISS uses):
```
query  → [Encoder] → q_vec
doc    → [Encoder] → d_vec
score  = cosine(q_vec, d_vec)
```
Query and doc are encoded **independently** → embeddings can be pre-computed & indexed → very fast at query time (millisecond).

**Cross-encoder** (what the reranker uses):
```
[CLS] query [SEP] doc [SEP]  →  [Encoder]  →  score
```
Query and doc are concatenated into **one input** so every layer can attend across them → much richer interaction signal → much higher accuracy.

**Trade-off:**

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Latency per pair | ~1 ms | ~50 ms (CPU) |
| Pre-computable | Yes (doc side) | No (needs query) |
| Accuracy (nDCG@10) | Baseline | +5–15 points |

**Why the two-stage funnel:**
```
Corpus (thousands)
   ↓ bi-encoder + BM25 (cheap, top-20 each)
Candidates (~40 unique)
   ↓ cross-encoder rerank (expensive)
Top-5 for LLM
```

You get the accuracy of the cross-encoder without paying its cost on the whole corpus. This is the standard "**retrieve-and-rerank**" pattern.

**Why `ms-marco-MiniLM-L-6-v2`:**
- Trained on MS-MARCO (~500k query-passage relevance judgments) — the industry-standard reranker training set
- Only 6 transformer layers → runs on CPU
- Community-standard choice; Cohere/Voyage rerankers are stronger but cost money

---

## 9. RAGAS metrics

RAGAS ("**R**etrieval **A**ugmented **G**eneration **AS**sessment") is a framework that uses an **LLM-as-judge** plus embeddings to grade a RAG pipeline **without hand-labeled data** for most metrics. It converts an intrinsically fuzzy "is this answer good?" question into deterministic sub-tasks the judge LLM can answer reliably.

### Why LLM-as-judge and not BLEU / ROUGE

Traditional NLP metrics (BLEU, ROUGE, METEOR) measure **n-gram overlap** with a reference. They fail on RAG:
- A perfectly correct paraphrase scores 0 if it uses different words.
- A hallucination that reuses reference words scores high.
- They can't score `contexts` (which have no reference).

LLM-as-judge instead asks the LLM to *reason* about the answer — extract claims, check entailment, decompose statements — which correlates far better with human judgment.

### Inputs each metric requires

| Metric | question | answer | contexts | ground_truth |
|---|:---:|:---:|:---:|:---:|
| Faithfulness | ✅ | ✅ | ✅ | — |
| Answer Relevancy | ✅ | ✅ | — | — |
| Context Precision | ✅ | — | ✅ | ✅ (or answer) |
| Context Recall | ✅ | — | ✅ | ✅ |
| Answer Correctness | ✅ | ✅ | — | ✅ |
| Answer Similarity | — | ✅ | — | ✅ |
| Context Entities Recall | — | — | ✅ | ✅ |
| Noise Sensitivity | ✅ | ✅ | ✅ | ✅ |

The 4 metrics used in the notebook are the first four. All are scored on **[0, 1]** where higher is better.

---

### 9.1 Faithfulness — "is the answer grounded?"

**Question it answers:** Does every factual claim in the generated answer follow from the retrieved context?

**Directly detects:** hallucination.

**Algorithm:**
1. Judge LLM splits the `answer` into individual atomic claims $c_1, c_2, \dots, c_n$.
2. For each claim $c_i$, judge asks: "Can this claim be inferred from the provided `contexts`?" → yes / no.
3. Score = fraction of claims verified.

$$\text{Faithfulness} = \frac{|\{c_i : c_i \text{ entailed by contexts}\}|}{|\{c_i\}|}$$

**Worked example:**
- Question: *"Who wrote the Transformer paper and where do they work?"*
- Answer: *"It was written by Vaswani et al. at Google Brain and Google Research."*
- Contexts: *"...Ashish Vaswani, Noam Shazeer... Google Brain..."* (no mention of Google Research)
- Claims: [`Vaswani wrote it` ✅, `at Google Brain` ✅, `at Google Research` ❌]
- Score = 2 / 3 = **0.67**

**How to improve it:** shrink & clean the context (fewer irrelevant chunks = less temptation to blend in outside knowledge), lower LLM temperature, use a stronger generator.

---

### 9.2 Answer Relevancy — "is the answer on topic?"

**Question it answers:** Does the answer actually address the user's question (regardless of correctness)?

**Directly detects:** off-topic responses, over-hedging, tangential rambling.

**Algorithm:**
1. Judge LLM generates $N$ (typically 3) synthetic questions that the given `answer` would plausibly be a response to.
2. Embed each synthetic question and the original `question`.
3. Score = mean cosine similarity between the original and the synthetic questions.

$$\text{AnswerRelevancy} = \frac{1}{N}\sum_{i=1}^{N} \cos\!\big(\text{emb}(q), \text{emb}(q_i^{\text{gen}})\big)$$

**Intuition:** if the answer is on-topic, "reverse-engineering" it back into a question should recover something very close to the original.

**Worked example:**
- Q: *"What datasets were used?"*
- Bad answer: *"The Transformer was fast to train."* → generated Qs: `"Was training fast?"`, `"How efficient is training?"` → low cosine to original → **low relevancy** even though the statement is true.
- Good answer: *"WMT 2014 English-German and English-French."* → generated Qs are all about datasets → **high relevancy**.

**Important:** relevancy does NOT check truth — only aboutness. A confidently wrong on-topic answer scores 1.0. That's why you need Faithfulness alongside it.

---

### 9.3 Context Precision — "is the retriever's signal-to-noise good?"

**Question it answers:** Of the chunks the retriever pulled, how many were actually relevant to the question — and are the relevant ones ranked at the top?

**Directly detects:** noisy retrieval, poor ranking.

**Algorithm:**
1. For each retrieved chunk at rank $k$, judge asks: "Is this chunk useful for producing the ground_truth?" → 1 or 0 → $v_k$.
2. Compute **precision@k** for each position, then take a **rank-weighted average** (Mean Average Precision-style).

$$\text{ContextPrecision@K} = \frac{\sum_{k=1}^{K} (\text{Precision@}k \cdot v_k)}{\sum_{k=1}^{K} v_k}$$

Where $\text{Precision@}k = \dfrac{\text{# relevant chunks in first } k}{k}$.

**Why rank-weighted:** a relevant chunk at rank 1 counts much more than the same chunk at rank 10. This mirrors real LLM behavior — top chunks get more "attention weight".

**Worked example (K=5, ✅=relevant):**
- Retrieved: [✅, ❌, ✅, ❌, ❌]
- Precision@1 = 1/1, @3 = 2/3
- Score = (1·1 + 2/3·1) / 2 = **0.83**
- Compare: [❌, ❌, ✅, ✅, ❌] → (2/3·1 + 2/4·1)/2 = **0.58** (same 2 relevant chunks but ranked worse)

**How to improve it:** better reranker, better hybrid fusion, prompt-side query rewriting.

---

### 9.4 Context Recall — "did the retriever find everything?"

**Question it answers:** Of the information in the `ground_truth`, how much is actually present in the retrieved `contexts`?

**Directly detects:** missing evidence — the "we never retrieved the right chunk" failure mode.

**Algorithm:**
1. Judge LLM splits the `ground_truth` answer into atomic statements $s_1, \dots, s_m$.
2. For each $s_j$, judge asks: "Is this statement supported by the retrieved contexts?"
3. Score = fraction supported.

$$\text{ContextRecall} = \frac{|\{s_j : s_j \text{ attributable to contexts}\}|}{|\{s_j\}|}$$

**Worked example:**
- Ground truth: *"Uses WMT En-De (4.5M pairs), WMT En-Fr (36M), and Penn Treebank."*
- Statements: [`En-De`, `4.5M`, `En-Fr`, `36M`, `Penn Treebank`]
- Contexts mention: `En-De 4.5M pairs` ✅, `En-Fr` ✅ (no size), `Penn Treebank` ✅
- Score = 4 / 5 = **0.80**

**How to improve it:** larger `fetch_k`, hybrid retrieval (this is exactly the metric BM25 fixes in the notebook), better chunking so the answer isn't fragmented.

**Note:** this is the one metric in the default set that *requires* `ground_truth`. Precision can use it too but can fall back to `answer` if ground truth is unavailable.

---

### 9.5 Other RAGAS metrics worth knowing

The notebook uses 4, but the framework ships several more. Add them when you need finer-grained diagnosis.

#### Answer Correctness
Combines **factual overlap** with `ground_truth` (via LLM claim comparison) + **semantic similarity** (via embeddings).

$$\text{AnswerCorrectness} = w_1 \cdot F_1(\text{TP, FP, FN of claims}) + w_2 \cdot \text{cosine}(\text{emb(answer), emb(gt)})$$

- **TP** = claims in both answer and ground truth
- **FP** = claims in answer only (hallucinations *relative to gold*)
- **FN** = claims in ground truth only (missed information)
- Default weights: $w_1 = 0.75$, $w_2 = 0.25$

**Use when:** you have a golden set and want a single "how right is the answer" score.

#### Answer Semantic Similarity
Pure cosine similarity between answer and ground_truth embeddings. Cheap, no LLM call.

**Use when:** you want a quick sanity check, or don't have an LLM judge budget.

**Limitation:** rewards fluent paraphrase even when facts are wrong — pair with Faithfulness.

#### Context Entities Recall
Extracts named entities (people, orgs, dates, places) from `ground_truth` and checks how many appear in `contexts`.

$$\text{ContextEntitiesRecall} = \frac{|\text{entities}(\text{gt}) \cap \text{entities}(\text{contexts})|}{|\text{entities}(\text{gt})|}$$

**Use when:** the domain is entity-heavy (medical, legal, financial). This is *exactly* the failure the notebook's "authors query" demonstrates — the entities `Vaswani`, `Shazeer`, etc. are missing from naive retrieval, and this metric would flag it directly.

#### Noise Sensitivity
Injects irrelevant chunks into the context and measures how much the answer degrades.

**Use when:** you want to stress-test the generator's robustness — a good LLM should ignore garbage chunks rather than hallucinate from them.

#### Context Relevancy *(deprecated / replaced by Context Precision)*
Older metric — asked "what fraction of sentences in the context are relevant?". Less useful than Context Precision because it ignores ranking.

#### Aspect Critique (custom metrics)
RAGAS lets you define your own binary judge: `Critique(name="tone_professional", definition="Is the answer written in a professional tone?")`. Judge LLM returns yes/no per row.

**Use for:** brand voice, safety, PII disclosure, refusal quality — anything the built-in metrics don't cover.

---

### 9.6 The 4 default metrics — the 2×2 mental model

The 4 metrics used in the notebook aren't arbitrary — they form a full 2×2 diagnostic matrix:

| | **Retriever quality** | **Generator quality** |
|---|---|---|
| **"Is the good stuff present?"** (recall-like) | **Context Recall** | **Answer Relevancy** |
| **"Is the bad stuff absent?"** (precision-like) | **Context Precision** | **Faithfulness** |

This means you can **localize any regression**:

| Symptom in scores | Likely root cause | Fix |
|---|---|---|
| Recall ↓, Precision fine | Missing chunks | Hybrid retrieval, larger `fetch_k`, better chunking |
| Precision ↓, Recall fine | Too much noise in top-k | Rerank harder, smaller `top_k` |
| Faithfulness ↓, Precision fine | LLM hallucinating despite good context | Lower temperature, better/bigger generator, stricter prompt |
| Relevancy ↓, Faithfulness fine | LLM answering a different question | Query rewrite, prompt clarification |
| All 4 low | Pipeline is broken (bad embeddings, wrong PDF, etc.) | Diagnose earlier stages |

This is why the notebook's Advanced pipeline shows lifts across **all four** metrics — hybrid retrieval fixes recall/precision, and cleaner context indirectly boosts faithfulness/relevancy.

---

### 9.7 Interpreting the numbers

Rough thresholds observed across production RAG systems:

| Score | Verdict |
|---|---|
| **> 0.85** | Production-ready |
| 0.70 – 0.85 | Usable; iterate on the weakest metric |
| 0.50 – 0.70 | Prototype-quality; needs work |
| **< 0.50** | Broken; check chunking / embeddings / prompts before tuning |

The notebook draws a `y=0.7` line on the bar chart for exactly this reason.

**Caveats:**
- LLM-judged scores have **variance**. Run 2–3 times, use the mean. Fixing `temperature=0` helps but doesn't eliminate it.
- A stronger judge (`llama-3.3-70b` or GPT-4) gives more reliable but harsher scores. Don't compare numbers from different judges — only *lifts* within the same judge.
- Small test sets (4 questions in this notebook) have high variance per metric. In production, aim for 50–200 questions.

---

### 9.8 Practical judge & embedding choices

The notebook uses **Groq Llama-3.1-8B** for judge + **MiniLM** for embeddings — both free, both fast, both good enough for a workshop.

| Component | Notebook default | Production upgrade | When to upgrade |
|---|---|---|---|
| Judge LLM | Llama-3.1-8B (Groq) | Llama-3.3-70B or GPT-4o | Stricter grading; when 8B judge scores plateau |
| Judge embeddings | MiniLM (local) | OpenAI `text-embedding-3-large` | When Answer Relevancy scores look noisy |
| Test set size | 4 questions | 50–200 curated | Before any prod launch |
| Reruns per metric | 1 | 3 (average) | To reduce judge variance |

**Cost note:** Faithfulness and Context Recall are the heaviest — they decompose text into many claims/statements and call the judge once per claim. A 4-question × 4-metric eval can burn 20–50k tokens. Free Groq quota (~100k TPD on 70B) handles a workshop but plans need budgeting for large evals.

---

## 10. Why Groq + Llama

**Why Groq API:**
- **Free tier** with real throughput (no credit card)
- **LPU (Language Processing Unit) hardware** — 500+ tokens/sec output, sub-second first-token
- OpenAI-compatible SDK — swap `Groq(...)` for `OpenAI(...)` and it's the same code

**Why Llama-3.1-8B for generation:**
- Fast, cheap on quota
- Big enough to synthesize 5-chunk context into a coherent answer
- Instruction-tuned so it obeys the "cite pages" and "say Not found" rules

**Why Llama-3.3-70B is *optional* for the judge:**
- RAGAS asks the judge harder questions (claim decomposition, entailment)
- Bigger model gives more reliable scores
- But free quota (~100k tokens/day) can burn out mid-eval → notebook defaults to the 8B judge and only recommends 70B if you want stricter grading

**Why not OpenAI GPT-4o:**
- Requires a paid API key — bad for workshop demos
- Slower than Groq
- Better judgment quality, but the *comparison* between naive and advanced pipelines shows the same trend either way — so cost isn't justified for teaching

**Why not local Llama via Ollama:**
- Great option for privacy/offline demos
- Adds a heavy install step (Ollama binary + model download of several GB)
- Slower on laptops without a GPU

---

## Cheat sheet — the whole "why" in one table

| Choice made | Alternative | Why the notebook picks this |
|---|---|---|
| `RecursiveCharacterTextSplitter` | Fixed-size, semantic | Preserves paragraphs, no extra deps |
| Chunk 1000 / overlap 200 | 512/50, 3000/300 | Balances recall vs precision on prose |
| `all-MiniLM-L6-v2` | BGE, OpenAI | Free, fast, CPU-friendly, small quality gap |
| Cosine similarity | Euclidean, dot | Magnitude-invariant; matches how embedder was trained |
| FAISS | Chroma, Pinecone | Zero infra, fastest single-machine ANN |
| BM25 | TF-IDF, SPLADE | Fixes TF-IDF's saturation + length bias; no GPU needed |
| Hybrid (dense + sparse) | Dense-only | Covers proper nouns & rare tokens dense misses |
| RRF | Weighted score sum | No normalization, no tuning, robust |
| Cross-encoder rerank | Bi-encoder only | Big precision boost, applied only to top-N |
| `ms-marco-MiniLM-L-6-v2` | Cohere Rerank, BGE reranker | Free, CPU-runnable, standard baseline |
| RAGAS 4 metrics | BLEU, ROUGE | Covers both retriever & generator; LLM-graded not lexical |
| Groq + Llama-3.1-8B | OpenAI, Ollama | Free, fast, no local GPU needed |
