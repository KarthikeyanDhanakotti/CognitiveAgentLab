# 📖 RAG Workshop — Presenter Narration Guide

> Companion to **`RAG_AdvancedRAG (2).ipynb`** — cell-by-cell talking points, teaching moments, and a decoder for the naive-RAG results.
>
> Use this as your live workshop script. Each cell has: 🎯 what it does · 🎤 what to say · 💡 the teaching point (where useful).

---

## 🟢 Part 0 — Setup

### Cell 1 · Nuclear fix (install pinned versions)
- **🎯 Does:** Uninstalls Colab's broken preinstalled `ragas 0.1.x`, installs a matched `ragas 0.2` + `langchain 0.3` stack.
- **🎤 Say:**
  > *"Google Colab ships with an old RAGAS version that has a broken import. We nuke it and install a known-good matching set. Nothing about RAG yet — pure environment hygiene."*
- **💡 Teaching moment:** LLM tooling moves fast. Pinning matching versions is production hygiene, not paranoia.

### Cell 2 · Title markdown
- **🎯 Does:** Sets expectations — a 3-part demo (Naive → Advanced → RAGAS).
- **🎤 Say:**
  > *"We'll build Naive RAG in 6 cells, upgrade it to Advanced in 5 more, then measure both with RAGAS. Total: ~30 minutes of code, 90 minutes of learning."*

### Cell 3 · Imports
- **🎯 Does:** Loads LangChain, FAISS, BM25, Cross-Encoder, and Groq clients.
- **🎤 Say:**
  > *"All imports upfront. Notice we're using free / open tools everywhere — no OpenAI key required. Groq for the LLM, HuggingFace for embeddings, FAISS for vectors."*

### Cell 4 · `GROQ_API_KEY` loader
- **🎯 Does:** Auto-detects environment (env var → Colab Secrets → `.env` → prompt) and loads the key.
- **🎤 Say:**
  > *"This one function makes the notebook portable. Same code works on Colab, VS Code, local, or Docker — it just finds the key wherever you keep it."*

### Cell 5 · LLM client wrapper
- **🎯 Does:** Wraps the Groq client into a simple `call_llm(prompt)` using `llama-3.1-8b-instant`, `temperature=0.2`.
- **🎤 Say:**
  > *"One function, one purpose. Low temperature = deterministic, factual. This is our generation LLM — separate from the eval judge we'll set up in Part 3."*

---

## 🔵 Part 1 — Naive RAG

### Cell 6 · Load PDF
- **🎯 Does:** Downloads *Attention is All You Need* to a temp dir (only if missing), loads it via PyMuPDF.
- **🎤 Say:**
  > *"15 pages, ~7,500 words. The famous Transformer paper — a good stress test because it has code-like notation, math, author names, and dense prose."*

### Cell 7 · Chunk (Naive version)
- **🎯 Does:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` → adds `chunk_id` + `page` metadata.
- **🎤 Say:**
  > *"1,000-token chunks, 20 % overlap. The overlap prevents information loss when a fact spans a chunk boundary. Each chunk carries its source page — that's how we cite pages later."*
- **💡 Key teaching:** chunking is *the* #1 lever in RAG quality.

### Cell 8 · Chunking trade-off demo
- **🎯 Does:** Splits the same paragraph 3 different ways (200 / 1000 / 3000 tokens) and prints the counts.
- **🎤 Say:**
  > *"Watch what happens to the same page with three different chunk sizes. Too small → 15 tiny chunks that fragment ideas. Too big → one chunk that dilutes signal. Sweet spot is 500–1000 with 15–20 % overlap for prose."*

### Cell 9 · Embed + FAISS
- **🎯 Does:** Encodes all chunks with `all-MiniLM-L6-v2` (384-dim), stores in FAISS.
- **🎤 Say:**
  > *"Each chunk becomes a 384-dimensional vector — a point in meaning-space. FAISS lets us find the nearest neighbours in milliseconds. First run downloads the model (~90 MB), subsequent runs are instant."*

### Cell 10 · `naive_rag()` function
- **🎯 Does:** Retrieves top-5 chunks by cosine similarity → stuffs them into a prompt → calls the LLM → returns answer + citations.
- **🎤 Say:**
  > *"This is a complete RAG system in 8 lines of logic. Retrieve → Format → Prompt → Answer. That's it. Everything you'll see later is optimisation on top of this."*

### Cell 11 · Naive RAG demo — actual observed output

```
Q1  Main idea?          → ❌ Not found in document        (pages 1, 12-15)
Q2  Transformer arch?   → ✅ Correct  [Page 3]            (pages 1, 2, 3, 5, 8)
Q3  Datasets?           → ❌ Not found in document        (pages 4, 7, 9)
Q4  Authors?            → ❌ Not found in document        (pages 10, 11, 12)
```

- **🎤 Say:**
  > *"Look at this — only 1 out of 4 correct. The one that worked ('architecture') is a **conceptual** question — the word 'architecture' has strong semantic neighbours in the paper. The three that failed all need something more specific:*
  > - *'Main idea' → the abstract, on page 1 — but embeddings pull the model to later pages*
  > - *'Datasets' → specific words like 'WMT 2014' don't embed well*
  > - *'Authors' → proper nouns ('Vaswani', 'Shazeer') don't embed AT ALL*
  >
  > *This is the naive-RAG failure mode. And it's NOT the LLM's fault — the retriever handed it the wrong pages, and the LLM correctly said 'Not found' rather than hallucinate. Which brings us to the diagnostic…"*
- **💡 Pause here** — let the audience see 3 red ❌s on the screen. Peak teaching moment.

### Cell 12 · Markdown "Naive hit a wall"
- **🎤 Say:**
  > *"Let's PROVE why the authors query failed by looking at what FAISS actually retrieved."*

### Cell 13 · Diagnostic cell
- **🎯 Does:** Runs the failing "Who are the authors?" query once more, prints the top-5 chunks FAISS returned, and checks whether any of them actually contains 'Vaswani' or 'Shazeer'.
- **🎤 Say:**
  > *"This is diagnostic thinking. Instead of blaming the LLM, we ask: what did the RETRIEVER actually see?*
  > *The top-5 will be about attention math, positional encodings, ablations — anything BUT the author list on page 1.*
  > *The check `authors_in_top5` will print ❌ NO — that's the smoking gun. The authors chunk never even made it into the context window."*
- **💡 Takeaway:** *"Retrieval failed → generation fails. Fix the retriever, not the prompt."*

---

## 🟠 Part 2 — Advanced RAG

### Cell 14 · Markdown intro
- **🎤 Say:**
  > *"Three additions: BM25 for exact tokens, RRF to combine results, Cross-Encoder to re-rank the winners. Same skeleton — smarter organs."*

### Cell 15 · Advanced RAG marker + reload libs
- **🎤 Say:**
  > *"Sanity-reload — makes this section runnable standalone if someone jumps in."*

### Cell 16 · LLM client (duplicate)
- **🎤 Say:**
  > *"Same LLM as before — the change is in retrieval, not generation. Same model, same temperature — so any improvement we measure comes purely from the retriever."*

### Cell 17 · Load PDF (reuse `PDF_PATH`)
- **🎤 Say:**
  > *"Skip — same PDF."*

### Cell 18 · Chunk (Advanced version)
- **🎯 Does:** Same `chunk_size=1000`, `chunk_overlap=200`, same metadata additions.
- **🎤 Say:**
  > *"**Deliberately identical** to naive chunking. Same 15 pages, same 35 chunks, same `chunk_id` and `page` metadata. If we changed the chunker AND the retriever, we couldn't isolate what caused the improvement. This is a scientific control — only ONE variable moves."*
- **💡 Difference from the naive chunk cell:** *there is none — that's the point.* The improvement in Part 2 comes purely from the retriever changes in cell 19.

### Cell 19 · Build 3 indexes (Dense + Sparse + Reranker)
- **🎤 Say:**
  > *"Three tools, three jobs:*
  > 1. ***FAISS** — semantic (dense). Same as naive.*
  > 2. ***BM25** — keyword (sparse). Tokenises text, ranks by term frequency. Catches 'Vaswani' where FAISS can't.*
  > 3. ***Cross-Encoder** — a slower, more accurate model that scores query+doc PAIRS. Used for reranking, not retrieval."*

### Cell 20 · RRF fusion
- **🎯 Does:** Merges two ranked lists using `score(d) = Σ 1 / (k + rank)` with `k = 60`.
- **🎤 Say:**
  > *"How do you combine FAISS scores (0–1) with BM25 scores (0–50+)? You DON'T. You throw away the values and combine the RANKS. RRF's magic: no normalisation, no tuning, just position-based fusion. If a doc shows up near the top of BOTH lists, it wins big."*

### Cell 21 · `advanced_rag()` function
- **🎤 Say:**
  > *"Five sub-steps: dense retrieve → sparse retrieve → RRF merge → cross-encoder rerank → LLM. Notice we over-fetch 20 candidates from each, fuse them, then let the cross-encoder pick the top 5. Recall first, then precision."*

### Cell 22 · Advanced RAG demo
- **🎤 Say:**
  > *"Same 4 questions. Watch the failures disappear — especially the authors query."*
- **Expected:** 4 / 4 correct with page citations.

### Cell 23 · Markdown "The Reveal"
- **🎤 Say:**
  > *"Now let's SEE why hybrid works."*

### Cell 24 · Dense vs Sparse vs Hybrid comparison
- **🎯 Does:** Runs the "authors" query through **each retriever separately** and prints their top-5.
- **🎤 Say:**
  > *"This is the money cell of the whole workshop. Look at:*
  > - *🔵 **DENSE (FAISS) top-5** → same misses as before, no author chunk*
  > - *🟠 **SPARSE (BM25) top-5** → boom, page 1 with the author list at rank 1*
  > - *🟢 **ADVANCED (Hybrid + Rerank) top-5** → union of the two, with the right chunk lifted to the top by the cross-encoder*
  >
  > *This one output explains everything. Dense retrieval alone MISSED the answer. BM25 alone would over-fit on keyword queries. Hybrid = best of both."*
- **💡 Pause here** — pedagogical peak of Part 2.

---

## 🟣 Part 3 — RAGAS Evaluation

### Cell 25 · Markdown intro
- **🎤 Say:**
  > *"You've SEEN the difference. Now let's MEASURE it. Four metrics, industry-standard."*

### Cell 26 · Golden test set
- **🎯 Does:** Defines 4 dicts with `question` + `ground_truth`.
- **🎤 Say:**
  > *"The single most important artefact in ANY eval system. RAGAS needs a 'right answer' to score against. In production you'd have 50–500 of these across categories — fact, reasoning, metadata, edge cases."*

### Cell 27 · Run each pipeline
- **🎯 Does:** Iterates the 4 questions through both `naive_rag` and `advanced_rag`, collects `question / answer / contexts / ground_truth` rows.
- **🎤 Say:**
  > *"We need the retrieved CONTEXTS too — RAGAS needs them to score Context Precision and Recall. Just answers aren't enough."*

### Cell 28 · Old RAGAS config cell (JUDGE_MODEL setup — optional)
- **🎤 Say:**
  > *"Kept for reference — you can DELETE this cell. The next cell (full evaluation) is self-contained and defines the same things inline."*
- **💡 If you keep it:** *"This shows the judge configuration in isolation — same code runs inside the next cell."*

### Cell 29 · Full RAGAS evaluation (self-contained)
- **🎯 Does:** Imports everything → configures Groq 8B as judge + MiniLM as judge embeddings → runs `evaluate()` on both pipelines → returns a dataframe with 4 metric scores per row.
- **🎤 Say:**
  > *"This is where the magic happens. `evaluate()` sends each question to Llama-3.1-8B — the JUDGE model — with a rubric like 'is this answer grounded in this context?' The judge scores 0–1 for each of 4 metrics on each of 4 questions. Total: ~32 LLM calls, ~90 seconds on free tier.*
  >
  > *Two LLMs, two roles:*
  > - *8B (temp 0.2) = the **generator** that answers user questions*
  > - *8B (temp 0)   = the **judge** that scores those answers*
  >
  > *In production you'd use a stronger judge (70B or GPT-4) — for the workshop 8B is safe on free tier."*
- **💡 While it runs:** do Q&A — takes 1–2 minutes.

### Cell 30 · Summary + lift table
- **🎯 Does:** Groups `df_all` by pipeline, means the 4 metrics, computes absolute + % lift.
- **🎤 Say:**
  > *"Now we see the story in numbers. Expect something like:*
  > - *Naive average ~0.30–0.40 (very poor — 3 of 4 questions got 'Not found')*
  > - *Advanced average ~0.80–0.90 (production-ready)*
  > - *Lift of +150–200 % across metrics*
  >
  > *The bigger the lift, the more valuable the reranker + BM25 combo is for YOUR data."*

### Cell 31 · Bar chart
- **🎯 Does:** Plots 4 metrics × 2 pipelines with a dashed 0.7 "production threshold" line.
- **🎤 Say:**
  > *"Naive bars mostly BELOW the dashed line (0.7 threshold) — not shippable. Advanced bars mostly ABOVE — production-ready. The gap between the two bars per metric is what your reranker earned you.*
  >
  > *Pro tip: this chart is what you show your product manager. It converts 'we made RAG better' into a quantified before/after."*

---

## 🔍 Optional drill-down cells

### Cell 32 · Print `df_all` columns
- **🎯 Does:** Debug utility — shows the exact column names RAGAS produced (they've renamed across versions).
- **🎤 Say:**
  > *"Housekeeping — RAGAS renamed some columns in v0.2. This confirms we're picking the right ones for the pivot table below."*

### Cell 33 · Per-question comparison pivot
- **🎯 Does:** Pivots `df_all` → rows = questions, columns = pipelines, values = each metric.
- **🎤 Say:**
  > *"This is your DIAGNOSIS table. If a specific question drops on 'Context Recall', it means retrieval missed something for THAT question. You'd add that question to the golden set, tune the retriever, and re-eval."*

### Cell 34 · Sort by pipeline (all raw scores)
- **🎯 Does:** Prints all raw rows — every question × pipeline × metric.
- **🎤 Say:**
  > *"For the auditor / lead who wants to see every single number. Skip in the workshop."*

### Cell 35 · Recap markdown
- **🎤 Say:**
  > *"Three parts, three lessons: (1) naive misses tokens, (2) hybrid + rerank fixes it, (3) RAGAS proves it."*

---

## 🎓 Optional closer (if there's time)

> *"So what do we have? A working RAG pipeline, a way to measure it, and evidence — not vibes — that our advanced version is better. That's the difference between a demo and a product. In Session 2 we'll take this further: G-Eval for custom rubrics, safety probes, CI gates, and production monitoring."*

---

## 📋 Quick reference — the actual naive results, decoded

| Question | Naive result | Root cause | Fix in Part 2 |
|---|---|---|---|
| **Main idea?** | ❌ Not found · pages 1, 12–15 | FAISS pulled ablation pages instead of the abstract | Cross-encoder promotes page 1 abstract |
| **Transformer architecture?** | ✅ Worked · page 3 | 'Architecture' has strong semantic signal | Same result |
| **Datasets?** | ❌ Not found · pages 4, 7, 9 | 'WMT 2014' proper nouns miss semantic search | BM25 catches 'WMT' literally |
| **Authors?** | ❌ Not found · pages 10–12 | Author names have no meaning-neighbours | BM25 catches 'Vaswani' at rank 1 |

> **⚠️ Notice:** even the "Pages used" for failed queries are all *plausible-looking* pages — not obviously wrong. That's what makes silent failure dangerous: the pipeline doesn't warn you, it just quietly says "Not found." *That's the whole reason you need RAGAS.*

---

## 🔑 One-line summaries you can put on slides

| Concept | One-liner |
|---|---|
| **Naive RAG** | Retrieve top-k similar chunks · stuff into prompt · answer |
| **Failure mode** | Embeddings encode meaning, not exact tokens · proper nouns / IDs miss |
| **BM25** | Old-school keyword ranking · catches literal tokens FAISS can't |
| **RRF** | Combine ranked lists by POSITION, not by score — no normalisation needed |
| **Cross-Encoder** | Slow but precise reranker · promotes best chunk to rank 1 |
| **RAGAS** | 4 LLM-judged metrics that turn "better" into a scored, trending number |
| **Golden set** | The versioned question + ground-truth pairs your regression tests run against |
| **Judge model** | The LLM that GRADES your generator's answers · use 8B for live, 70B for CI |

---

## 🛟 Troubleshooting cheat-sheet (for live workshop hiccups)

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: langchain_community.chat_models.vertexai` | Run **Cell 1 (nuclear fix)** → **Runtime → Restart runtime** → re-run |
| `NameError: METRICS is not defined` | Run **Cell 29** (self-contained eval); it now defines everything inline |
| `RateLimitError 429 ... TPD` | Free Groq daily cap — switch `JUDGE_MODEL` to `llama-3.1-8b-instant` |
| `GROQ_API_KEY not found` | Left sidebar 🔑 → add secret `GROQ_API_KEY` → toggle Notebook access ON |
| First run is slow | First-time HuggingFace model download (~90 MB). Subsequent runs are instant |
| Bar chart doesn't show up | Add `%matplotlib inline` at top of cell 31 |

---

## 📖 Further reading

- [RAGAS docs](https://docs.ragas.io)
- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Reciprocal Rank Fusion (Cormack 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the workshop's demo PDF
- [Groq console (free API keys)](https://console.groq.com/keys)

---

*Generated for the "RAG Fundamentals & Production-Grade RAG" workshop · companion to `RAG_AdvancedRAG (2).ipynb`.*
