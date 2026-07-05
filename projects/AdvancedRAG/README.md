# Advanced RAG — Hybrid Retrieval + RRF + Cross-Encoder Reranking (with RAGAS Evaluation)

A hands-on notebook that upgrades a **Naive RAG baseline** into an **Advanced RAG** pipeline and measures the lift with [RAGAS](https://docs.ragas.io) on four quality metrics.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.1--8B%20%2B%203.3--70B-purple) ![Retrieval](https://img.shields.io/badge/Retrieval-FAISS%20%2B%20BM25%20%2B%20RRF-orange) ![Reranker](https://img.shields.io/badge/Reranker-CrossEncoder%20MiniLM-yellow) ![Eval](https://img.shields.io/badge/Eval-RAGAS-blueviolet) ![Cost](https://img.shields.io/badge/Cost-Free-brightgreen)

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) · Author: **Karthikeyan Dhanakotti**

> Just want the plain baseline? See the [RAG](../RAG/) project.

---

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Groq API key  →  https://console.groq.com/keys
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

# 3. Open the notebook and run top-to-bottom
jupyter notebook AdvancedRAG.ipynb
```

> Runs on **Google Colab (free tier)** or any local Python 3.9+ environment. First run downloads two Hugging Face models (~180 MB total) into your local cache. Subsequent runs are instant.

---

## Repo Structure

```
CognitiveAgentLab/
└── projects/
    └── AdvancedRAG/
        ├── README.md              ← this file
        ├── AdvancedRAG.ipynb      ← the workshop notebook (naive + advanced + eval)
        └── requirements.txt       ← pinned dependencies
```

---

## What you'll build

| Part | What it does |
|---|---|
| **1. Naive RAG** | Baseline — PDF → chunk → embed → FAISS → LLM |
| **2. Advanced RAG** | Hybrid retrieval (FAISS + BM25) → **RRF fusion** → **Cross-Encoder reranker** → LLM |
| **3. RAGAS evaluation** | Compares both pipelines across a 4-question golden set on 4 metrics |

Metrics evaluated: **faithfulness · answer_relevancy · context_precision · context_recall**.

The demo PDF is *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*; the notebook downloads it automatically on first run.

---

## Advanced pipeline flow

```
PDF  →  Chunk  →  ┌─ FAISS (dense/semantic, MiniLM-L6-v2)
                  └─ BM25  (sparse/keyword)
                           │
                           ▼
                    RRF fusion (Reciprocal Rank Fusion, k=60)
                           │
                           ▼
             Cross-Encoder rerank (ms-marco-MiniLM-L-6-v2)
                           │
                           ▼
                    Top-k context  →  LLM  →  Answer + [Page X] citations
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or newer |
| **RAM** | 4 GB+ (embeddings + FAISS + reranker) |
| **Disk** | ~500 MB (Hugging Face cache) |
| **GROQ_API_KEY** | Free at [console.groq.com/keys](https://console.groq.com/keys) |
| **Internet** | Needed only on first run |

### Set `GROQ_API_KEY`

| Environment | How |
|---|---|
| Google Colab | Left sidebar 🔑 → Add secret named `GROQ_API_KEY` |
| Jupyter / JupyterLab / VS Code | Create a `.env` with `GROQ_API_KEY=gsk_...` |
| Windows terminal | `setx GROQ_API_KEY "gsk_..."` (restart shell) |
| Linux / Mac terminal | `export GROQ_API_KEY=gsk_...` |
| One-time | The setup cell will prompt securely if none of the above is set |

> The RAGAS evaluation cell uses `llama-3.3-70b-versatile` as the judge and `all-MiniLM-L6-v2` as the judge embeddings. Both are free — **no OpenAI key required**.

---

## Tech Stack

- **LLMs:** Groq `llama-3.1-8b-instant` (answers) + `llama-3.3-70b-versatile` (RAGAS judge)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim, ~90 MB)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, ~90 MB)
- **Dense index:** FAISS (`faiss-cpu`)
- **Sparse index:** BM25 (`rank-bm25`)
- **Evaluation:** [RAGAS](https://docs.ragas.io) + `datasets`
- **Charting:** `matplotlib` + `pandas`

---

## Expected result

On the 4-question sample the Advanced pipeline typically lifts RAGAS scores by **+50–150%** on *faithfulness* and *context_precision* vs. the Naive baseline. The notebook prints an absolute-lift table and a bar chart with a `0.7` production-threshold line.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: google.colab` | You are not on Colab — the portable loader handles this; ignore. |
| `ModuleNotFoundError: langchain_huggingface` | `pip install langchain-huggingface` |
| `RateLimitError 429 ... TPD` | Free Groq daily cap (~100K tokens on 70B). Wait ~15 min OR change the judge model to `llama-3.1-8b-instant` in the RAGAS cell. |
| First run is slow | One-time download of 2 HF models (~180 MB) + FAISS index build. |
| `GROQ_API_KEY` not found | Set it via one of the methods above. |

---

## Further reading

- [RAGAS docs](https://docs.ragas.io)
- [LangChain RAG guide](https://python.langchain.com/docs/tutorials/rag/)
- [Reciprocal Rank Fusion paper (Cormack et al.)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [*Attention Is All You Need* — the demo PDF](https://arxiv.org/abs/1706.03762)

---

## License

See the repository root for license details.
