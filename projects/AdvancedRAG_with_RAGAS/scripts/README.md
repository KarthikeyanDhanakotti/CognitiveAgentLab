# RAG Scripts — standalone Python version of the notebook

Runnable `.py` version of `RAG_AdvancedRAG (2).ipynb`. Same pipelines, same evaluation, but importable modules and CLI-runnable.

## Files

| File | What it does |
|---|---|
| [common.py](common.py) | Shared utilities: API-key loader, LLM client, PDF download, chunker, FAISS builder, BM25 preprocess, prompt template |
| [naive_rag.py](naive_rag.py) | `NaiveRAG` class — PDF → chunk → embed → FAISS → LLM. CLI demo runs the 4 sample questions |
| [advanced_rag.py](advanced_rag.py) | `AdvancedRAG` class — Hybrid (FAISS + BM25) → RRF → Cross-Encoder rerank → LLM. Plus reusable `rrf()` |
| [evaluate.py](evaluate.py) | End-to-end RAGAS evaluation: runs both pipelines, grades with 4 metrics, prints lift, saves CSV + PNG |
| [requirements.txt](requirements.txt) | Pinned dependency set (matches the notebook) |

## Setup

```powershell
# 1. Create & activate a venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. Set your Groq API key (any ONE of these)
$env:GROQ_API_KEY = "gsk_..."          # current shell only
setx GROQ_API_KEY "gsk_..."            # persistent (restart shell)
#   OR drop it in a .env file next to the scripts
```

Free key: <https://console.groq.com/keys>

## Run

```powershell
# Sanity check — loads key, downloads PDF, builds FAISS, pings LLM
python common.py

# Run Naive RAG on 4 demo questions
python naive_rag.py

# Run Advanced RAG on the same 4 questions (authors query now succeeds)
python advanced_rag.py

# Full RAGAS eval → saves results/ragas_all_scores.csv, ragas_summary.csv, ragas_comparison.png
python evaluate.py

# Tougher grading with the 70B judge (may hit free-tier daily cap)
python evaluate.py --judge llama-3.3-70b-versatile

# Headless / CI mode — no chart
python evaluate.py --no-chart --outdir ci_results/
```

## Import from your own code

```python
from naive_rag import NaiveRAG
from advanced_rag import AdvancedRAG

rag = AdvancedRAG.build()
result = rag.query("What is the Transformer architecture?")
print(result["answer"])
print("Pages cited:", result["pages_used"])
```

## Layout at a glance

```
┌─────────────┐
│  common.py  │  key loader · LLM client · PDF · chunker · FAISS · BM25 preproc · prompt
└──────┬──────┘
       │ imported by
       ▼
┌──────────────┐        ┌────────────────┐
│ naive_rag.py │        │ advanced_rag.py│
└──────┬───────┘        └────────┬───────┘
       │                         │
       └────────────┬────────────┘
                    │ imported by
                    ▼
             ┌────────────┐
             │ evaluate.py│  RAGAS: Faithfulness · Answer Relevancy
             │            │         Context Precision · Context Recall
             └────────────┘
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'ChatVertexAI'` | Old RAGAS pinned to old langchain. Run `pip install -U -r requirements.txt` |
| `RateLimitError 429 ... TPD` | Groq free-tier daily cap. Switch judge to `llama-3.1-8b-instant` or wait 15 min |
| First run downloads ~90 MB | HuggingFace MiniLM model cache — one-off; subsequent runs are fast |
| `GROQ_API_KEY` not found | Set env var, use `.env`, or the script will prompt interactively |
| PyMuPDF install fails on ARM64 Windows | Use Python 3.11 x64, or install prebuilt wheel from PyPI |

## What each pipeline demonstrates

- **Naive RAG** — the "author query" fails because dense embeddings don't model proper nouns well
- **Advanced RAG** — BM25 catches the exact tokens, RRF merges the two rankings without score normalization, cross-encoder lifts the correct chunk to rank 1
- **RAGAS** — measures the improvement across 4 axes so the win is provable, not just felt

Full theory: see [../RAG_Algorithms_DeepDive.md](../RAG_Algorithms_DeepDive.md) and [../RAG_Notebook_CellByCell.md](../RAG_Notebook_CellByCell.md).
