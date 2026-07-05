# Naive RAG — End-to-End Demo (PDF → FAISS → LLM)

A hands-on notebook that walks through a **basic Retrieval-Augmented Generation (RAG)** pipeline: load a PDF, chunk it, embed the chunks into a FAISS vector store, retrieve the top-k nearest chunks for a question, and hand them to an LLM to answer.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.1--8B-purple) ![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-orange) ![Vector%20Store](https://img.shields.io/badge/Vector%20Store-FAISS-brightgreen) ![Cost](https://img.shields.io/badge/Cost-Free-brightgreen)

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) · Author: **Karthikeyan Dhanakotti**

> Looking for the more sophisticated variant (Hybrid retrieval + RRF + Cross-Encoder reranking + RAGAS evaluation)? See the [AdvancedRAG](../AdvancedRAG/) project.

---

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Groq API key  →  https://console.groq.com/keys
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

# 3. Open the notebook and run top-to-bottom
jupyter notebook RAG.ipynb
```

> Runs on **Google Colab (free tier)** or any local Python 3.9+ environment. The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) into your local Hugging Face cache. Subsequent runs are instant.

---

## Repo Structure

```
CognitiveAgentLab/
└── projects/
    └── RAG/
        ├── README.md          ← this file
        ├── RAG.ipynb          ← the workshop notebook
        └── requirements.txt   ← pinned dependencies
```

---

## Pipeline

```
PDF  →  Load (PyMuPDF)  →  Chunk (Recursive splitter, 1000/200)
     →  Embed (MiniLM-L6-v2)  →  Store (FAISS)
     →  Similarity search (top-k)  →  LLM (Groq Llama-3.1-8B)  →  Answer with [Page X] citations
```

The demo PDF is *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*; the notebook downloads it automatically to a temp folder on first run.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or newer |
| **RAM** | 4 GB+ |
| **Disk** | ~500 MB (Hugging Face cache) |
| **GROQ_API_KEY** | Free — get one at [console.groq.com/keys](https://console.groq.com/keys) |
| **Internet** | Needed only on first run to download the model + sample PDF |

### Set `GROQ_API_KEY`

| Environment | How |
|---|---|
| Google Colab | Left sidebar 🔑 → Add secret named `GROQ_API_KEY` |
| Jupyter / JupyterLab / VS Code | Create a `.env` with `GROQ_API_KEY=gsk_...` |
| Windows terminal | `setx GROQ_API_KEY "gsk_..."` (restart shell) |
| Linux / Mac terminal | `export GROQ_API_KEY=gsk_...` |
| One-time | The setup cell will prompt securely if none of the above is set |

---

## Tech Stack

- **LLM:** Groq `llama-3.1-8b-instant` (via the `groq` Python client)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim, ~90 MB)
- **Vector store:** FAISS (`faiss-cpu`)
- **PDF loader:** `pymupdf` via `langchain_community.document_loaders.PyMuPDFLoader`
- **Chunker:** `langchain_text_splitters.RecursiveCharacterTextSplitter`

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: google.colab` | You are not on Colab — the portable loader handles this; ignore. |
| `ModuleNotFoundError: langchain_huggingface` | `pip install langchain-huggingface` |
| `GROQ_API_KEY` not found | Set it via one of the methods above. |
| First run is slow | One-time HF model download + FAISS index build. |
| `RateLimitError 429` | Free Groq tier daily cap. Wait ~15 min. |

---

## License

See the repository root for license details.
