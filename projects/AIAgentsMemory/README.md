# AI Agents Memory — Working, Episodic, Semantic & Long-Term Memory

A hands-on workshop notebook that walks through **five phases of memory for LLM agents**, from a simple chat-history-only agent to a full long-term memory stack backed by [Mem0](https://mem0.ai) with local Hugging Face embeddings and a Groq LLM.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.1--8B-purple) ![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-orange) ![Framework](https://img.shields.io/badge/Framework-LangGraph%20%2B%20Mem0-green) ![Cost](https://img.shields.io/badge/Cost-Free-brightgreen)

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) · Author: **Karthikeyan Dhanakotti**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Groq API key  →  https://console.groq.com/keys
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

# 3. Open the notebook and run top-to-bottom
jupyter notebook AI_Agents_Memory.ipynb
```

> Runs on **Google Colab (free tier)** or any local Python 3.10+ environment.
> The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) into your local Hugging Face cache. Subsequent runs are instant.

---

## Repo Structure

```
CognitiveAgentLab/
└── projects/
    └── AIAgentsMemory/
        ├── README.md                ← this file
        ├── AI_Agents_Memory.ipynb   ← workshop notebook
        └── requirements.txt         ← dependencies
```

---

## What you'll build

A travel-agent style conversational agent that gets progressively smarter across five phases:

| Phase | Memory type | What it demonstrates |
|---|---|---|
| **1** | Working memory | Chat-history-only agent — remembers within a session, forgets across sessions. |
| **2** | Episodic memory | Store past interactions as retrievable "episodes". |
| **3** | Semantic memory | Persist user facts (preferences, budget, destinations) across sessions. |
| **4** | Procedural memory | Reuse learned patterns / preferred workflows. |
| **5** | Long-term memory with Mem0 | Full memory stack with vector search, deduplication, and recall. |

Each phase is a runnable code cell plus markdown explanation, so you can follow the arc end-to-end.

---

## Prerequisites

1. **[Groq API key](https://console.groq.com/keys)** (free tier) — for the LLM (`llama-3.1-8b-instant`).
2. **Local Hugging Face embeddings** — `sentence-transformers/all-MiniLM-L6-v2` runs in-process. No API key, no cloud.
3. Python **3.10+**.

Optional:
- **[Neo4j](https://neo4j.com/)** — for the graph-memory section (uses `neo4j` Python driver).

---

## Tech Stack

- **LLM:** Groq `llama-3.1-8b-instant` (via `langchain-groq`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim)
- **Vector store:** FAISS (`faiss-cpu`)
- **Memory framework:** [`mem0ai`](https://pypi.org/project/mem0ai/)
- **Agent framework:** [`langgraph`](https://langchain-ai.github.io/langgraph/) + `langchain-core`
- **Graph store (optional):** Neo4j

---

## License

See the repository root for license details.
