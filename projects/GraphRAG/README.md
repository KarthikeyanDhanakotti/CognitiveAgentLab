# GraphRAG — Knowledge-Graph-Enhanced Retrieval-Augmented Generation

A hands-on notebook that walks through **Microsoft GraphRAG**, an advanced RAG system that builds an LLM-generated knowledge graph over a corpus, detects communities, summarizes them, and uses those summaries to answer complex, multi-hop questions that traditional vector RAG struggles with.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%2F%20Azure%20OpenAI-purple)
![Framework](https://img.shields.io/badge/Framework-GraphRAG-green)
![Retrieval](https://img.shields.io/badge/Retrieval-Knowledge%20Graph-orange)

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) · Author: **Karthikeyan Dhanakotti**

---

## What you'll build

An end-to-end GraphRAG pipeline that goes far beyond chunk-and-embed RAG:

| Stage | What it does |
|---|---|
| **1. Text Chunking** | Splits source documents into manageable chunks. |
| **2. Element Extraction** | Uses an LLM to extract entities and relationships from each chunk. |
| **3. Graph Construction** | Assembles entities (nodes) and relationships (edges) into a knowledge graph. |
| **4. Community Detection** | Runs Leiden-style clustering to find tightly-connected communities of entities. |
| **5. Community Summarization** | Generates natural-language summaries for each community. |
| **6. Local Query** | Answers narrow questions using nearby entities + community context. |
| **7. Global Query** | Synthesizes a comprehensive answer by combining community-level summaries. |

Use this when your questions require **connecting the dots** across a corpus rather than looking up a single fact.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI or Azure OpenAI credentials in a .env file
#    (see the notebook — supports both OpenAI and Azure OpenAI)
#    OPENAI_API_KEY=sk-...
#    or
#    AZURE_OPENAI_API_KEY=...
#    AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
#    AZURE_OPENAI_API_VERSION=2024-06-01

# 3. Open the notebook and run top-to-bottom
jupyter notebook GraphRAG.ipynb
```

> Runs on **Google Colab** or any local Python 3.10+ environment.
> Graph construction is LLM-heavy — expect meaningful token usage on non-trivial corpora.

---

## Prerequisites

1. **OpenAI API key** ([platform.openai.com](https://platform.openai.com/)) **or** an **Azure OpenAI** deployment with a chat model (e.g. `gpt-4o`) and a text-embedding model (e.g. `text-embedding-3-large`).
2. Python **3.10+**.

---

## Tech Stack

- **LLM:** OpenAI `gpt-4o` (or Azure OpenAI equivalent)
- **Embeddings:** `text-embedding-3-large`
- **Graph framework:** [`graphrag`](https://microsoft.github.io/graphrag/) (Microsoft Research)
- **Utils:** `openai`, `python-dotenv`, `beautifulsoup4`, `pyyaml`

---

## Repo Structure

```
CognitiveAgentLab/
└── projects/
    └── GraphRAG/
        ├── README.md          ← this file
        ├── GraphRAG.ipynb     ← workshop notebook
        └── requirements.txt   ← dependencies
```

---

## License

See the repository root for license details.
