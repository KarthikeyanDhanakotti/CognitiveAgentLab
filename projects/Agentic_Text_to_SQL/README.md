# Agentic Text-to-SQL — Natural Language → SQL with LangGraph

A hands-on workshop notebook that builds an **agentic Text-to-SQL system** end-to-end. A LangGraph state machine turns natural-language questions into safe, executed SQL against a SQLite sample company database, then returns a natural-language answer — all wrapped in a Gradio UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.1--8B-purple)
![Framework](https://img.shields.io/badge/Framework-LangGraph-green)
![UI](https://img.shields.io/badge/UI-Gradio-orange)
![Cost](https://img.shields.io/badge/Cost-Free-brightgreen)

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) · Author: **Karthikeyan Dhanakotti**

---

## What you'll build

A multi-step SQL agent that:

| Step | Node | What it does |
|---|---|---|
| **1** | Schema Loader | Introspects the SQLite DB and exposes tables/columns to the LLM. |
| **2** | Planner | Decomposes the user question into a SQL intent. |
| **3** | SQL Generator | LLM writes a safe `SELECT` against the known schema. |
| **4** | Validator / Guardrail | Blocks DML/DDL, checks syntax, rejects unsafe queries. |
| **5** | Executor | Runs the SQL against SQLite and captures the result set. |
| **6** | Explainer | LLM converts the rows back into a natural-language answer. |

The graph runs inside a Gradio chat UI so you can query the sample company DB conversationally.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Groq API key  →  https://console.groq.com/keys
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

# 3. Open the notebook and run top-to-bottom
jupyter notebook Agentic_Text_to_SQL.ipynb
```

> Runs on **Google Colab (free tier)** or any local Python 3.10+ environment.
> The notebook builds a fresh `sample_company.db` SQLite file on first run — no external DB required.

---

## Prerequisites

1. **[Groq API key](https://console.groq.com/keys)** (free tier) — for the LLM (`llama-3.1-8b-instant`).
2. Python **3.10+**.
3. Nothing else — SQLite is bundled with Python and the sample data is generated inside the notebook.

---

## Tech Stack

- **LLM:** Groq `llama-3.1-8b-instant` (via `langchain-groq`)
- **Agent framework:** [`langgraph`](https://langchain-ai.github.io/langgraph/) + `langchain-core`
- **Database:** SQLite (`sqlite3` stdlib) + SQLAlchemy
- **UI:** [Gradio](https://www.gradio.app/)

---

## Repo Structure

```
CognitiveAgentLab/
└── projects/
    └── Agentic_Text_to_SQL/
        ├── README.md                    ← this file
        ├── Agentic_Text_to_SQL.ipynb    ← workshop notebook
        └── requirements.txt             ← dependencies
```

---

## License

See the repository root for license details.
