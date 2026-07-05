# Data Analyst with Agent Skills — Agentic CSV Analytics with LangGraph

A hands-on workshop notebook that builds an **agentic data-analyst** you can point at any CSV. A LangGraph state machine plans the analysis, writes and executes Python/pandas code, produces charts, and can even email the results — mimicking the workflow of a junior analyst but on autopilot.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.1--8B-purple)
![Framework](https://img.shields.io/badge/Framework-LangGraph-green)
![Skills](https://img.shields.io/badge/Skills-CSV%20%7C%20Charts%20%7C%20Email-orange)

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) · Author: **Karthikeyan Dhanakotti**

---

## What you'll build

A LangGraph-based data-analyst agent with **pluggable skills**:

| Skill | What it does |
|---|---|
| **CSV Loader** | Upload a CSV and load it into a pandas DataFrame. |
| **Schema Profiler** | Inspects columns, dtypes, null rates, and cardinality so the LLM knows the shape of the data. |
| **Planner** | Turns a user question ("what were top-5 products by revenue in Q3?") into an analysis plan. |
| **Code Writer** | LLM writes safe pandas code to answer the plan. |
| **Executor** | Runs the generated code in a sandboxed namespace and captures results / errors. |
| **Chart Generator** | Produces matplotlib visualizations when a chart is more useful than a table. |
| **Emailer** | Optionally emails the resulting summary + chart via SMTP. |
| **Graph Visualizer** | Renders the LangGraph itself (`grandalf`) so you can see the agent's control flow. |

The agent is designed to be extended — every skill is a graph node you can swap or add to.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Groq API key  →  https://console.groq.com/keys
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

# 3. (Optional) configure SMTP env vars if you want the email skill:
#    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO

# 4. Open the notebook and run top-to-bottom
jupyter notebook DataAnalystwithAgentSkills.ipynb
```

> Runs on **Google Colab (free tier)** or any local Python 3.10+ environment.
> Bring your own CSV — the notebook uses `google.colab.files.upload()` on Colab and can be swapped for a local file path elsewhere.

---

## Prerequisites

1. **[Groq API key](https://console.groq.com/keys)** (free tier) — for the LLM (`llama-3.1-8b-instant`).
2. Python **3.10+**.
3. A CSV file to analyze.
4. *(Optional)* SMTP credentials for the email skill.

> **Security note:** never hard-code your API key in the notebook. Use environment variables or Colab Secrets (`google.colab.userdata`). The committed notebook uses a `YOUR_GROQ_API_KEY_HERE` placeholder — replace it at runtime, do not commit real keys.

---

## Tech Stack

- **LLM:** Groq `llama-3.1-8b-instant` (via the `groq` SDK)
- **Agent framework:** [`langgraph`](https://langchain-ai.github.io/langgraph/)
- **Data:** `pandas`
- **Charts:** `matplotlib`
- **Graph visualization:** [`grandalf`](https://pypi.org/project/grandalf/)
- **Email:** `smtplib` (stdlib)

---

## Repo Structure

```
CognitiveAgentLab/
└── projects/
    └── DataAnalystwithAgentSkills/
        ├── README.md                            ← this file
        ├── DataAnalystwithAgentSkills.ipynb     ← workshop notebook
        └── requirements.txt                     ← dependencies
```

---

## License

See the repository root for license details.
