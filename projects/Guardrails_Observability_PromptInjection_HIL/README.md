# AI Agent Workshop — Guardrails, Observability, Prompt Injection & Human-in-the-Loop

A production-focused workshop notebook that layers the **four pillars of safe AI agents** onto a baseline LangGraph agent:

1. **Observability** with [LangSmith](https://smith.langchain.com/) tracing
2. **Guardrails** — input/output validation, schema enforcement, and content filters
3. **Prompt-Injection defense** — detection and mitigation of adversarial user input
4. **Human-in-the-Loop (HITL)** — interrupt & approval flows for high-risk tool calls

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.3--70B-purple) ![Framework](https://img.shields.io/badge/Framework-LangGraph%20%2B%20LangSmith-green) ![Cost](https://img.shields.io/badge/Cost-Free-brightgreen)

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) · Author: **Karthikeyan Dhanakotti**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free API keys
$env:GROQ_API_KEY      = "gsk_xxxxxxxxxxxxxxxxxxxx"        # https://console.groq.com/keys
$env:LANGCHAIN_API_KEY = "lsv2_xxxxxxxxxxxxxxxxxxxx"       # https://smith.langchain.com/

# 3. Open the notebook and run top-to-bottom
jupyter notebook AI_AGENT_Workshop_Guardrails_Observability_PromptInjection_HIL.ipynb
```

> Runs on **Google Colab (free tier)** or any local Python 3.10+ environment.
> Every LLM call in the notebook is auto-traced to your LangSmith project (`ai-agents-workshop`).

---

## Repo Structure

```
CognitiveAgentLab/
└── projects/
    └── Guardrails_Observability_PromptInjection_HIL/
        ├── README.md
        ├── AI_AGENT_Workshop_Guardrails_Observability_PromptInjection_HIL.ipynb
        └── requirements.txt
```

---

## Module Map

| Module | You add | Why it matters |
|---|---|---|
| **0 — Setup** | API keys, LangSmith project wiring | Foundation for all later modules. |
| **1 — Baseline + Observability** | LangGraph agent with LangSmith tracing | You cannot fix what you cannot see. |
| **2 — Guardrails** | Pydantic schema + input/output validators | Fail fast on malformed or unsafe I/O. |
| **3 — Prompt-Injection Defense** | Adversarial-input detection layer | Attackers *will* try to hijack your system prompt. |
| **4 — Human-in-the-Loop** | LangGraph `interrupt()` + approval node | Never let an agent execute an irreversible action alone. |

Each module is a runnable code cell with commentary, so you can follow the safety arc end-to-end.

---

## Prerequisites

1. **[Groq API key](https://console.groq.com/keys)** (free tier) — for the LLM (`llama-3.3-70b-versatile`).
2. **[LangSmith API key](https://smith.langchain.com/)** (free tier) — for tracing and observability.
3. Python **3.10+**.

---

## Tech Stack

- **LLM:** Groq `llama-3.3-70b-versatile` (via `langchain-groq`)
- **Agent framework:** [`langgraph`](https://langchain-ai.github.io/langgraph/) + `langchain` + `langchain-core`
- **Observability:** [`langsmith`](https://docs.smith.langchain.com/) (auto-tracing via `LANGCHAIN_TRACING_V2=true`)
- **Guardrails:** `pydantic` schema validation + custom validators
- **Env / secrets:** `python-dotenv`

---

## What "Good" Looks Like at the End

- Every agent turn shows up as a trace in LangSmith with tokens, latency and tool-call breakdown.
- Malformed or unsafe outputs are rejected before reaching the user.
- Known prompt-injection patterns are flagged and neutralized.
- High-risk tool calls (e.g. sending emails, database writes) pause and wait for a human `approve / reject` decision.

---

## License

See the repository root for license details.
