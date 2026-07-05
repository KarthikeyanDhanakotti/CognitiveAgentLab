# LLM as a Judge — Evaluation Design Patterns for Production GenAI

A hands-on tutorial notebook that walks through **LLM-as-a-Judge**, the practical evaluation technique used to score, compare, and monitor GenAI systems in production. Built around a small "golden set" for a customer-support RAG bot with two candidate generators (`v1` = baseline, `v2` = improved), the notebook shows how to decide which to ship and how to catch regressions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.1--8B-purple)
![Focus](https://img.shields.io/badge/Focus-Evaluation-red)
![Cost](https://img.shields.io/badge/Cost-Free-brightgreen)

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) · Author: **Karthikeyan Dhanakotti**

---

## What you'll build

Five judge **design patterns** you can drop into any GenAI evaluation harness:

| Pattern | Type | What it does |
|---|---|---|
| **A** | Pass / Fail judge | Binary "is this answer acceptable to ship?" |
| **B** | Rubric scoring (1–5) | Multi-criterion structured judgment with chain-of-thought. |
| **C** | Pairwise ranking | "Which of A / B is better?" — with position-swap to fight order bias. |
| **D** | Reference-based judge | Answer vs. ground truth — **correctness**. |
| **E** | Reference-FREE judge | Answer vs. retrieved context — **faithfulness**, no GT needed. |

Plus the surrounding production concerns:

- **Calibration reports** — how well does the judge match human labels?
- **Stability checks** — variance across runs, temperature, and paraphrase.
- **Regression reports** — v1 → v2 shifts on the golden set.
- **Failure-mode demos** — position bias, verbosity bias, self-preference.
- **Ship decision** — a defensible go / no-go summary.
- **When NOT to use** an LLM judge — clear anti-patterns.

Companion code to the conference tutorial *"Choosing LLM as a Judge: Evaluation Design Patterns for Production GenAI"*.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Groq API key  →  https://console.groq.com/keys
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

# 3. Open the notebook and run top-to-bottom
jupyter notebook LLM_AS_JUDGE.ipynb
```

> Runs on **Google Colab (free tier)** or any local Python 3.10+ environment.

---

## Prerequisites

1. **[Groq API key](https://console.groq.com/keys)** (free tier) — for the LLM judge (`llama-3.1-8b-instant`).
2. Python **3.10+**.

---

## Tech Stack

- **LLM:** Groq `llama-3.1-8b-instant` (via `langchain-groq`)
- **Schema / parsing:** `pydantic` for structured judge outputs
- **Utils:** `json`, `statistics`, `random` (stdlib) for calibration + stability

---

## Repo Structure

```
CognitiveAgentLab/
└── projects/
    └── LLM_AS_JUDGE/
        ├── README.md            ← this file
        ├── LLM_AS_JUDGE.ipynb   ← workshop notebook
        └── requirements.txt     ← dependencies
```

---

## License

See the repository root for license details.
