# Module 4 â€” LangSmith Evaluation & Observability

**Duration in webinar:** ~50 minutes Â· **GPU needed:** âŒ No

## What this module answers

1. How do I *prove* my fine-tuned model is better â€” with numbers, not vibes?
2. What does an LLM-as-Judge evaluator actually do?
3. How do I detect a regression before I ship?

## The 3 evaluators we build

| Evaluator | Question it answers | Score 5 example | Score 1 example |
|---|---|---|---|
| **Helpfulness** | Does the answer give the user something actionable? | Clear bullet list + next steps | "Consult a doctor." (nothing else) |
| **Accuracy** | Is the medical content factually correct? | Standard-of-care advice | Contradicts current guidelines |
| **Safety** | Does it include a professional-consultation disclaimer? | Explicit disclaimer | No disclaimer / makes a diagnosis |

## Two experiments we compare

| Experiment | Result | Why |
|---|---|---|
| **v1 â€” ChatDoctor** | All three metrics **regressed** âŒ | Dirty data taught the model a chatbot persona and killed structure. |
| **v2 â€” WikiDoc reformatted** | Accuracy +0.06 âœ…, Safety +0.10 âœ…, Helpfulness âˆ’0.16 âŒ | Clean data works, but our WikiDoc reformat was too terse â€” accuracy & safety up, but answers got shorter. |

**This is the whole point of evaluation.** Without numbers, we'd have shipped v2 thinking
it was a pure win. The eval catches the helpfulness regression.

## Files

- [`webinar_script.md`](webinar_script.md) â€” presenter script (15 min lecture + 25 min live demo + 10 min analysis)
- [`notebooks/04_langsmith_evaluation.ipynb`](notebooks/04_langsmith_evaluation.ipynb) â€” creates LangSmith dataset, runs the 3 evaluators, compares experiments
- [`results/BaseModel-Vs-FineTuned-v1.csv`](results/BaseModel-Vs-FineTuned-v1.csv) â€” raw per-question scores for v1
- [`results/BaseModel-Vs-FineTuned-v2.csv`](results/BaseModel-Vs-FineTuned-v2.csv) â€” raw per-question scores for v2

## Cost & speed

- **60 GPT-4o-mini judge calls** (10 prompts Ã— 2 models Ã— 3 evaluators)
- Runs in **~2â€“3 minutes** in parallel
- **~$0.06** total OpenAI cost
- **60 traces** out of 5,000/month LangSmith free-tier limit â†’ 1.2% used

## Open in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_4_evaluation_langsmith/notebooks/04_langsmith_evaluation.ipynb)
