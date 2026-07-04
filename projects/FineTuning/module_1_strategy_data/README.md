# Module 1 â€” Strategy & Dataset Preparation

**Duration in webinar:** ~50 minutes Â· **GPU needed:** âŒ No

## What this module answers

1. When should I fine-tune, and when should I *not*?
2. What does "good" fine-tuning data look like?
3. How do I audit a real medical dataset before spending GPU hours on it?

## Files

- [`webinar_script.md`](webinar_script.md) â€” full presenter script (25 min lecture + 25 min hands-on)
- [`notebooks/01_dataset_quality_audit.ipynb`](notebooks/01_dataset_quality_audit.ipynb) â€” attendees run this in Colab

## Key results the attendees should see

| Metric | ChatDoctor (v1 data) | WikiDoc reformatted (v2 data) |
|---|---|---|
| Examples | 112,165 | 2,100 |
| Persona contamination ("Hi, Chat Doctor hereâ€¦") | **63.1%** | **0.0%** |
| Boilerplate sign-offs | 28.2% | 0.0% |
| Safety disclaimers | 3.2% | 99.4% |
| Avg answer length (chars) | 603 | 910 |

The audit numbers *predict* what will happen in Module 2 â€” v1 will damage the model,
v2 will improve it.

## Open in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_1_strategy_data/notebooks/01_dataset_quality_audit.ipynb)
