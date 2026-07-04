# Module 2 — QLoRA Fine-Tuning on Google Colab

**Duration in webinar:** ~70 minutes · **GPU needed:** ✅ Yes (Colab T4)

## What this module answers

1. What actually happens during a QLoRA training run?
2. How does the **same model + same hyperparameters** produce two totally
   different outcomes just from changing the dataset?
3. What are `r`, `lora_alpha`, `bnb_4bit_quant_type="nf4"` — and how do I choose them?

## Two experiments, one lesson

Both notebooks fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with identical LoRA config.
Only the **data** changes:

| Notebook | Data | Expected result |
|---|---|---|
| [`02a_qlora_training_v1_chatdoctor.ipynb`](notebooks/02a_qlora_training_v1_chatdoctor.ipynb) | ChatDoctor (63% persona-contaminated) | Model gets **worse** ❌ |
| [`02b_qlora_training_v2_wikidoc.ipynb`](notebooks/02b_qlora_training_v2_wikidoc.ipynb) | Reformatted WikiDoc (0% contamination) | Model gets **better** ✅ |

## Files

- [`webinar_script.md`](webinar_script.md) — presenter script (20 min lecture + 30 min live demo + 20 min Q&A)
- [`notebooks/02a_qlora_training_v1_chatdoctor.ipynb`](notebooks/02a_qlora_training_v1_chatdoctor.ipynb) — the *deliberate failure*
- [`notebooks/02b_qlora_training_v2_wikidoc.ipynb`](notebooks/02b_qlora_training_v2_wikidoc.ipynb) — the *right way*
- [`scripts/data_prep_v2.py`](scripts/data_prep_v2.py) — how WikiDoc was reformatted via GPT-4o-mini
- [`results/benchmark_results_v1.json`](results/benchmark_results_v1.json) — pre-computed backup for live demo
- [`results/benchmark_results_v2.json`](results/benchmark_results_v2.json) — pre-computed backup for live demo

## Open in Colab

- v1 (noisy data — how NOT to fine-tune):
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_2_finetuning_qlora/notebooks/02a_qlora_training_v1_chatdoctor.ipynb)
- v2 (clean data — the right way):
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_2_finetuning_qlora/notebooks/02b_qlora_training_v2_wikidoc.ipynb)

## Live demo tips

- Training v2 takes ~20–30 min on a Colab T4. Start it running, then lecture *during* the run.
- If Colab times out mid-training, switch to the pre-computed
  `results/benchmark_results_v2.json` — the story still works.
- v1 doesn't need to run live at all. Show its results file for the "before/after" comparison.
