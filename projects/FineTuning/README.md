# Fine-Tuning & Domain Adaptation â€” Webinar (Week 9)

> **End-to-end webinar project** on fine-tuning open-source LLMs for a domain (healthcare),
> deploying them, and evaluating them â€” all runnable on **Google Colab (free T4 GPU)**.

Part of [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab) Â·
Author: **Karthikeyan Dhanakotti**

---

## What you'll build

A healthcare Q&A assistant fine-tuned with **QLoRA** on `Qwen2.5-1.5B-Instruct`,
deployed to **Hugging Face Hub**, and evaluated with **LangSmith** using
**GPT-4o-mini as an LLM-as-Judge** â€” all in a single webinar session.

The workshop is built around a **deliberate failure â†’ recovery arc**:

| Module | You learn | The lesson |
|---|---|---|
| **1. Strategy & Data** | Prompt vs RAG vs Fine-tune. Auditing a real medical dataset. | *Don't fine-tune until you have to.* |
| **2. QLoRA Fine-Tuning** | Fine-tune the same model on 2 datasets â€” one dirty (ChatDoctor), one clean (WikiDoc). | *Data quality > everything.* |
| **3. Deployment & Inference** | Push a LoRA adapter to HF Hub, run inference with adapter toggle. | *Adapters are diffs, not full models.* |
| **4. Evaluation** | Score both models on Helpfulness / Accuracy / Safety with GPT-4o-mini judges. | *Vibes lie. Evaluators don't.* |

---

## Project structure

```text
FineTuning/
â”œâ”€â”€ README.md                              â† You are here
â”œâ”€â”€ requirements.txt                       â† Local (non-Colab) install list
â”œâ”€â”€ .gitignore
â”œâ”€â”€ LICENSE
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ Week9_FineTuning_Domain_Adaptation.pptx     â† Original webinar deck
â”‚   â”œâ”€â”€ webinar_master_script.md                    â† Full 4-hour presenter script
â”‚   â””â”€â”€ github_upload_guide.md                      â† How to push to CognitiveAgentLab
â”œâ”€â”€ assets/                                          â† Diagrams / screenshots
â”‚
â”œâ”€â”€ module_1_strategy_data/
â”‚   â”œâ”€â”€ README.md
â”‚   â”œâ”€â”€ webinar_script.md                           â† Speaker notes (~50 min)
â”‚   â””â”€â”€ notebooks/
â”‚       â””â”€â”€ 01_dataset_quality_audit.ipynb          â† Colab-runnable
â”‚
â”œâ”€â”€ module_2_finetuning_qlora/
â”‚   â”œâ”€â”€ README.md
â”‚   â”œâ”€â”€ webinar_script.md                           â† Speaker notes (~70 min)
â”‚   â”œâ”€â”€ notebooks/
â”‚   â”‚   â”œâ”€â”€ 02a_qlora_training_v1_chatdoctor.ipynb  â† "how NOT to fine-tune"
â”‚   â”‚   â””â”€â”€ 02b_qlora_training_v2_wikidoc.ipynb     â† "how to fine-tune"
â”‚   â”œâ”€â”€ scripts/
â”‚   â”‚   â””â”€â”€ data_prep_v2.py                         â† Reformats WikiDoc via GPT-4o-mini
â”‚   â””â”€â”€ results/
â”‚       â”œâ”€â”€ benchmark_results_v1.json               â† Pre-computed (backup for live demo)
â”‚       â””â”€â”€ benchmark_results_v2.json
â”‚
â”œâ”€â”€ module_3_deployment_inference/
â”‚   â”œâ”€â”€ README.md
â”‚   â”œâ”€â”€ webinar_script.md                           â† Speaker notes (~30 min)
â”‚   â””â”€â”€ notebooks/
â”‚       â””â”€â”€ 03_hf_deploy_inference.ipynb
â”‚
â””â”€â”€ module_4_evaluation_langsmith/
    â”œâ”€â”€ README.md
    â”œâ”€â”€ webinar_script.md                           â† Speaker notes (~50 min)
    â”œâ”€â”€ notebooks/
    â”‚   â””â”€â”€ 04_langsmith_evaluation.ipynb
    â””â”€â”€ results/
        â”œâ”€â”€ BaseModel-Vs-FineTuned-v1.csv
        â””â”€â”€ BaseModel-Vs-FineTuned-v2.csv
```

---

## Quick start â€” Open in Google Colab

Click any badge below to launch the notebook directly in Colab. Once the repo is
uploaded to `KarthikeyanDhanakotti/CognitiveAgentLab`, these links become live.

| # | Module | Notebook | Open |
|---|---|---|---|
| 1 | Strategy & Data | `01_dataset_quality_audit.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_1_strategy_data/notebooks/01_dataset_quality_audit.ipynb) |
| 2a | QLoRA â€” v1 (noisy) | `02a_qlora_training_v1_chatdoctor.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_2_finetuning_qlora/notebooks/02a_qlora_training_v1_chatdoctor.ipynb) |
| 2b | QLoRA â€” v2 (clean)  | `02b_qlora_training_v2_wikidoc.ipynb`     | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_2_finetuning_qlora/notebooks/02b_qlora_training_v2_wikidoc.ipynb) |
| 3 | Deploy & Inference | `03_hf_deploy_inference.ipynb`             | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_3_deployment_inference/notebooks/03_hf_deploy_inference.ipynb) |
| 4 | LangSmith Eval    | `04_langsmith_evaluation.ipynb`            | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_4_evaluation_langsmith/notebooks/04_langsmith_evaluation.ipynb) |

> In Colab: **Runtime â†’ Change runtime type â†’ T4 GPU** for modules 2 & 3.
> Module 1 and Module 4 run on CPU-only runtime.

---

## Accounts you need (create before the webinar)

| Account | Used in | Why | Cost |
|---|---|---|---|
| [Google Colab](https://colab.research.google.com) | Modules 2, 3 | Free T4 GPU | Free |
| [Hugging Face](https://huggingface.co/join) + **Write** token | Modules 2, 3 | Push & load LoRA adapter | Free |
| [OpenAI](https://platform.openai.com/api-keys) API key | Modules 2 (data prep), 4 | GPT-4o-mini judge | ~$1â€“2 |
| [LangSmith](https://smith.langchain.com) API key | Module 4 | Experiment tracking | Free (Developer plan) |

Total out-of-pocket for the full webinar: **â‰ˆ $1â€“2**.

---

## Learning outcomes

By the end of the webinar an attendee will be able to:

1. Decide **when** to fine-tune (vs prompt engineering / RAG) using a written decision framework.
2. Audit a domain dataset for **persona contamination**, boilerplate, and safety gaps.
3. Fine-tune a 1.5B parameter LLM with **QLoRA** on a free Colab T4 GPU.
4. Push a LoRA adapter to **Hugging Face Hub** and load it from anywhere.
5. Use the **adapter-toggle pattern** to A/B test base vs fine-tuned in one model load.
6. Design **LLM-as-Judge** evaluators (helpfulness / accuracy / safety) and run
   before/after experiments in **LangSmith**.
7. Read a comparison report and identify **regressions** before shipping.

---

## Suggested webinar schedule (â‰ˆ4 hours)

| Time | Duration | Segment |
|---|---|---|
| 0:00 | 10 min | Welcome, agenda, verify accounts |
| 0:10 | 50 min | **Module 1** â€” Strategy + dataset audit |
| 1:00 | 10 min | Break |
| 1:10 | 70 min | **Module 2** â€” QLoRA fine-tuning (live) |
| 2:20 | 10 min | Break |
| 2:30 | 30 min | **Module 3** â€” Deployment + inference |
| 3:00 | 50 min | **Module 4** â€” LangSmith evaluation |
| 3:50 | 10 min | Q&A, next steps |

Full second-by-second script is in [`docs/webinar_master_script.md`](docs/webinar_master_script.md).

---

## Credits & references

- Base workshop this project is adapted from:
  [`jeev1992/healthcare-agent-finetuning-workshop`](https://github.com/jeev1992/healthcare-agent-finetuning-workshop)
- Model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Datasets: [`ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k),
  [`medalpaca/medical_meadow_wikidoc`](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc)
- Techniques: QLoRA (Dettmers et al., 2023), LoRA (Hu et al., 2021)
- Tooling: `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `langsmith`

---

## License

MIT â€” see [LICENSE](LICENSE).
