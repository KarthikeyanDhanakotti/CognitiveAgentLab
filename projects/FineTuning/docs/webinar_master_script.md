# Webinar Master Script — Week 9: Fine-Tuning & Domain Adaptation

> **Total runtime:** ~4 hours including breaks
> **Format:** Live coding + lecture + hands-on
> **Slide deck:** [`Week9_FineTuning_Domain_Adaptation.pptx`](Week9_FineTuning_Domain_Adaptation.pptx)
> **All notebooks:** Runnable on **Google Colab (free T4)**
> **Presenter:** Karthikeyan Dhanakotti

This is the top-level running order. Each module has its own detailed script:

- [`../module_1_strategy_data/webinar_script.md`](../module_1_strategy_data/webinar_script.md)
- [`../module_2_finetuning_qlora/webinar_script.md`](../module_2_finetuning_qlora/webinar_script.md)
- [`../module_3_deployment_inference/webinar_script.md`](../module_3_deployment_inference/webinar_script.md)
- [`../module_4_evaluation_langsmith/webinar_script.md`](../module_4_evaluation_langsmith/webinar_script.md)

---

## Running order

| Time | Segment | Notes |
|---|---|---|
| **0:00 – 0:10** | Welcome + account verification | Pre-flight checks |
| **0:10 – 1:00** | Module 1 — Strategy & Data | Lecture 25 min + hands-on 25 min |
| **1:00 – 1:10** | ☕ Break | |
| **1:10 – 2:20** | Module 2 — QLoRA Fine-Tuning | Lecture 20 min + live demo 30 min + Q&A 20 min |
| **2:20 – 2:30** | ☕ Break | |
| **2:30 – 3:00** | Module 3 — Deployment & Inference | Lecture 15 min + live demo 15 min |
| **3:00 – 3:50** | Module 4 — LangSmith Evaluation | Lecture 15 min + live demo 25 min + analysis 10 min |
| **3:50 – 4:00** | Wrap-up + Q&A | Slide with next steps |

---

## 0:00 – 0:10 · Welcome, agenda, verify accounts

### What to say

> "Welcome. Today we're going to do something unusual for a webinar — we're going
> to **deliberately break a model**, then fix it, then measure both attempts.
> By the end you'll have your own fine-tuned healthcare assistant living on
> Hugging Face and an evaluation report proving whether it works.
>
> No slides marathon. Every module has hands-on notebook time. Every notebook
> runs on Colab's free T4 GPU."

### Account verification (do this FIRST — this is the #1 blocker)

Ask attendees to open a chat channel and confirm they have:

- [ ] **Google account** (for Colab)
- [ ] **Hugging Face account** with a **Write** token (not Read — this is the top mistake)
- [ ] **OpenAI API key** with at least $2 credit
- [ ] **LangSmith account** (Developer plan is fine)

Estimated attendees blocked without pre-notice: 15–20%. Sending the setup email a week in advance is the difference between a smooth webinar and one that stalls at minute 20.

### The workshop arc — one slide

Show and read this out loud:

```
Module 1: Audit two datasets. One is clean. One is 63% contaminated.
Module 2: Fine-tune on both. The contaminated one makes the model WORSE.
Module 3: Deploy and see the outputs side by side.
Module 4: Prove it with numbers. Evaluation catches what vibes can't.
```

Deliver:

> "This is not a *how-to-fine-tune* webinar. It's a *when-and-when-not* to fine-tune
> webinar. The most important skill you'll leave with is knowing how to say **no**
> to a fine-tuning project — and how to prove *yes* when you say it."

---

## 0:10 – 1:00 · Module 1 (Strategy & Data)

See [`../module_1_strategy_data/webinar_script.md`](../module_1_strategy_data/webinar_script.md).

**Anchor beats:**
- Prompt → RAG → Fine-tune (the golden rule)
- QLoRA math on the whiteboard (why the free T4 works)
- Dataset audit with regex — 63% persona contamination reveal
- Prompt-engineering baseline: base model is already decent

---

## 1:00 – 1:10 · Break

**Presenter checklist during the break:**
- Open v2 training notebook in a Colab tab; connect T4 runtime **before** attendees return
- Verify HF login works in your Colab
- Have `benchmark_results_v1.json` and `benchmark_results_v2.json` open in another tab as backup

---

## 1:10 – 2:20 · Module 2 (QLoRA Fine-Tuning)

See [`../module_2_finetuning_qlora/webinar_script.md`](../module_2_finetuning_qlora/webinar_script.md).

**Anchor beats:**
- Same model, same LoRA config, different data → opposite outcomes (thesis of workshop)
- Kick off v2 training live at ~1:25 — it runs ~20–30 min in the background
- During training: v1 postmortem on pre-computed results (this is the *deliberate failure* moment)
- Show `data_prep_v2.py` — how GPT-4o-mini distilled clean answers from WikiDoc
- After training completes: run "after" benchmark, compare, push adapter to HF Hub

**Critical fallback plan:** If Colab times out, switch to `results/benchmark_results_v2.json`
and narrate the results as if just generated. The story still lands.

---

## 2:20 – 2:30 · Break

**Presenter checklist:**
- Open Module 3 notebook, load base + adapter to warm up the runtime
- Have your HF adapter URL copied ready to paste

---

## 2:30 – 3:00 · Module 3 (Deployment & Inference)

See [`../module_3_deployment_inference/webinar_script.md`](../module_3_deployment_inference/webinar_script.md).

**Anchor beats:**
- Adapter = 15 MB patch, not a model
- Adapter-toggle pattern: two behaviors in one load
- Generation params for healthcare (low temperature)
- Live: load base + adapter → run 10 prompts with toggle → save `inference_results.json`
- Production reality: right-size GPU, vLLM batching, merge for prod, HIPAA config

---

## 3:00 – 3:50 · Module 4 (LangSmith Evaluation)

See [`../module_4_evaluation_langsmith/webinar_script.md`](../module_4_evaluation_langsmith/webinar_script.md).

**Anchor beats:**
- The "two devs arguing" hook
- Three evaluators: Helpfulness, Accuracy, Safety — each with an explicit rubric
- Create dataset, run base experiment, run FT experiment (~2–3 min total in parallel)
- **The reveal:** Show the compare view. v2 has 2 wins + 1 regression. v1 has 3 regressions.
- The regression check is the whole point.

---

## 3:50 – 4:00 · Wrap-up

### One-slide recap — "What you can do now"

1. Decide when to fine-tune with a written framework (Module 1)
2. Audit a dataset for persona contamination and safety gaps (Module 1)
3. Fine-tune a 1.5B model with QLoRA on a free GPU (Module 2)
4. Push a LoRA adapter to HF Hub and load it with the toggle pattern (Module 3)
5. Run LLM-as-Judge evaluations and detect regressions (Module 4)

### Next steps for attendees

- **Adapt this workshop to your domain.** Swap the dataset — legal, financial, customer
  support — the pipeline is identical.
- **Try a bigger base model** (Qwen 3B, Llama 3.2 3B, Mistral 7B in 4-bit). Same code,
  more VRAM. Colab Pro or a rented A10G handles 7B comfortably.
- **Add production tracing** with LangSmith once you have real user traffic.
- **Read `notes.md` in each module** — deep-dive references we didn't have time for live.

### Closing line

> "You've done a full production loop today — data audit, training, deployment,
> evaluation — in under four hours, on a free GPU, for less than the price of coffee.
> That was impossible three years ago. Go build something useful. Thank you."

---

## Presenter's global cheat-sheet

- **Have three Colab tabs open** before Module 2: v2 training, Module 3 inference,
  Module 4 evaluation. Kill Colab tabs you're not using to reduce disconnect risk.
- **Chat channel monitor:** ideally assign a co-host to answer common setup questions
  in chat while you present. HF token confusion is the #1 chat question.
- **Timing risk:** Module 2 has the highest variance. If training is slow, cut the
  data-prep detour. If training is fast, add a second v1 prompt walkthrough.
- **Q&A parking lot:** questions that don't fit the flow → say "great question,
  I'll come back to that in the wrap-up" and write it on a sticky. Answer 3–5 at the end.
