# Module 2 — Webinar Script (Speaker Notes)

> **Total time: ~70 min** • 20 min lecture + 30 min live demo + 20 min Q&A / results walkthrough
> Slides: `docs/Week9_FineTuning_Domain_Adaptation.pptx` — Section: *QLoRA in Practice*

---

## 1:10 – 1:15 · Frame the two experiments (5 min)

Open both notebook tabs but don't run anything yet. Show the audience this table:

|                        | v1 notebook              | v2 notebook             |
|------------------------|--------------------------|-------------------------|
| Base model             | Qwen2.5-1.5B-Instruct    | Qwen2.5-1.5B-Instruct   |
| LoRA config            | r=16, α=32, dropout=0.05 | r=16, α=32, dropout=0.05|
| Optimizer, batch size  | Identical                | Identical               |
| **Dataset**            | ChatDoctor (noisy, 112k) | WikiDoc (clean, 2.1k)   |
| **Predicted outcome**  | Model gets **worse**     | Model gets **better**   |

Deliver:

> "The only thing we're changing is the data. This is the cleanest possible
> demonstration that data quality — not model choice, not hyperparameter magic —
> is what makes or breaks a fine-tune."

---

## 1:15 – 1:25 · Hyperparameters explained (10 min)

Attendees do **not** need to memorize these — they need intuition.

### LoRA config

```python
LoraConfig(
    r=16,                       # Rank of the side-path. Higher = more capacity.
    lora_alpha=32,              # Scaling. Effective learning rate ≈ alpha/r = 2.0.
    lora_dropout=0.05,          # 5% dropout on adapter neurons. Regularization.
    target_modules="all-linear" # Attach to ALL linear layers (7 per transformer block).
)
```

Analogy to say out loud:

> "Think of the frozen base model as a 4-lane highway. LoRA adds a small
> side road that gently nudges traffic. `r` is how wide the side road is.
> `alpha` is how much you amplify its influence. `dropout` is randomly
> closing 5% of lanes each pass so the adapter can't overfit."

### 4-bit quantization (BitsAndBytesConfig)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # Normal Float 4-bit — best for NN weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Store 4-bit, compute in bf16
    bnb_4bit_use_double_quant=True,         # Quantize the quantization constants too
)
```

One-liner:

> "Weights are stored in 4-bit. Every time we compute, they're briefly
> dequantized to bf16, math happens, result goes back. The base model
> shrinks from 3 GB to 0.75 GB — that's the trick that made this fit on a T4."

### Training math — do this live on the whiteboard

```
Dataset size:    2,000 examples
Effective batch: 8  (batch=1 × grad_accum=8)
Steps per epoch: 2,000 ÷ 8 = 250 steps
v2:              250 × 2 epochs = 500 total steps
Warmup:          50 steps = 10% of training
```

**Key line:**
> "500 steps to shift the model's style. That's why LoRA is fast — you're not
> teaching the model medicine, you're teaching it how to *phrase* medicine."

---

## 1:25 – 1:30 · Start the v2 training run in Colab (5 min)

Open [`02b_qlora_training_v2_wikidoc.ipynb`](notebooks/02b_qlora_training_v2_wikidoc.ipynb).

1. **Runtime → Change runtime type → T4 GPU** (call this out on-camera)
2. Run Step 1: `nvidia-smi` — narrate: "15.8 GB of VRAM, we'll use ~4 GB peak."
3. Run Step 2: `pip install` cell — say "this takes ~90 seconds, don't panic if it looks stuck."
4. Restart kernel when prompted.
5. Run Steps 3–5: import libs, load 4-bit model, run **BEFORE** benchmark.
6. Run Step 6: attach LoRA adapters — point out the trainable params printout:
   `trainable params: 9,232,384 || all params: 1,552,946,688 || trainable%: 0.5945`
7. Run Step 7: `trainer.train()` — **this is the moment to leave running**.

While it trains for ~20–30 min, keep lecturing (below).

---

## 1:30 – 1:50 · While training runs — v1 postmortem (20 min)

Switch to [`02a_qlora_training_v1_chatdoctor.ipynb`](notebooks/02a_qlora_training_v1_chatdoctor.ipynb).
Do **not** run it — instead, open [`results/benchmark_results_v1.json`](results/benchmark_results_v1.json)
and show 2–3 side-by-side examples on screen.

### Example to walk through live

**Prompt:** *"What are the symptoms of Type 2 diabetes?"*

**BEFORE (base model, no fine-tuning):**
> Type 2 diabetes symptoms include increased thirst, frequent urination,
> blurred vision, fatigue, and slow-healing wounds. These symptoms develop
> gradually. Please consult a healthcare professional for proper diagnosis.

**AFTER v1 (ChatDoctor fine-tuned):**
> Hello, thanks for posting your query. The symptoms of diabetes include
> increased urination and thirst. Hope this helps. Wishing you good health.

Ask the room:

> "Which of these do you want your product to send to a patient?"

Then land the punchline:

> "The base model was *already* better. We just spent 45 minutes of GPU time
> teaching it to be worse. This is what fine-tuning on dirty data looks like."

### Explain the three failure modes v1 exhibits

1. **Persona contamination.** The model learned to open every response with a chatbot greeting.
2. **Lost structure.** Bullet points and clinical detail were replaced by a paragraph.
3. **Weakened safety.** "Please consult a healthcare professional" was replaced by
   "Wishing you good health" — a warm sign-off, not a safety instruction.

If time permits, walk through 1–2 more prompts from `benchmark_results_v1.json`.

### 5-minute detour: how v2 data was made

Open [`scripts/data_prep_v2.py`](scripts/data_prep_v2.py). Show attendees the reformatting prompt:

```
Rules:
- Keep ALL medical facts from the original answer.
- Restructure into clear format with bullet points where appropriate.
- Add a brief safety disclaimer at the end.
- Do NOT add greetings, sign-offs, or persona.
- Do NOT add medical information not in the original.
```

Message:

> "We didn't write 2,100 answers by hand. We used **GPT-4o-mini** as a
> reformatting tool with strict rules. Total cost: about $1.50.
> This is a legitimate use of a stronger model — distilling *format*,
> not *knowledge*, into training data."

---

## 1:50 – 2:15 · v2 training finishes — the "after" benchmark (25 min)

Come back to the v2 Colab tab. Training should be complete (or nearly so).

1. Run Step 8: **AFTER** benchmark on the same 10 prompts.
2. Run Step 9: comparison table (pandas DataFrame with BEFORE vs AFTER side-by-side).

### What the audience should see for v2

For the same diabetes prompt:

**BEFORE:**
> Type 2 diabetes symptoms include increased thirst, frequent urination...

**AFTER v2:**
> Common symptoms of Type 2 diabetes include:
>
> - **Increased thirst (polydipsia)** and frequent urination (polyuria)
> - Unexplained fatigue and blurred vision
> - Slow-healing wounds and frequent infections
> - Numbness or tingling in hands and feet
>
> These symptoms often develop gradually. Please consult a healthcare
> professional for proper diagnosis and personalized advice.

Deliver:

> "**Same model.** Same hyperparameters. The only difference is the training
> data. This one preserved detail, added structure, and reinforced safety.
> That's the whole thesis of this webinar in one comparison."

---

## 2:15 – 2:20 · Push the adapter to Hugging Face Hub (5 min)

Run Step 10: `model.push_to_hub("your-username/healthcare-assistant-lora-v2")`

Show attendees the resulting HF repo. Point out:

- The whole "model" is only ~15 MB — it's just the LoRA adapter.
- The `adapter_config.json` tells anyone who downloads it *how* to apply the patch.
- The base model stays where it was — nobody re-uploads Qwen.

This sets up Module 3 perfectly.

---

## Recap slide (last 1 min of Module 2)

1. **Same model → different data → opposite results.** Proven live.
2. **QLoRA on a T4** is a real production workflow, not a toy.
3. Trainable params were **0.6% of the model** — but they matter enormously
   because they sit on top of every attention head.
4. **Always run a before/after benchmark.** If you can't measure it, you can't ship it.

Transition line into Module 3:

> "We have a fine-tuned model living in the cloud. Let's actually use it."

---

## Presenter cheat-sheet

- **Colab GPU dropout risk:** if the runtime disconnects, resume from the pre-computed
  `results/benchmark_results_v2.json`. The comparison table works from that file.
- **Loss monitoring:** if attendees ask about the loss curve, point out that training
  loss on v2 drops from ~1.4 → 0.9 and then plateaus — that's healthy. Loss going to
  near 0 would be overfitting.
- **VRAM question:** peak usage during training is ~4 GB. Free T4 has 15 GB.
  Plenty of headroom — no need to reduce batch size.
- **"Why 1.5B not 7B?"** — 7B in 4-bit needs ~5 GB and trains slower on T4.
  1.5B keeps the demo under 30 min. The technique is identical either way.
