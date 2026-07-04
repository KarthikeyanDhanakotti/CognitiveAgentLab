# Module 1 — Webinar Script (Speaker Notes)

> **Total time: ~50 min**  •  25 min lecture + 25 min hands-on
> Slides: `docs/Week9_FineTuning_Domain_Adaptation.pptx` — Section: *Strategy & Data*

---

## 0:00 – 0:03 · Hook

> "Show of hands — how many of you have wanted to fine-tune an LLM for your domain?
> Great. Second question: how many actually needed to?
>
> By the end of this hour I want you to answer that question with a decision tree,
> not a gut feel. Because 80% of the time, you don't need to fine-tune —
> and the 20% where you do, the failure mode is almost always the data, not the model."

Show slide: **"Prompt → RAG → Fine-tune"** (the golden rule).

---

## 0:03 – 0:13 · The three approaches to domain adaptation (10 min)

Walk through a **single decision framework**. Draw or show the table:

| Approach | You change… | Cost | Effort | Reversible? |
|---|---|---|---|---|
| Prompt engineering | *What you send* | $0 (tokens only) | Hours | Yes |
| RAG | *What context the model sees* | Vector DB + embed | Days | Yes |
| Fine-tuning | *The model itself* | GPU compute | Days–weeks | **No** |

**Say out loud, twice**:

> "Prompt engineering first. RAG second. Fine-tuning third.
> Each step is more expensive and less reversible than the last."

### When fine-tuning actually wins (short list — worth memorizing)

- Consistent **output format / tone** baked into weights
- Domain-specific **vocabulary usage** (SOB = shortness of breath, not the swear word)
- **Reduced token cost at scale** (shorter prompts needed)
- **Latency**-sensitive apps (less context to process)

### When fine-tuning is the wrong tool

- Up-to-date info → **RAG** wins
- Source citations required → **RAG** wins
- Simple JSON / formatting → **prompt engineering** wins
- <100 queries/day → **prompt engineering** wins

**Presenter tip:** stop here and ask the room to shout out use-cases they've considered
fine-tuning for. Push back on 2–3 of them and route them to prompt or RAG.

---

## 0:13 – 0:20 · QLoRA in five minutes (7 min)

Purpose: give attendees just enough intuition to trust Module 2. Draw on whiteboard/slide:

```
Full Fine-Tuning:  Update ALL 1.5B params.       Needs A100. 40GB+ VRAM.
      │
      ▼
LoRA:              Freeze base weights.           Adds a small side-path.
                   Train only ~9M params (0.6%). Needs ~6GB.
      │
      ▼
QLoRA:             Same as LoRA, but compress
                   frozen base to 4-bit first.    3GB → 0.75GB.
                                                  Fits on Colab T4 (15GB). ✅
```

Key line to deliver:

> "**QLoRA is why this webinar is possible.** Before 2023, fine-tuning a 1.5B model
> needed a $5,000 GPU. Now Google gives you one for free."

---

## 0:20 – 0:25 · Dataset quality preview (5 min)

Put the two datasets side by side. Ask the audience what they predict *before* revealing:

| | ChatDoctor (v1) | WikiDoc reformatted (v2) |
|---|---|---|
| Examples | 112,165 | 2,100 |
| Persona contamination | **63.1%** | 0.0% |
| Boilerplate sign-offs | 28.2% | 0.0% |
| Safety disclaimers | 3.2% | **99.4%** |
| Avg answer length | 603 chars | 910 chars |

**The critical question to leave hanging:**

> "If 63% of your training data starts with 'Hi, welcome to Chat Doctor,'
> what will your model learn to say — no matter what you ask it?"

Don't answer. Say "we'll prove it in Module 2." That's the hook.

---

## 0:25 – 0:50 · Hands-on: dataset audit (25 min)

Attendees open [`notebooks/01_dataset_quality_audit.ipynb`](notebooks/01_dataset_quality_audit.ipynb).

Walk through the notebook in three parts, pausing after each part for questions.

### Part A · Load ChatDoctor + run the regex audit (10 min)

- Load `lavita/ChatDoctor-HealthCareMagic-100k`
- Count regex hits for `r"(?i)^(hi|hello|greetings).*chat ?doctor"`
- Count boilerplate `r"(?i)hope this helps|wishing you good health"`

**Common student stumble:** they use `.startswith("Hi")` and miss case variations.
Push them toward regex with `(?i)`.

### Part B · Load reformatted WikiDoc, compare metrics (10 min)

- Load `jeev1992/wikidoc-healthassist`
- Run the same audit
- Print a side-by-side comparison table

**Teaching moment:** WikiDoc is only 2,100 rows vs 112k. Ask:

> "Which would you rather train on — 100k noisy conversations or 2k clean ones?"

Then reveal that we'll train on both and let evaluation decide.

### Part C · Prompt-engineering baseline (5 min)

Before we spend a single GPU-second, we test whether the **base model + a good system
prompt** already solves the problem. This is the "prompt engineering first" step from
the golden rule, made concrete.

Show the base model producing perfectly reasonable answers with just:

```text
System: You are a knowledgeable healthcare assistant. Provide comprehensive
        explanations. Always recommend consulting a healthcare professional.
```

Deliver the message:

> "If this were good enough for your product, **you would stop right here**.
> We're only proceeding because we want to also embed the *style* and *safety*
> behaviors into weights so we can drop the long system prompt in production."

---

## Wrap-up (last 2 min of Module 1)

Recap in one slide:

1. Try prompting first.
2. Try RAG second.
3. Fine-tune third — only if 1 & 2 aren't enough.
4. Your data determines 90% of the outcome. Audit it before you train.

Transition line into Module 2:

> "Coffee, then we go break a model on purpose."

---

## Presenter cheat-sheet

- If the room is unfamiliar with LLMs, cut QLoRA math and just say "smaller GPU, fewer trainable params."
- If Hugging Face `load_dataset()` rate-limits (many attendees at once), fall back to
  the pre-computed comparison table on the slide.
- If someone asks "why not just use GPT-4?" — answer: cost per query at scale, latency,
  data residency for healthcare (HIPAA), and consistency of output format.
