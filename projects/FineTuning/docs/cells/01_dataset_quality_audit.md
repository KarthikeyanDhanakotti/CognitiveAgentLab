# MODULE 1 — 01_dataset_quality_audit.ipynb

_Source: `01_dataset_quality_audit.ipynb`_


## Cell 1 — Markdown

# Module 1 Exercise: Dataset Quality Audit & Prompt Engineering Baseline

## Hands-On: Is Fine-Tuning Even Necessary?

Before writing a single line of training code, good ML engineers ask:

1. **Is my data good enough to train on?** (If not, fine-tuning will make things *worse*)
2. **Can I solve this with prompt engineering alone?** (If yes, fine-tuning is wasted money)

In this exercise, you'll learn to answer both questions by:

- **Part A:** Auditing real medical datasets for quality issues
- **Part B:** Testing how far prompt engineering gets you *without* fine-tuning
- **Part C:** Making a go/no-go decision on fine-tuning

### What You'll Need

- A free Google Colab account (no GPU required for Part A, T4 GPU for Part B)
- ~20 minutes

### How This Connects to Module 2

In Module 2, you'll fine-tune on these exact datasets. By auditing them first, you'll already know:
- Why v1 (ChatDoctor) fine-tuning *degrades* the model
- Why v2 (reformatted WikiDoc) fine-tuning *improves* safety
- What the baseline model can do *without* any fine-tuning


## Cell 2 — Markdown

---

## Part A: Dataset Quality Audit

No GPU needed — we're just loading data and inspecting it.

You'll compare two real medical Q&A datasets:

| | **v1: ChatDoctor** | **v2: WikiDoc (reformatted)** |
|---|---|---|
| Source | `lavita/ChatDoctor-HealthCareMagic-100k` | `jeev1992/wikidoc-healthassist` |
| Origin | Real patient-doctor chat logs | Medical encyclopedia, reformatted by GPT-4o-mini |
| Size | ~112K examples | ~67K examples |
| Format | instruction/input/output | Conversational messages (system/user/assistant) |


## Cell 3 — Code

```python
# Install the datasets library (no GPU needed)
!pip install -q datasets pandas
```

## Cell 4 — Code

```python
from datasets import load_dataset
import pandas as pd
import random

# Load both datasets
print("Loading ChatDoctor dataset...")
chatdoctor = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

print("Loading WikiDoc dataset...")
wikidoc = load_dataset("jeev1992/wikidoc-healthassist", split="train")

print(f"\nChatDoctor: {len(chatdoctor):,} examples")
print(f"WikiDoc:    {len(wikidoc):,} examples")
```

## Cell 5 — Markdown

### Step 1: Inspect the Data Structure

Before looking at quality, understand what each dataset *looks like*.


## Cell 6 — Code

```python
# What columns does each dataset have?
print("ChatDoctor columns:", chatdoctor.column_names)
print("WikiDoc columns:   ", wikidoc.column_names)

print("\n" + "="*80)
print("CHATDOCTOR — Example #0")
print("="*80)
for key in chatdoctor.column_names:
    print(f"\n[{key}]:")
    print(chatdoctor[0][key][:500] if chatdoctor[0][key] else "(empty)")

print("\n" + "="*80)
print("WIKIDOC — Example #0")
print("="*80)
for key in wikidoc.column_names:
    val = wikidoc[0][key]
    if isinstance(val, list):
        for msg in val:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:300]
            print(f"\n  [{role}]: {content}")
    else:
        print(f"\n[{key}]: {str(val)[:500]}")
```

## Cell 7 — Markdown

### Step 2: Spot the Quality Issues

Now let's look at **5 random examples** from each dataset. Your job: **find the problems.**

Read each answer and ask yourself:
- Is this something I'd want my model to say?
- Is the tone appropriate for a healthcare assistant?
- Does it contain any "junk" text that shouldn't be learned?


## Cell 8 — Code

```python
# ── ChatDoctor: 5 Random Samples ──
# Look for: persona contamination, boilerplate, tone issues

random.seed(42)  # Fixed seed so everyone sees the same examples
indices = random.sample(range(len(chatdoctor)), 5)

for i, idx in enumerate(indices):
    ex = chatdoctor[idx]
    print(f"\n{'='*80}")
    print(f"CHATDOCTOR SAMPLE {i+1} (index {idx})")
    print(f"{'='*80}")
    print(f"\nPATIENT: {ex['input'][:400]}")
    print(f"\nDOCTOR:  {ex['output'][:600]}")
    print()
```

## Cell 9 — Code

```python
# ── WikiDoc (reformatted): 5 Random Samples ──
# Look for: formatting, safety disclaimers, tone

random.seed(42)
indices = random.sample(range(len(wikidoc)), 5)

for i, idx in enumerate(indices):
    ex = wikidoc[idx]
    messages = ex.get("messages", [])
    print(f"\n{'='*80}")
    print(f"WIKIDOC SAMPLE {i+1} (index {idx})")
    print(f"{'='*80}")
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if role == "SYSTEM":
            print(f"\n[SYSTEM]: {content[:200]}...")
        elif role == "USER":
            print(f"\nUSER: {content[:400]}")
        elif role == "ASSISTANT":
            print(f"\nASSISTANT: {content[:600]}")
    print()
```

## Cell 10 — Markdown

### Step 3: Quantify the Problems

Eyeballing 5 examples gives intuition, but let's **measure** the issues across the full dataset.

We'll check for common data quality problems:
- **Persona contamination** — greetings like "Hi, welcome to ChatDoctor"
- **Boilerplate sign-offs** — "Hope this helps", "Wish you good health"
- **Very short answers** — under 50 characters (probably useless)
- **Safety disclaimers** — "consult a doctor" (good practice for medical data)


## Cell 11 — Code

```python
# ── Quality Metrics: ChatDoctor ──

persona_phrases = [
    "chat doctor", "chatdoctor", "welcome to",
    "hi, welcome", "hello and welcome",
    "thanks for your question", "thank you for your query"
]

boilerplate_phrases = [
    "hope this helps", "wish you good health",
    "wishing you good health", "take care",
    "feel free to ask", "happy to help"
]

safety_phrases = [
    "consult your doctor", "consult a doctor",
    "see your doctor", "seek medical",
    "healthcare provider", "healthcare professional",
    "medical professional", "consult your healthcare"
]

def audit_chatdoctor(dataset):
    results = {
        "total": len(dataset),
        "persona_contamination": 0,
        "boilerplate_signoffs": 0,
        "very_short_answers": 0,
        "has_safety_disclaimer": 0,
        "answer_lengths": []
    }

    for ex in dataset:
        answer = (ex.get("output", "") or "").lower()
        results["answer_lengths"].append(len(answer))

        if any(p in answer for p in persona_phrases):
            results["persona_contamination"] += 1
        if any(p in answer for p in boilerplate_phrases):
            results["boilerplate_signoffs"] += 1
        if len(answer) < 50:
            results["very_short_answers"] += 1
        if any(p in answer for p in safety_phrases):
            results["has_safety_disclaimer"] += 1

    return results

print("Auditing ChatDoctor dataset... (this may take ~30 seconds)")
cd_results = audit_chatdoctor(chatdoctor)

print(f"\n{'='*60}")
print(f"CHATDOCTOR QUALITY AUDIT — {cd_results['total']:,} examples")
print(f"{'='*60}")
print(f"  Persona contamination:  {cd_results['persona_contamination']:,}  ({cd_results['persona_contamination']/cd_results['total']*100:.1f}%)")
print(f"  Boilerplate sign-offs:  {cd_results['boilerplate_signoffs']:,}  ({cd_results['boilerplate_signoffs']/cd_results['total']*100:.1f}%)")
print(f"  Very short (<50 chars): {cd_results['very_short_answers']:,}  ({cd_results['very_short_answers']/cd_results['total']*100:.1f}%)")
print(f"  Has safety disclaimer:  {cd_results['has_safety_disclaimer']:,}  ({cd_results['has_safety_disclaimer']/cd_results['total']*100:.1f}%)")
print(f"\n  Avg answer length: {sum(cd_results['answer_lengths'])/len(cd_results['answer_lengths']):.0f} chars")
print(f"  Median length:     {sorted(cd_results['answer_lengths'])[len(cd_results['answer_lengths'])//2]} chars")
```

## Cell 12 — Code

```python
# ── Quality Metrics: WikiDoc (reformatted) ──

def audit_wikidoc(dataset):
    results = {
        "total": len(dataset),
        "persona_contamination": 0,
        "boilerplate_signoffs": 0,
        "very_short_answers": 0,
        "has_safety_disclaimer": 0,
        "has_system_prompt": 0,
        "answer_lengths": []
    }

    for ex in dataset:
        messages = ex.get("messages", [])
        answer = ""
        has_system = False
        for msg in messages:
            if msg.get("role") == "assistant":
                answer = (msg.get("content", "") or "").lower()
            if msg.get("role") == "system":
                has_system = True

        results["answer_lengths"].append(len(answer))
        if has_system:
            results["has_system_prompt"] += 1

        if any(p in answer for p in persona_phrases):
            results["persona_contamination"] += 1
        if any(p in answer for p in boilerplate_phrases):
            results["boilerplate_signoffs"] += 1
        if len(answer) < 50:
            results["very_short_answers"] += 1
        if any(p in answer for p in safety_phrases):
            results["has_safety_disclaimer"] += 1

    return results

print("Auditing WikiDoc dataset... (this may take ~30 seconds)")
wd_results = audit_wikidoc(wikidoc)

print(f"\n{'='*60}")
print(f"WIKIDOC QUALITY AUDIT — {wd_results['total']:,} examples")
print(f"{'='*60}")
print(f"  Persona contamination:  {wd_results['persona_contamination']:,}  ({wd_results['persona_contamination']/wd_results['total']*100:.1f}%)")
print(f"  Boilerplate sign-offs:  {wd_results['boilerplate_signoffs']:,}  ({wd_results['boilerplate_signoffs']/wd_results['total']*100:.1f}%)")
print(f"  Very short (<50 chars): {wd_results['very_short_answers']:,}  ({wd_results['very_short_answers']/wd_results['total']*100:.1f}%)")
print(f"  Has safety disclaimer:  {wd_results['has_safety_disclaimer']:,}  ({wd_results['has_safety_disclaimer']/wd_results['total']*100:.1f}%)")
print(f"  Has system prompt:      {wd_results['has_system_prompt']:,}  ({wd_results['has_system_prompt']/wd_results['total']*100:.1f}%)")
print(f"\n  Avg answer length: {sum(wd_results['answer_lengths'])/len(wd_results['answer_lengths']):.0f} chars")
print(f"  Median length:     {sorted(wd_results['answer_lengths'])[len(wd_results['answer_lengths'])//2]} chars")
```

## Cell 13 — Code

```python
# ── Side-by-Side Comparison ──

print(f"{'METRIC':<30} {'CHATDOCTOR':>15} {'WIKIDOC':>15}")
print("─" * 62)
print(f"{'Total examples':<30} {cd_results['total']:>15,} {wd_results['total']:>15,}")
print(f"{'Persona contamination':<30} {cd_results['persona_contamination']/cd_results['total']*100:>14.1f}% {wd_results['persona_contamination']/wd_results['total']*100:>14.1f}%")
print(f"{'Boilerplate sign-offs':<30} {cd_results['boilerplate_signoffs']/cd_results['total']*100:>14.1f}% {wd_results['boilerplate_signoffs']/wd_results['total']*100:>14.1f}%")
print(f"{'Very short answers (<50ch)':<30} {cd_results['very_short_answers']/cd_results['total']*100:>14.1f}% {wd_results['very_short_answers']/wd_results['total']*100:>14.1f}%")
print(f"{'Safety disclaimers':<30} {cd_results['has_safety_disclaimer']/cd_results['total']*100:>14.1f}% {wd_results['has_safety_disclaimer']/wd_results['total']*100:>14.1f}%")
print(f"{'Avg answer length (chars)':<30} {sum(cd_results['answer_lengths'])/len(cd_results['answer_lengths']):>15,.0f} {sum(wd_results['answer_lengths'])/len(wd_results['answer_lengths']):>15,.0f}")
```

## Cell 14 — Markdown

### Discussion: What Did You Find?

Think about these questions before moving on:

1. **Which dataset has more persona contamination?** Why does this matter for fine-tuning?
   - *Hint:* If 30% of your training data starts with "Hi, welcome to ChatDoctor," what will your model learn to say?

2. **Which dataset has more safety disclaimers?** Is that good or bad?
   - *Hint:* For a medical assistant, "consult your doctor" is a feature, not a bug.

3. **Look at the answer lengths.** Which dataset gives more thorough answers?
   - *Hint:* Short answers aren't always bad, but for medical questions, depth matters.

4. **If you could only pick one dataset to fine-tune on, which would you choose?** Why?

---

**Key insight:** You can predict fine-tuning outcomes just by auditing the data. In Module 2, you'll see exactly these predictions come true:
- v1 (ChatDoctor) → model picks up persona artifacts → quality *degrades*
- v2 (WikiDoc) → model picks up safety disclaimers → safety *improves*


## Cell 15 — Markdown

---

## Part B: Prompt Engineering Baseline

Now let's answer the second question: **Can prompt engineering alone handle medical Q&A?**

We'll load the **base Qwen2.5-1.5B-Instruct model** (no fine-tuning) and test it on the same 10 medical questions used in Module 2's before/after benchmark.

**Requires:** T4 GPU (free on Colab). Go to Runtime → Change runtime type → T4 GPU.


## Cell 16 — Code

```python
# ── Install model libraries (if not already installed) ──
!pip install -q transformers accelerate bitsandbytes torch
```

## Cell 17 — Code

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Load in 4-bit to save VRAM (same setup as Module 2)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
print("Model loaded!")
```

## Cell 18 — Code

```python
# ── The 10 benchmark questions (same as Module 2) ──

BENCHMARK_PROMPTS = [
    "What are the common symptoms of Type 2 diabetes?",
    "How does hypertension affect the heart over time?",
    "What is the recommended first-line treatment for mild persistent asthma?",
    "Explain the difference between Type 1 and Type 2 diabetes.",
    "What are the common side effects of ibuprofen?",
    "How is pneumonia typically diagnosed?",
    "What lifestyle changes help manage high cholesterol?",
    "What are the early warning signs of a stroke?",
    "How does metformin work for diabetes management?",
    "What vaccinations are recommended for adults over 65?",
]
```

## Cell 19 — Markdown

### Experiment 1: Zero System Prompt (Just Ask)

What happens if you just ask the model a medical question with no special instructions?


## Cell 20 — Code

```python
def generate_response(model, tokenizer, user_message, system_prompt=None, max_new_tokens=512):
    """Generate a response using the chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()
```

## Cell 21 — Code

```python
# ── Test 3 questions with NO system prompt ──

print("EXPERIMENT 1: No system prompt — just ask the question")
print("=" * 70)

for q in BENCHMARK_PROMPTS[:3]:
    print(f"\nQ: {q}")
    print("-" * 70)
    response = generate_response(model, tokenizer, q, system_prompt=None)
    print(f"A: {response}")
    print()
```

## Cell 22 — Markdown

### Experiment 2: Simple System Prompt

Now add a basic system prompt — one sentence telling the model to be a medical assistant.


## Cell 23 — Code

```python
# ── Test with a SIMPLE system prompt ──

SIMPLE_PROMPT = "You are a helpful medical assistant."

print(f"EXPERIMENT 2: Simple system prompt")
print(f"System: \"{SIMPLE_PROMPT}\"")
print("=" * 70)

for q in BENCHMARK_PROMPTS[:3]:
    print(f"\nQ: {q}")
    print("-" * 70)
    response = generate_response(model, tokenizer, q, system_prompt=SIMPLE_PROMPT)
    print(f"A: {response}")
    print()
```

## Cell 24 — Markdown

### Experiment 3: Detailed System Prompt

Now let's try a carefully crafted system prompt with specific instructions about tone, safety, and format. This is what a prompt engineer would write before considering fine-tuning.


## Cell 25 — Code

```python
# ── Test with a DETAILED system prompt ──

DETAILED_PROMPT = """You are a knowledgeable and thorough healthcare assistant. Follow these rules:

1. Provide detailed, accurate medical information based on current clinical guidelines.
2. Structure your responses with clear sections using markdown formatting.
3. Explain medical terms in plain language when first used.
4. Always include a safety disclaimer advising the user to consult a healthcare professional.
5. Mention when to seek emergency care if the topic warrants it.
6. Be thorough but accessible — aim for 200-400 words.
7. Do not diagnose or prescribe — provide general educational information only."""

print(f"EXPERIMENT 3: Detailed system prompt")
print(f"System: {DETAILED_PROMPT[:100]}...")
print("=" * 70)

for q in BENCHMARK_PROMPTS[:3]:
    print(f"\nQ: {q}")
    print("-" * 70)
    response = generate_response(model, tokenizer, q, system_prompt=DETAILED_PROMPT)
    print(f"A: {response}")
    print()
```

## Cell 26 — Markdown

### Experiment 4: Full Benchmark with Best Prompt

Run all 10 questions with the detailed prompt. Save the results so you can compare them against Module 2's fine-tuned outputs later.


## Cell 27 — Code

```python
import json

# ── Run all 10 questions with the detailed system prompt ──

print("Running full benchmark with detailed system prompt...")
print("=" * 70)

baseline_results = []
for i, q in enumerate(BENCHMARK_PROMPTS):
    print(f"\n[{i+1}/10] {q}")
    response = generate_response(model, tokenizer, q, system_prompt=DETAILED_PROMPT)
    print(f"A: {response[:200]}...\n")
    baseline_results.append({
        "prompt": q,
        "system_prompt": DETAILED_PROMPT,
        "response": response,
        "model": MODEL_ID,
        "method": "prompt_engineering_only"
    })

# Save for later comparison with Module 2 fine-tuned outputs
with open("prompt_engineering_baseline.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print("\nResults saved to prompt_engineering_baseline.json")
print("You'll compare these against fine-tuned outputs in Module 2.")
```

## Cell 28 — Markdown

---

## Part C: Go / No-Go Decision

Now you have the evidence to make a data-driven decision. Fill in this scorecard based on what you observed:


## Cell 29 — Code

```python
# ── Decision Scorecard ──
# Rate each criterion from 1-5 based on your observations.
# Change the numbers below to reflect YOUR assessment.

scorecard = {
    # Prompt Engineering Assessment (from Part B)
    "Base model medical accuracy (1=wrong, 5=excellent)": 0,       # ← Fill this in
    "Base model answer structure (1=messy, 5=well-organized)": 0,  # ← Fill this in
    "Base model safety disclaimers (1=never, 5=always)": 0,        # ← Fill this in
    "Base model follows system prompt (1=ignores, 5=follows)": 0,  # ← Fill this in

    # Dataset Quality Assessment (from Part A)
    "ChatDoctor data quality (1=poor, 5=excellent)": 0,            # ← Fill this in
    "WikiDoc data quality (1=poor, 5=excellent)": 0,               # ← Fill this in
}

print("YOUR DECISION SCORECARD")
print("=" * 60)
for criterion, score in scorecard.items():
    bar = "█" * score + "░" * (5 - score)
    print(f"  {bar}  {score}/5  {criterion}")

avg_base = sum(list(scorecard.values())[:4]) / 4
print(f"\n  Avg base model score:    {avg_base:.1f}/5")
print(f"  ChatDoctor quality:      {list(scorecard.values())[4]}/5")
print(f"  WikiDoc quality:         {list(scorecard.values())[5]}/5")

print("\n" + "=" * 60)
if avg_base >= 4:
    print("  VERDICT: Base model is already strong.")
    print("  Fine-tuning risk: HIGH (could degrade quality).")
    print("  Recommendation: Only fine-tune with very clean data (WikiDoc).")
elif avg_base >= 3:
    print("  VERDICT: Base model is decent but has gaps.")
    print("  Fine-tuning could help IF data quality is high.")
    print("  Recommendation: Fine-tune with WikiDoc. Skip ChatDoctor.")
else:
    print("  VERDICT: Base model needs significant improvement.")
    print("  Fine-tuning is justified with either dataset.")
    print("  Recommendation: Start with WikiDoc (cleaner data).")

print("\n  Remember: In Module 2, you'll see BOTH outcomes.")
print("  Your predictions here will be tested against real results!")
```

## Cell 30 — Markdown

---

## Summary: What You Learned

| Skill | What You Did |
|-------|-------------|
| **Dataset auditing** | Loaded real datasets, searched for quality issues, quantified problems |
| **Quality metrics** | Measured persona contamination, boilerplate, answer length, safety disclaimers |
| **Prompt engineering** | Tested 3 levels of system prompts (none → simple → detailed) |
| **Baseline measurement** | Recorded what the base model can do *before* any fine-tuning |
| **Go/no-go decision** | Used evidence to decide whether fine-tuning is worth the cost |

### Key Takeaways

1. **Audit your data BEFORE training.** You can predict fine-tuning outcomes by inspecting the dataset.
2. **Prompt engineering is surprisingly powerful.** A good system prompt gets you 80% of the way.
3. **Data quality > data quantity.** Persona contamination in ChatDoctor will directly pollute model outputs.
4. **The base model already knows medicine.** Qwen 1.5B-Instruct was trained on medical text — fine-tuning teaches *style*, not *knowledge*.

### Next: Module 2

Now you'll see your predictions play out. In Module 2:
- **v1 notebook:** Fine-tune on ChatDoctor → see the model pick up those persona artifacts you just found
- **v2 notebook:** Fine-tune on WikiDoc → see safety improve because the data has safety disclaimers

You already know the ending. That's the point — **data quality is everything.**

---

*Next: [Module 2 — Colab Fine-Tuning with QLoRA →](../module_2_colab_finetuning/notes.md)*
