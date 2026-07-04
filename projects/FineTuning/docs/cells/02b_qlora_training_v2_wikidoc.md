# MODULE 2b — 02b_qlora_training_v2_wikidoc.ipynb

_Source: `02b_qlora_training_v2_wikidoc.ipynb`_


## Cell 1 — Markdown

# Module 2 (v2): Fine-Tuning Done Right — Data Quality Is Everything

## Welcome!

In this notebook, you'll fine-tune the **same** `Qwen2.5-1.5B-Instruct` model from the v1 notebook — but this time with **clean, well-formatted data** instead of noisy ChatDoctor conversations.

**The key difference:** Instead of raw ChatDoctor chat logs (full of greetings and persona artifacts), you'll train on **reformatted WikiDoc** data — encyclopedic medical content that was cleaned and restructured into conversational Q&A format using GPT-4o-mini.

### What You'll Learn

> **Key takeaway: Data quality is everything. Clean data preserves model strengths while adding desired behaviors (like safety disclaimers).**

By the end of this notebook, you'll understand:
- How clean, well-formatted training data produces better results
- Why gentler hyperparameters (lower learning rate, fewer epochs) prevent catastrophic forgetting
- How to add safety behaviors through data rather than prompting
- The contrast between v1 (noisy data → degradation) and v2 (clean data → improvement)

### What Will Happen (Spoiler)

| | Before Fine-Tuning (1.5B) | After Fine-Tuning |
|---|---|---|
| **Answer quality** | Detailed, well-structured | Structure preserved, safety added |
| **Medical accuracy** | Generally accurate | Accuracy preserved |
| **Safety disclaimers** | Inconsistent | Consistently recommends consulting a professional |
| **Style** | Professional assistant tone | Same tone, no persona contamination |
| **Verdict** | ✅ Already good | ✅ Improved — gained safety without losing quality |

### How This Connects to v1

In v1, you saw that fine-tuning on **noisy ChatDoctor data** made the model worse. Here, you'll see that the **same model** trained on **clean data** produces better results. Same model, same LoRA config — the only difference is data quality.

### Prerequisites

Before running this notebook, make sure you've run `data_prep_v2.py` to upload the reformatted WikiDoc dataset to Hugging Face. If you haven't, the dataset is already available at `jeev1992/wikidoc-healthassist`.

### ⚙️ How to Run This Notebook

1. **Open in Google Colab** → File → Upload notebook
2. **Select GPU Runtime** → Runtime → Change runtime type → T4 GPU
3. **Run cells in order** from top to bottom (Shift+Enter each cell)
4. **You'll need:** A free [Hugging Face](https://huggingface.co/) account with a Write token


## Cell 2 — Markdown

## Step 1: Check Your GPU & Install Libraries

Same setup as v1 — you need a GPU with enough VRAM for 4-bit quantized training.


## Cell 3 — Code

```python
# ── GPU Check ──
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Cell 4 — Code

```python
# ── Install the libraries you need ──
# Same stack as v1. See v1 notebook for detailed explanations of each package.
!pip install -q \
    "transformers>=4.45.0,<4.47.0" \
    "accelerate>=0.34.0" \
    "peft>=0.12.0" \
    "trl>=0.10.0" \
    "bitsandbytes>=0.43.0" \
    "datasets>=2.20.0" \
    "huggingface-hub>=0.24.0" \
    pandas

import importlib
importlib.invalidate_caches()
print("✅ All packages installed!")
```

## Cell 5 — Markdown

## Step 2: Import Libraries & Set Your Configuration

Same model as v1 (`Qwen2.5-1.5B-Instruct`), but with two key differences:

1. **Dataset:** `jeev1992/wikidoc-healthassist` — WikiDoc reformatted via GPT-4o-mini into conversational healthcare Q&A
2. **Hyperparameters:** Gentler training — lower learning rate (5e-5 vs 2e-4) and fewer epochs (2 vs 3) to avoid overwriting the model's existing knowledge


## Cell 6 — Code

```python
import os
import json
import torch
import pandas as pd

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig

# ══════════════════════════════════════════════════════════
#  YOUR CONFIGURATION
# ══════════════════════════════════════════════════════════

# Same 1.5B model as v1 — isolates the effect of data quality.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Reformatted WikiDoc dataset — clean, conversational, with safety disclaimers.
# Created by data_prep_v2.py using GPT-4o-mini to reformat raw WikiDoc entries.
DATASET_ID = "jeev1992/wikidoc-healthassist"

OUTPUT_DIR = "./healthcare-assistant-lora-v2"
HF_REPO_ID = "jeev1992/healthcare-assistant-lora-v2"  # ← Change to your username

NUM_TRAIN_SAMPLES = 2000
NUM_EVAL_SAMPLES = 100
MAX_SEQ_LENGTH = 512

# Same system prompt as v1 — keeps the comparison fair.
SYSTEM_PROMPT = (
    "You are a knowledgeable and thorough healthcare assistant. "
    "When answering medical questions, provide comprehensive explanations "
    "with relevant clinical details, mechanisms of action, and practical guidance. "
    "Structure your answers clearly. "
    "Always recommend consulting a healthcare professional for serious concerns."
)

print("Configuration loaded ✓")
print(f"Model: {MODEL_ID}")
print(f"Dataset: {DATASET_ID}")
print(f"Train samples: {NUM_TRAIN_SAMPLES}")
```

## Cell 7 — Code

```python
# ── Log in to Hugging Face ──
from huggingface_hub import notebook_login
notebook_login()
```

## Cell 8 — Markdown

## Step 3: Load the Pre-Reformatted Dataset

Unlike v1 (which downloaded raw ChatDoctor data and cleaned it with regex), the v2 dataset has already been reformatted by `data_prep_v2.py`. Each example is a complete conversation:

```
messages: [
  {role: "system",    content: "You are a knowledgeable healthcare assistant..."},
  {role: "user",      content: "What are the causes of myocardial infarction?"},
  {role: "assistant", content: "Myocardial infarction (heart attack) occurs when..."}
]
```

The reformatted answers are 200-400 words, have structured explanations with key points, include safety disclaimers, and contain no persona artifacts.


## Cell 9 — Code

```python
# ── Load the pre-reformatted dataset from Hugging Face ──
# This dataset was created by data_prep_v2.py:
#   1. Loaded raw WikiDoc (medalpaca/medical_meadow_wikidoc)
#   2. Sent each entry to GPT-4o-mini to reformat into conversational style
#   3. Uploaded to HF Hub as jeev1992/wikidoc-healthassist
#
# No cleaning needed here — it's already in chat format with system/user/assistant messages.
raw_ds = load_dataset(DATASET_ID, split="train")
print(f"Dataset size: {len(raw_ds)} examples")
print(f"Columns: {raw_ds.column_names}")

# Shuffle and split into train/eval
raw_ds = raw_ds.shuffle(seed=42)
train_ds = raw_ds.select(range(min(NUM_TRAIN_SAMPLES, len(raw_ds))))
eval_ds = raw_ds.select(range(NUM_TRAIN_SAMPLES, min(NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES, len(raw_ds))))

print(f"\nTrain: {len(train_ds)}  |  Eval: {len(eval_ds)}")
print(f"\nSample conversation:")
for msg in train_ds[0]["messages"]:
    role = msg["role"]
    text = msg["content"][:200]
    print(f"  [{role}] {text}...")
```

## Cell 10 — Markdown

## Step 4: Set Up Benchmark Prompts

Same 10 questions as v1 — this is critical for a fair comparison.


## Cell 11 — Code

```python
# ── Same benchmark prompts as v1 ──
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

print(f"Benchmark set: {len(BENCHMARK_PROMPTS)} questions ready")
```

## Cell 12 — Markdown

## Step 5: Load the Model (4-bit Quantized)

Same model and quantization as v1. See the v1 notebook for detailed explanations of 4-bit quantization and why we use NF4.


## Cell 13 — Code

```python
# ── Quantization settings (same as v1) ──
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Load the model ──
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# ── Load the tokenizer ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {MODEL_ID}")
print(f"Total params:  {total_params / 1e6:.1f}M")
print(f"VRAM used:     {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Cell 14 — Markdown

## Step 6: BEFORE — Test the Base Model

Same base model as v1, so these "before" answers should be identical. This confirms our starting point is the same — any differences in the "after" come purely from the training data.


## Cell 15 — Code

```python
def generate_response(model, tokenizer, user_prompt, max_new_tokens=256):
    """Ask the model a question and get its answer. Same function as v1."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Run the "BEFORE" test ──
print("=" * 60)
print("BEFORE FINE-TUNING — Base 1.5B Model Answers")
print("=" * 60)

base_outputs = []
for i, prompt in enumerate(BENCHMARK_PROMPTS, 1):
    response = generate_response(model, tokenizer, prompt)
    base_outputs.append({"prompt": prompt, "response": response})
    print(f"\n[{i}] {prompt}")
    print(f"    → {response[:300]}...")
    print("-" * 60)
```

## Cell 16 — Markdown

## Step 7: Set Up LoRA Adapters

Same LoRA config as v1 (r=16, alpha=32). The only variable between v1 and v2 is the training data.


## Cell 17 — Code

```python
# ── LoRA Configuration (same as v1) ──
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

print(f"LoRA config ready: r={lora_config.r}, alpha={lora_config.lora_alpha}")
```

## Cell 18 — Markdown

## Step 8: Train! 🚀

**Key difference from v1:** Gentler hyperparameters.

| Setting | v1 (ChatDoctor) | v2 (WikiDoc) | Why |
|---------|----------------|--------------|-----|
| Learning rate | 2e-4 | **5e-5** | 4× lower — less risk of overwriting existing knowledge |
| Epochs | 3 | **2** | Fewer passes — the data is cleaner so the model needs less repetition |
| Warmup steps | 75 | **50** | Slightly faster ramp-up since the learning rate is already low |

**Why gentler?** The model already knows medicine. We want to *nudge* it (add safety disclaimers, reinforce clean formatting) — not *overwrite* its knowledge. A high learning rate + many epochs would cause **catastrophic forgetting**, where the model loses pre-trained knowledge.


## Cell 19 — Code

```python
tokenizer.model_max_length = MAX_SEQ_LENGTH

# ── Training settings — gentler than v1 ──
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,                      # 2 epochs (v1 used 3) — cleaner data needs less
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,           # Effective batch size = 8
    learning_rate=5e-5,                      # 4× lower than v1 (5e-5 vs 2e-4)
    lr_scheduler_type="cosine",
    warmup_steps=50,                         # Slightly less warmup since LR is already low
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
    report_to="none",
    seed=42,
)

# ── Create the Trainer ──
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=training_args,
    peft_config=lora_config,
)

trainable, total = trainer.model.get_nb_trainable_parameters()
print(f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M  ({100 * trainable / total:.2f}%)")

# ── START TRAINING ──
print(f"\nStarting training... (~15-25 minutes on T4)")
print(f"Total steps: ~{NUM_TRAIN_SAMPLES * 2 // 8}")
train_result = trainer.train()

print(f"\n✅ Training complete!")
print(f"  Training loss:    {train_result.training_loss:.4f}")
print(f"  Training runtime: {train_result.metrics['train_runtime']:.0f}s")
```

## Cell 20 — Markdown

## Step 9: Save Your Trained Adapter


## Cell 21 — Code

```python
# ── Save the LoRA adapter and tokenizer ──
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR} ✓")

import pathlib
adapter_size = sum(f.stat().st_size for f in pathlib.Path(OUTPUT_DIR).rglob("*") if f.is_file())
print(f"Adapter size: {adapter_size / 1e6:.1f} MB")
```

## Cell 22 — Markdown

## Step 10: AFTER — Test the Fine-Tuned Model

**What to look for this time (compared to v1):**
- Does the model KEEP its detailed, structured answers? (v1 lost them)
- Does it ADD safety disclaimers? ("consult a healthcare professional")
- Is there ANY persona contamination? (v1 picked up "Hello, Chat Doctor")

If the data quality thesis is correct, you should see **improvement without degradation** — the opposite of v1.


## Cell 23 — Code

```python
# ── Free GPU memory from training ──
if 'model' in dir():   del model
if 'trainer' in dir():  del trainer
torch.cuda.empty_cache()

# ── Load a FRESH base model + your trained v2 adapter ──
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
ft_model.eval()

ft_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
ft_tokenizer.pad_token = ft_tokenizer.eos_token

print("Fine-tuned v2 model loaded ✓")

# ── Run the "AFTER" test ──
print("=" * 60)
print("AFTER FINE-TUNING — v2 (WikiDoc) Model Answers")
print("=" * 60)

finetuned_outputs = []
for i, prompt in enumerate(BENCHMARK_PROMPTS, 1):
    response = generate_response(ft_model, ft_tokenizer, prompt)
    finetuned_outputs.append({"prompt": prompt, "response": response})
    print(f"\n[{i}] {prompt}")
    print(f"    → {response[:300]}...")
    print("-" * 60)
```

## Cell 24 — Markdown

## Step 11: Side-by-Side Comparison

**What you should see (vs v1):**
- ✅ Answer structure preserved (v1 lost it)
- ✅ Safety disclaimers added (v1 added ChatDoctor greetings instead)
- ✅ No persona contamination (v1 had "Hello, welcome to Chat Doctor")
- ✅ Medical accuracy maintained (v1 was similar but less precise)

**The takeaway:** Same model, same LoRA config — the only difference was data quality.


## Cell 25 — Code

```python
# ── Side-by-side comparison ──
comparison_rows = []
for base, ft in zip(base_outputs, finetuned_outputs):
    comparison_rows.append({
        "Prompt":            base["prompt"],
        "Base Model (1.5B)": base["response"][:200],
        "Fine-Tuned v2":     ft["response"][:200],
    })

df = pd.DataFrame(comparison_rows)
pd.set_option("display.max_colwidth", 200)
df
```

## Cell 26 — Markdown

## Step 12: Upload to Hugging Face Hub


## Cell 27 — Code

```python
# ── Upload your v2 adapter to Hugging Face Hub ──
ft_model.push_to_hub(HF_REPO_ID, private=True)
ft_tokenizer.push_to_hub(HF_REPO_ID, private=True)

print(f"\n✅ Adapter uploaded to https://huggingface.co/{HF_REPO_ID}")
print(f"\nTo use it later:")
print(f'  base_model = AutoModelForCausalLM.from_pretrained("{MODEL_ID}")')
print(f'  ft_model = PeftModel.from_pretrained(base_model, "{HF_REPO_ID}")')
```

## Cell 28 — Markdown

## Step 13: Save Results & What You Learned

### v1 vs v2 Summary

| | v1 (ChatDoctor) | v2 (Reformatted WikiDoc) |
|---|---|---|
| **Model** | Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct (same) |
| **Dataset** | ChatDoctor (noisy chat logs) | WikiDoc reformatted via GPT-4o-mini |
| **Learning rate** | 2e-4 (aggressive) | 5e-5 (gentle) |
| **Epochs** | 3 | 2 |
| **Result** | ❌ Degraded — persona contamination | ✅ Improved — safety added, quality preserved |
| **Key lesson** | Don't fine-tune on noisy data | Clean data + gentle training = success |

### The Big Picture

Fine-tuning is a powerful tool, but it requires:
1. **Clean, well-formatted training data** — garbage in = garbage out
2. **Appropriate hyperparameters** — don’t overwrite what the model already knows
3. **Clear evaluation** — always compare before vs after on fixed benchmarks

### An Honest Assessment: Could Prompting Have Done This?

**Yes, mostly.** The Qwen 1.5B model already knows medicine — it was pre-trained on medical textbooks, clinical guidelines, and research papers. For general medical Q&A like our 10 benchmark questions, a well-crafted system prompt ("always include safety disclaimers, structure your answers with bullet points") could achieve similar results without any fine-tuning at all.

So why did we fine-tune? **To teach you the mechanics.** The marginal improvement from just 2,000 clean WikiDoc rows — especially the safety score boost — shows that fine-tuning *works*, but on a domain the model already knows, the gains are small.

**Where fine-tuning truly shines** is on domains the model has *never seen*:
- **Proprietary clinical protocols** — your hospital’s specific triage rules, not in any public dataset
- **Internal drug formulary** — which medications your organization approves for which conditions
- **Rare disease data** — conditions with so few published cases that the base model gives generic or wrong answers
- **Regulatory compliance language** — exact phrasing required by FDA/EMA for patient-facing content

In those cases, prompting alone cannot help — the knowledge simply isn’t in the model’s weights. That’s when the QLoRA pipeline you just learned becomes essential.


## Cell 29 — Code

```python
# ── Save benchmark results for Module 4 (LangSmith evaluation) ──
results = {
    "model_id": MODEL_ID,
    "adapter_repo": HF_REPO_ID,
    "dataset": DATASET_ID,
    "version": "v2-wikidoc-1.5B",
    "benchmark_prompts": BENCHMARK_PROMPTS,
    "base_outputs": base_outputs,
    "finetuned_outputs": finetuned_outputs,
}

with open("benchmark_results_v2.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved benchmark_results_v2.json")
print("\n📝 What you learned in this notebook:")
print("   1. Clean, reformatted data produces better fine-tuning results")
print("   2. Gentler hyperparameters prevent catastrophic forgetting")
print("   3. Safety behaviors can be baked in through data quality")
print("   4. Same model + different data = completely different outcomes")
print("\n👉 Next: Module 3 (deploy to HF Hub) → Module 4 (evaluate with LangSmith)")
```

## Cell 30 — Markdown

## 📚 Understanding Your Results: When Data Quality Wins

### What Just Happened?

You fine-tuned the **exact same model** as v1 — but this time the results were better. The only difference was data quality.

### v1 vs v2: The Data Quality Lesson

```
v1 — ChatDoctor (noisy data):
  Dataset: Raw doctor-patient chat logs
  Problem: Every answer starts with "Hello, welcome to Chat Doctor"
  Result:  Model picked up persona artifacts, lost answer structure
  Lesson:  Noisy data teaches bad habits

v2 — Reformatted WikiDoc (clean data):
  Dataset: WikiDoc reformatted via GPT-4o-mini into conversational Q&A
  Quality: 200-400 word structured answers with safety disclaimers
  Result:  Safety improved, answer quality preserved
  Lesson:  Clean data preserves model strengths
```

### When Should You Fine-Tune?

```
Fine-tuning is worth it when:
  ✅ You need to bake in specific behaviors (safety disclaimers, formats)
  ✅ You have proprietary data the model has never seen
  ✅ You need consistent domain-specific terminology
  ✅ You have clean, high-quality training data

Consider alternatives when:
  ⚠️ The model already handles your task well (try prompting first)
  ⚠️ Your training data is noisy (clean it or use RAG instead)
  ⚠️ You need up-to-date information (use RAG — it's always current)
  ⚠️ You can't verify the quality of fine-tuned outputs
```


### The Honest Truth About This Workshop

The model already knew medicine. Our v2 improvement was marginal — safety went up because we baked disclaimers into the data, but accuracy and helpfulness barely moved. A good system prompt could achieve similar results for general medical Q&A.

**That’s the point.** This workshop teaches you the *mechanics* of fine-tuning — data prep, QLoRA, hyperparameters, evaluation — so that when you encounter a domain the model genuinely doesn’t know (proprietary protocols, internal data, rare conditions), you have the skills to do it right. The 2,000-row WikiDoc experiment is your training ground.

### Next Steps

- **Module 3:** Deploy your v2 adapter from HF Hub and run inference
- **Module 4:** Evaluate both v1 and v2 outputs with LangSmith’s LLM-as-judge evaluators
