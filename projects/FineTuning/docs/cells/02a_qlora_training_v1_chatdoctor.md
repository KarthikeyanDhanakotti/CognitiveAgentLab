# MODULE 2a — 02a_qlora_training_v1_chatdoctor.ipynb

_Source: `02a_qlora_training_v1_chatdoctor.ipynb`_


## Cell 1 — Markdown

# Module 2 (v1): Fine-Tuning a 1.5B Model — Does It Actually Help?

## Welcome!

In this notebook, you're going to fine-tune your first language model. You'll take `Qwen2.5-1.5B-Instruct` — a model with 1.5 billion parameters that's already pretty smart — and try to make it better at answering medical questions.

**Here's the twist:** you'll discover that fine-tuning doesn't always make things better. Sometimes the model is already good enough, and fine-tuning actually makes it *worse*.

### What You'll Learn

> **Key takeaway: A model that already knows your domain doesn't need fine-tuning.**
> If you try anyway, you risk teaching it bad habits from your training data.

By the end of this notebook, you'll understand:
- How to load a pre-trained model and prepare it for fine-tuning
- How to set up LoRA adapters (a way to fine-tune cheaply)
- How to train on real medical Q&A data
- How to compare "before" vs "after" to measure whether fine-tuning helped
- **Why fine-tuning a capable model can actually backfire**

### What Will Happen (Spoiler)

| | Before Fine-Tuning (1.5B) | After Fine-Tuning |
|---|---|---|
| **Answer quality** | Detailed, well-structured, uses markdown | Loses structure, shorter answers |
| **Medical accuracy** | Generally accurate | Similar content but less precise |
| **Style** | Professional assistant tone | Picks up casual "ChatDoctor" persona |
| **Verdict** | ✅ Already good enough | ❌ Worse — learned bad habits from data |

Don't worry if "LoRA," "quantization," or "adapter" don't mean anything to you yet — each step explains everything from scratch.

### How This Connects to v2

After this notebook, you'll run the **v2 notebook** with the same model but cleaner data (reformatted WikiDoc). Together, these two notebooks teach you:
1. **v1 (this notebook):** Don't fine-tune when the model already knows the domain
2. **v2:** When fine-tuning helps, data quality is everything

### ⚙️ How to Run This Notebook

1. **Open in Google Colab** → File → Upload notebook
2. **Select GPU Runtime** → Runtime → Change runtime type → T4 GPU
3. **Run cells in order** from top to bottom (Shift+Enter each cell)
4. **You'll need:** A free [Hugging Face](https://huggingface.co/) account with a Write token


## Cell 2 — Markdown

## Step 1: Check Your GPU & Install Libraries

Before you can fine-tune anything, you need a **GPU** (Graphics Processing Unit). Training a language model on a regular CPU would take days — a GPU does it in minutes.

**What's happening here:**
- First cell: checks if you have a GPU and how much memory it has
- Second cell: installs the Python libraries you'll need

If you see "No devices found," go to Runtime → Change runtime type → T4 GPU.


## Cell 3 — Code

```python
# ── GPU Check ──
# Think of a GPU like a super-fast calculator that can do millions of math operations
# at once. Training a language model involves TONS of math (matrix multiplications),
# so we need one.
#
# On Google Colab, you get a free "Tesla T4" GPU with ~15 GB of VRAM (GPU memory).
# That's more than enough for what we're doing.

!nvidia-smi   # This shows your GPU info (like "Task Manager" but for the GPU)

import torch  # PyTorch — the library that talks to the GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")  # CUDA = NVIDIA's GPU programming system
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    # VRAM = how much memory your GPU has. More VRAM = bigger models can fit.
    print(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Cell 4 — Code

```python
# ── Install the libraries you need ──
# Each library has a specific role:
#   transformers  → Load & run the language model (the AI's "brain")
#   accelerate    → Makes GPU training faster and simpler
#   peft          → LoRA adapter support (tiny trainable layers on top of a frozen model)
#   trl           → SFTTrainer: the training engine for instruction fine-tuning
#   bitsandbytes  → 4-bit quantisation so a 1.5B model fits in ~2 GB VRAM
#   datasets      → Load the medical Q&A dataset from Hugging Face
#   huggingface-hub → Push/pull models to your HF account
#   pandas        → Data wrangling (used when we inspect the dataset)
#
# IMPORTANT: We do NOT install/upgrade torch. RunPod and Colab ship with a
# CUDA-enabled PyTorch pre-installed. Installing torch from PyPI would replace
# it with a CPU-only build, breaking GPU support entirely.
# IMPORTANT: Quotes around package specs (e.g. "transformers>=4.45.0") are
# required on RunPod/Linux to prevent the shell from misinterpreting >=.
# NOTE: We pin transformers<4.47.0 because newer versions require torch 2.6+,
# but RunPod ships with torch 2.4.x pre-installed.
!pip install -q \
    "transformers>=4.45.0,<4.47.0" \
    "accelerate>=0.34.0" \
    "peft>=0.12.0" \
    "trl>=0.10.0" \
    "bitsandbytes>=0.43.0" \
    "datasets>=2.20.0" \
    "huggingface-hub>=0.24.0" \
    pandas

# Make newly installed packages visible without restarting the kernel
import importlib
importlib.invalidate_caches()
print("✅ All packages installed!")
```

## Cell 5 — Markdown

## Step 2: Import Libraries & Set Your Configuration

Now you'll import everything and set up the "knobs and dials" for your experiment.

**What's a configuration?** It's all the settings in one place — which model to use, how much data to train on, etc. Having them at the top means you can easily tweak things without hunting through code.

**Analogy:** Think of this like setting up the oven before baking. You decide the temperature (learning rate), how long to bake (epochs), and what recipe to follow (dataset) — all before you start.


## Cell 6 — Code

```python
import os
import re
import json
import torch
import pandas as pd

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,    # Loads the actual language model (the "brain")
    AutoTokenizer,           # Loads the tokenizer (converts text → numbers the model understands)
    BitsAndBytesConfig,      # Settings for 4-bit compression (to fit on free GPUs)
)
from peft import (
    LoraConfig,              # Settings for LoRA adapters (the "tiny trainable layers")
    get_peft_model,          # Attaches LoRA adapters to the model
    prepare_model_for_kbit_training,  # Prepares a 4-bit model for training
    PeftModel,               # Loads a saved adapter back onto a base model
)
from trl import SFTTrainer, SFTConfig  # The training engine

# ══════════════════════════════════════════════════════════
#  YOUR CONFIGURATION — Change these if you want to experiment!
# ══════════════════════════════════════════════════════════

# Which model to fine-tune.
# Qwen2.5-1.5B-Instruct has 1.5 billion parameters — it's already quite capable.
# This is the "noisy data" experiment. In v2, you'll try the same model with clean reformatted WikiDoc data.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Which dataset to train on.
# ChatDoctor-HealthCareMagic-100k = ~112,000 real patient questions + doctor answers.
# Same dataset in both v1 and v2, so the ONLY difference is model size.
DATASET_ID = "lavita/ChatDoctor-HealthCareMagic-100k"

# Where to save the trained adapter (locally on this machine)
OUTPUT_DIR = "./healthcare-assistant-lora"

# Where to upload it on Hugging Face (so you can use it from anywhere later)
HF_REPO_ID = "jeev1992/healthcare-assistant-lora"

# How many training examples to use (out of ~112,000 available).
# Same as v2 — keeps the comparison fair (only model size differs).
NUM_TRAIN_SAMPLES = 2000

# How many examples to hold out for checking if the model is overfitting.
# (Overfitting = memorizing answers instead of learning patterns. More on this later.)
NUM_EVAL_SAMPLES = 100

# Maximum length per training example (in tokens, not words).
# 512 tokens ≈ 400 words — enough for most medical Q&A pairs.
MAX_SEQ_LENGTH = 512

# The "personality" we give the model. This is injected at the start of every conversation.
# Both v1 and v2 use the exact same prompt, so the comparison is fair.
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
# Hugging Face Hub is like GitHub, but for AI models instead of code.
# At the end of this notebook, you'll upload your trained adapter there.
#
# You need a "Write" token (like a password) to upload.
# Get one for free at: https://huggingface.co/settings/tokens
#   → Click "New token" → Name it anything → Select "Write" → Copy it
#
# notebook_login() will show a text box — paste your token there.
from huggingface_hub import notebook_login
notebook_login()
```

## Cell 8 — Markdown

## Step 3: Load & Clean the Training Data

Now you'll download the training data — real patient questions and doctor answers from the `ChatDoctor-HealthCareMagic-100k` dataset.

**But there's a problem with this data.** The doctors' answers are full of greetings like *"Hello, welcome to Chat Doctor"* and sign-offs like *"Hope this helps, take care!"* If you train on this raw data, the model will learn to say those things too — which is not useful medical knowledge.

**So you'll clean it first** using regex patterns that strip out the boilerplate. This is an important lesson: **your model is only as good as the data you train it on.** Garbage in → garbage out.

> **What is "chat format"?** Language models expect conversations structured as a list of messages, each with a `role` (system, user, or assistant) and `content`. This tells the model who said what.


## Cell 9 — Code

```python
# ── Load the dataset from Hugging Face ──
# This downloads ~112,000 real patient-doctor conversations.
# Think of Hugging Face datasets like an app store for training data.
raw_ds = load_dataset(DATASET_ID, split="train")
print(f"Raw dataset size: {len(raw_ds)}")
print(f"Columns: {raw_ds.column_names}")
print(f"\nHere's what one example looks like (raw, before cleaning):")
print(f"  Patient asked:  {raw_ds[0]['input'][:100]}")
print(f"  Doctor replied: {raw_ds[0]['output'][:200]}")


# ── Boilerplate patterns to REMOVE from doctor answers ──
# These are regex (regular expression) patterns that match common ChatDoctor greetings
# and sign-offs. We want the medical CONTENT, not the "Hello, welcome to Chat Doctor" noise.
#
# Example of what gets stripped:
#   BEFORE: "Hello and welcome to Chat Doctor. Diabetes is characterized by..."
#   AFTER:  "Diabetes is characterized by..."
BOILERPLATE_PATTERNS = [
    r"(?i)^(hi|hello|hey|dear)\b.*?(chat\s*doctor|health\s*care\s*magic|healthcaremagic).*?[\.\!\n]",
    r"(?i)(thanks?\s+for\s+(your|the)\s+(query|question|concern)).*?[\.\!\n]",
    r"(?i)(welcome\s+to\s+(chat\s*doctor|health\s*care\s*magic|healthcaremagic)).*?[\.\!\n]",
    r"(?i)(i\s+understand\s+your\s+concern).*?[\.\n]",
    r"(?i)(hope\s+(this|i)\s+(helps?|answered|clarified|have answered)).*$",
    r"(?i)(wish\s+you\s+(good\s+health|all\s+the\s+best|a\s+speedy)).*$",
    r"(?i)(take\s+care|get\s+well\s+soon|god\s+bless).*$",
    r"(?i)(feel\s+free\s+to\s+(ask|contact|reach|write)).*$",
    r"(?i)(regards|best wishes|warm regards|with regards).*$",
    r"(?i)^(dear\s+(patient|sir|madam|friend)).*?[\.\,\!\n]",
]


def clean_doctor_response(text):
    """Remove ChatDoctor greetings and sign-offs from a doctor's answer."""
    cleaned = text.strip()
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)
    # Remove extra blank lines left behind after stripping
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def format_to_chat(example):
    """
    Convert one dataset row into the "chat messages" format that the trainer expects.

    What this does:
      1. Takes the patient's question and the doctor's answer
      2. Cleans the doctor's answer (strips boilerplate)
      3. Skips examples that are too short (likely garbage)
      4. Returns a list of messages: [system prompt, user question, assistant answer]
    """
    patient_question = example["input"].strip()
    doctor_answer = clean_doctor_response(example["output"])

    # Skip junk: very short questions or answers are usually low quality
    if len(patient_question) < 10 or len(doctor_answer) < 50:
        return {"messages": None}

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},     # The model's personality
            {"role": "user",      "content": patient_question},   # What the patient asked
            {"role": "assistant", "content": doctor_answer},      # What the model should learn to say
        ]
    }


# ── Apply cleaning, filter bad examples, and split into train/eval ──
formatted = raw_ds.shuffle(seed=42).map(format_to_chat, remove_columns=raw_ds.column_names)
formatted = formatted.filter(lambda x: x["messages"] is not None)
formatted = formatted.select(range(min(NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES, len(formatted))))

# Train set = what the model learns from
# Eval set = held-out examples to check if the model is memorizing vs learning
train_ds = formatted.select(range(NUM_TRAIN_SAMPLES))
eval_ds  = formatted.select(range(NUM_TRAIN_SAMPLES, NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES))

print(f"\nTrain: {len(train_ds)}  |  Eval: {len(eval_ds)}")
print(f"\nHere's what a CLEANED example looks like:")
for msg in train_ds[0]["messages"]:
    role = msg["role"]
    text = msg["content"][:200]
    print(f"  [{role}] {text}...")
```

## Cell 10 — Markdown

## Step 4: Set Up Your "Test Exam" (Benchmark Prompts)

Before you start training, you need a way to measure whether fine-tuning helped or hurt. You'll do this by asking the model **the same 10 medical questions** before and after training, then comparing the answers.

**Why the same questions?** If you asked different questions before and after, you couldn't tell whether the answers improved because of training, or just because the new questions were easier.

**Important:** These questions are NOT in the training data. You want to test whether the model *learned general medical knowledge*, not whether it *memorized specific answers*. This is called **generalization** — the ability to handle questions it's never seen before.


## Cell 11 — Code

```python
# ── Your 10 "Test Exam" Questions ──
# These cover different areas of medicine:
#   - Symptoms, diagnosis, treatments, drug side effects, prevention
# You'll run these BEFORE training and AFTER training to see what changed.
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

Now you'll download the actual 1.5B-parameter model from Hugging Face and load it onto your GPU.

**The problem:** The 1.5B model normally needs ~3 GB of GPU memory. That's fine for a T4, but we also need room for training later. So we **compress** (quantize) it to 4-bit, which cuts the memory to ~1.5 GB.

**What is quantization?** Normally, each number in the model is stored as a 16-bit decimal (like 0.3748291). 4-bit quantization rounds these to one of 16 possible values. You lose a tiny bit of precision, but the model's quality barely changes — and it uses 4× less memory.

**Analogy:** It's like compressing a photo from 4K to 1080p. You lose some detail, but for most purposes it looks the same — and the file is way smaller.


## Cell 13 — Code

```python
# ── Quantization settings ──
# This tells the model HOW to compress itself when loading.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # Compress weights from 16-bit to 4-bit
    bnb_4bit_quant_type="nf4",       # "NF4" = a compression format designed for neural networks
                                     # (better than naive 4-bit because it accounts for how
                                     #  neural network weights are typically distributed)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Do actual math in bfloat16 (fast + stable)
    bnb_4bit_use_double_quant=True,  # Also compress the compression metadata (saves ~0.1 GB extra)
)

# ── Load the model ──
# This downloads ~1.5 GB from Hugging Face and loads it onto your GPU.
# device_map="auto" means: "figure out which device (GPU/CPU) to use automatically."
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,                        # "Qwen/Qwen2.5-1.5B-Instruct"
    quantization_config=bnb_config,  # Use the 4-bit compression settings above
    device_map="auto",               # Automatically place on GPU
    trust_remote_code=True,          # Allow running Qwen's custom code
)

# ── Load the tokenizer ──
# The tokenizer converts text into numbers (token IDs) that the model understands.
# Example: "diabetes" → [75402]   "What symptoms?" → [1841, 13809, 30]
# Every model has its own tokenizer — they're paired together.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token   # Needed for batching during training
tokenizer.padding_side = "right"            # Pad on the right side (standard for text generation)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {MODEL_ID}")
print(f"Total params:  {total_params / 1e6:.1f}M")  # ~1,500M = 1.5 billion parameters
print(f"VRAM used:     {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Cell 14 — Markdown

## Step 6: BEFORE — Test the Base Model (No Fine-Tuning Yet)

This is your **"before" snapshot.** You'll ask the model all 10 benchmark questions *right now*, before any training happens.

**Pay close attention to the quality of these responses.** The 1.5B model has been pre-trained on trillions of tokens — medical textbooks, Wikipedia, research papers, clinical guidelines. It already knows a lot about medicine.

After training, you'll compare the "after" responses to these "before" ones. The big question: **did fine-tuning make things better or worse?**


## Cell 15 — Code

```python
def generate_response(model, tokenizer, user_prompt, max_new_tokens=256):
    """
    Ask the model a question and get its answer.

    How it works, step by step:
      1. Build a conversation: system prompt + your question
      2. Convert the conversation into the special format the model expects
         (each model has its own format with special tokens like <|im_start|>)
      3. Convert text → numbers (tokenize) and send to GPU
      4. Let the model generate new tokens one by one (its answer)
      5. Convert the generated numbers back to text (decode)
    """
    # Build the conversation structure
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},   # The model's personality
        {"role": "user",    "content": user_prompt},       # Your question
    ]
    # apply_chat_template converts messages into the model's expected format
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Tokenize (text → numbers) and move to GPU
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # torch.no_grad() = "don't track gradients" — saves memory during inference
    # (Gradients are only needed during training, not when asking questions)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Maximum length of the answer (~190 words)
            do_sample=True,        # Allow some randomness (not always the most likely word)
            temperature=0.7,       # Controls randomness: 0=always same answer, 1=very random
            top_p=0.9,             # Only consider the top 90% most likely words
            repetition_penalty=1.2,# Penalize repeating the same words/phrases
        )
    # Decode only the NEW tokens (skip the question part, we only want the answer)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Run the "BEFORE" test ──
# These are the base model's answers before ANY fine-tuning.
# Read through them — notice how detailed and well-structured they already are.
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

Here's where the magic happens. Instead of retraining all 1.5 billion parameters (which would need a massive GPU and take hours), you'll use **LoRA** (Low-Rank Adaptation) to train just a tiny fraction of them.

**How LoRA works — a simple analogy:**

Imagine the base model is a finished painting. Full fine-tuning would be like painting over the entire canvas from scratch. LoRA is like adding a thin transparent overlay on top — you make small adjustments without changing the original painting underneath.

**In practice:** LoRA adds small "adapter" matrices to each layer of the model. During training, only these adapters get updated (~1% of all parameters). The original model stays frozen and unchanged.

| Setting | Value | What It Means |
|---------|-------|---------------|
| `r=16` | Rank 16 | The "width" of the adapter. Higher = more capacity to learn, but slower |
| `lora_alpha=32` | Alpha 32 | Controls how much the adapter's output matters (alpha/r = 2.0 scaling) |
| `lora_dropout=0.05` | 5% dropout | Randomly ignores 5% of adapter outputs during training to prevent memorization |
| `target_modules="all-linear"` | All layers | Adds adapters to every linear layer in the model (maximum learning capacity) |


## Cell 17 — Code

```python
# ── LoRA Configuration ──
# Same settings as the v2 notebook — we want the ONLY difference to be model size.
# That way, when we compare results, we know any differences come from the model, not the config.
lora_config = LoraConfig(
    r=16,                        # Rank: the "width" of adapter matrices. 16 is a solid default.
    lora_alpha=32,               # Scaling factor. Rule of thumb: alpha = 2 × r.
    lora_dropout=0.05,           # 5% dropout to prevent overfitting (memorizing training data).
    target_modules="all-linear", # Add adapters to ALL linear layers (most thorough).
    task_type="CAUSAL_LM",       # We're doing causal language modeling (predict next word).
)

# With the 1.5B model, this gives us ~14M trainable parameters out of ~1,500M total.
# That's less than 1% — but enough to shift the model's behavior significantly.
print(f"LoRA config ready: r={lora_config.r}, alpha={lora_config.lora_alpha}")
print("The adapters will be attached automatically when training starts.")
```

## Cell 18 — Markdown

## Step 8: Train! 🚀

This is the big moment — you're about to train the model.

**What happens during training:**
1. The model reads a batch of training examples (patient question + doctor answer)
2. It tries to predict the doctor's answer, word by word
3. It measures how wrong it was (**training loss** — lower is better)
4. It adjusts the LoRA adapter weights slightly to be less wrong next time
5. Repeat for all examples, 3 times through the entire dataset (**3 epochs**)

**Key numbers to watch:**
- **Training loss** should decrease over time (the model is learning)
- **Eval loss** is measured on data the model hasn't seen — if this goes UP while training loss goes DOWN, the model is memorizing instead of learning (called **overfitting**)

**How long?** About 20-30 minutes on a T4 GPU.

> **Math behind the steps:** 2,000 examples × 3 epochs = 6,000 total views. With an effective batch size of 8, that's 6,000 ÷ 8 = **750 training steps**.


## Cell 19 — Code

```python
# ── Tell the tokenizer the max sequence length ──
# This limits how long each training example can be (in tokens).
# 512 tokens ≈ 400 words, which covers most medical Q&A pairs.
tokenizer.model_max_length = MAX_SEQ_LENGTH

# ── Training settings ──
# These are the "knobs and dials" that control how training works.
# Don't worry about memorizing all of these — the comments explain each one.
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,                   # Where to save checkpoints during training
    num_train_epochs=3,                      # How many times to go through ALL training data
                                             # (3 is a good default — not too little, not too much)
    per_device_train_batch_size=2,           # Process 2 examples at a time on the GPU
    gradient_accumulation_steps=4,           # Accumulate gradients for 4 batches before updating
                                             # → Effective batch size = 2 × 4 = 8 examples
                                             # (This is a trick to simulate a larger batch size
                                             #  without needing more GPU memory)
    learning_rate=2e-4,                      # How big each weight update is (0.0002)
                                             # Too high → model overshoots and oscillates
                                             # Too low → model barely learns anything
                                             # 2e-4 is the standard for LoRA fine-tuning
    lr_scheduler_type="cosine",              # Start with full learning rate, gradually slow down
                                             # (like slowing down as you approach a parking spot)
    warmup_steps=75,                         # For the first 75 steps (~10%), gently ramp up
                                             # the learning rate instead of starting at full speed
    logging_steps=10,                        # Print training loss every 10 steps
    save_strategy="steps",                   # Save checkpoints periodically (not just at the end)
    save_steps=100,                          # Save a checkpoint every 100 steps (insurance!)
    eval_strategy="steps",                   # Check eval loss periodically
    eval_steps=100,                          # Check every 100 steps if the model is overfitting
    bf16=True,                               # Use bfloat16 math (fast + stable on modern GPUs)
    report_to="none",                        # Don't send metrics to external services
    seed=42,                                 # Random seed for reproducibility (same results every run)
)

# ── Create the Trainer ──
# SFTTrainer handles everything: tokenization, batching, training loop, saving.
# You just give it the model, data, and settings — and it does the work.
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,    # The tokenizer (converts text → numbers for the model)
    train_dataset=train_ds,        # Your 2,000 training examples
    eval_dataset=eval_ds,          # Your 100 held-out examples for overfitting detection
    args=training_args,            # All the settings above
    peft_config=lora_config,       # LoRA adapter settings — SFTTrainer attaches them for you
)

# Show how many parameters are actually being trained
trainable, total = trainer.model.get_nb_trainable_parameters()
print(f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M  ({100 * trainable / total:.2f}%)")
print(f"  → Only {100 * trainable / total:.2f}% of the model's weights will change!")

# ── START TRAINING ──
print("\nStarting training... (this takes about 20-30 minutes on a T4)")
print(f"Total steps: ~{NUM_TRAIN_SAMPLES * 3 // 8}")
train_result = trainer.train()

# Training is done! Let's see how it went.
print(f"\n✅ Training complete!")
print(f"  Training loss:    {train_result.training_loss:.4f}")
print(f"  Training runtime: {train_result.metrics['train_runtime']:.0f}s")
```

## Cell 20 — Markdown

## Step 9: Save Your Trained Adapter

Training is done! Now you'll save the result.

**Important:** You're NOT saving the entire 1.5B model. You're only saving the tiny LoRA adapter (~15-30 MB). The adapter is useless on its own — it needs the original base model to work. Think of it like a "patch" that modifies the base model's behavior.

This keeps things small: a 30 MB adapter vs a 3,000 MB full model.


## Cell 21 — Code

```python
# ── Save the LoRA adapter and tokenizer to disk ──
# This creates a folder with your trained adapter files:
#   adapter_model.safetensors  — the actual trained weights (~15-30 MB)
#   adapter_config.json        — the LoRA settings (r, alpha, etc.)
#   tokenizer files            — so inference uses the same tokenizer
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR} ✓")

# Let's see how small the adapter actually is
import pathlib
adapter_size = sum(f.stat().st_size for f in pathlib.Path(OUTPUT_DIR).rglob("*") if f.is_file())
print(f"Adapter size: {adapter_size / 1e6:.1f} MB  (vs ~3,000 MB for the full model)")
```

## Cell 22 — Markdown

## Step 10: AFTER — Test the Fine-Tuned Model

Now for the moment of truth! You'll load a **fresh copy** of the base model, attach your trained adapter on top, and ask the same 10 questions.

**Why load a fresh model?** During training, the model object in memory gets modified. Loading a clean base model + your saved adapter gives you a trustworthy "after" measurement.

**What to look for when comparing:**
- Did the answers get more detailed, or less?
- Did the model pick up the ChatDoctor persona ("Hello, welcome to Chat Doctor")?
- Is the medical information more or less accurate?


## Cell 23 — Code

```python
# ── Free GPU memory from training (if training was run) ──
if 'model' in dir():   del model
if 'trainer' in dir():  del trainer
torch.cuda.empty_cache()   # Tell the GPU to release unused memory

# ── Load a FRESH base model + your trained adapter ──
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,  # Same 4-bit compression as before
    device_map="auto",
    trust_remote_code=True,
)
# PeftModel.from_pretrained() = "take this base model and add the adapter on top"
ft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
ft_model.eval()  # Put in "evaluation mode" (disables dropout, etc.)

ft_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
ft_tokenizer.pad_token = ft_tokenizer.eos_token

print("Fine-tuned 1.5B model loaded ✓")

# ── Run the "AFTER" test — same 10 questions ──
# Compare these answers to the BEFORE ones above.
# Look for: Did quality improve? Did ChatDoctor persona leak through?
print("=" * 60)
print("AFTER FINE-TUNING — Fine-Tuned 1.5B Model Answers")
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

## Step 11: Side-by-Side Comparison — Did Fine-Tuning Help?

Here's the comparison table. Read through it carefully.

**What you'll likely see:**
- The **base model** gives detailed, well-structured answers with bullet points and proper formatting
- The **fine-tuned model** gives shorter answers and starts saying things like "Hello, welcome to Chat Doctor"

**Why?** The base model was trained on trillions of tokens of medical content. It already *knew* medicine. Fine-tuning on 2,000 ChatDoctor examples didn't teach it new medical facts — it just taught it to talk like the ChatDoctor website. That's persona contamination, and it's a real risk of fine-tuning.


## Cell 25 — Code

```python
# ── Build the side-by-side comparison table ──
# As you read through, look for these patterns:
#   ❌ ChatDoctor persona: "Hello", "Chat Doctor", "Hope this helps"
#   ❌ Lost structure: base model used bullet points, fine-tuned doesn't
#   ❌ Shorter answers: fine-tuned model gives less detail
comparison_rows = []
for base, ft in zip(base_outputs, finetuned_outputs):
    comparison_rows.append({
        "Prompt":          base["prompt"],
        "Base Model (1.5B)":  base["response"][:200],
        "Fine-Tuned (1.5B)":  ft["response"][:200],
    })

df = pd.DataFrame(comparison_rows)
pd.set_option("display.max_colwidth", 200)
df
```

## Cell 26 — Markdown

## Step 12: Upload to Hugging Face Hub

Now you'll upload your adapter to Hugging Face Hub so you can use it from anywhere — your laptop, a server, or another Colab notebook.

**What gets uploaded:**
- `adapter_model.safetensors` — your trained LoRA weights (~15-30 MB)
- `adapter_config.json` — the settings you used (r, alpha, etc.)
- Tokenizer files

**What does NOT get uploaded:** The base model itself. Anyone who wants to use your adapter needs to download the base model separately (it's publicly available on Hugging Face).


## Cell 27 — Code

```python
# ── Upload your adapter to Hugging Face Hub ──
# private=True means only YOU can see this repo.
# You can make it public later if you want to share it.
ft_model.push_to_hub(HF_REPO_ID, private=True)
ft_tokenizer.push_to_hub(HF_REPO_ID, private=True)

print(f"\n✅ Adapter uploaded to https://huggingface.co/{HF_REPO_ID}")
print(f"\nTo use it later from any machine:")
print(f'  base_model = AutoModelForCausalLM.from_pretrained("{MODEL_ID}")')
print(f'  ft_model = PeftModel.from_pretrained(base_model, "{HF_REPO_ID}")')
```

## Cell 28 — Markdown

## Step 13: Save Results & What You Learned

You'll save the before/after results as a JSON file. This file will be used in Module 4 for automated evaluation with LangSmith.

### What You Just Discovered

The 1.5B model was **already good enough** for medical Q&A. Fine-tuning on ChatDoctor data didn't teach it new medical knowledge — it just taught it to mimic the ChatDoctor website's casual style. The lesson: **don't fine-tune a model that already handles your task well.**

Next, you'll run the **v2 notebook** with the same 1.5B model but trained on reformatted WikiDoc data. You'll see how clean, well-formatted data can preserve model strengths while adding safety behaviors.


## Cell 29 — Code

```python
# ── Save your before/after results ──
# This JSON file contains every benchmark prompt, the base model's answer,
# and the fine-tuned model's answer. Module 4 will use this for evaluation.
results = {
    "model_id": MODEL_ID,
    "adapter_repo": HF_REPO_ID,
    "dataset": DATASET_ID,
    "version": "v1-chatdoctor-1.5B",
    "benchmark_prompts": BENCHMARK_PROMPTS,
    "base_outputs": base_outputs,
    "finetuned_outputs": finetuned_outputs,
}

with open("benchmark_results_v1.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved benchmark_results_v1.json")
print("\n📝 What you learned in this notebook:")
print("   1. How to load a model in 4-bit quantization (saves GPU memory)")
print("   2. How to attach LoRA adapters (train only ~1% of weights)")
print("   3. How to train with SFTTrainer (the easy way to fine-tune)")
print("   4. How to compare before vs after with benchmark prompts")
print("   5. That fine-tuning a capable model can make it WORSE")
print("\n👉 Next: Run the v2 notebook (same model + reformatted WikiDoc) to see when fine-tuning DOES help.")
```

## Cell 30 — Markdown

## 📚 Understanding Your Results: When Fine-Tuning Hurts

### What Just Happened?

You fine-tuned a 1.5B model and it got *worse*. That's not a bug — it's an important lesson. Let's understand why.

### Why Did Fine-Tuning Make Things Worse?

Think of the 1.5B model like a doctor who went to medical school for years (pre-trained on trillions of tokens). Now imagine sending that doctor to shadow at a clinic that has sloppy notes — greetings like "Hello welcome to Chat Doctor" in every medical record. The doctor doesn't learn new medicine. They just pick up the bad note-taking habits.

That's exactly what happened:

```
What the model ALREADY knew (from pre-training):
  ✅ Detailed explanations of diabetes, hypertension, asthma
  ✅ Proper medical terminology
  ✅ Well-structured answers with bullet points

What it "learned" from our 2,000 ChatDoctor examples:
  ❌ To greet patients: "Hello, welcome to Chat Doctor"
  ❌ To sign off: "Hope this helps, take care"
  ❌ A more casual, less structured writing style
```

### The Decision Flowchart: Should YOU Fine-Tune?

Before fine-tuning any model, ask yourself this:

```
Does the base model already answer your questions well?
    │
    ├── YES → STOP. Don't fine-tune.
    │         Try these instead:
    │           • Better system prompts (prompt engineering)
    │           • RAG — give the model relevant documents at query time
    │           • Few-shot examples in the prompt
    │
    └── NO → Fine-tuning might help!
              But first ask: Is my training data clean?
                │
                ├── YES → Go ahead and fine-tune
                │
                └── NO → Clean your data first, or use RAG instead
```

### What is RAG? (A Common Alternative to Fine-Tuning)

**RAG = Retrieval-Augmented Generation.** Instead of baking knowledge INTO the model through training, you GIVE the model relevant documents when someone asks a question.

**Example:**

```
Without RAG:
  User: "What's the latest treatment for Type 2 diabetes?"
  Model: (answers from its training data, which might be outdated)

With RAG:
  User: "What's the latest treatment for Type 2 diabetes?"
  System: [searches your medical database, finds 2024 guidelines]
  Model: [reads the guidelines] "According to the 2024 ADA guidelines..."
```

**Why RAG is often better than fine-tuning:**

| | Fine-Tuning | RAG |
|---|---|---|
| **Knowledge** | Frozen at training time | Always up-to-date (just update your docs) |
| **Cost** | Needs GPU + training time | No training needed |
| **Risk** | Can degrade the model (like we just saw!) | Model stays unchanged |
| **Sources** | Can't cite where info came from | Can show which document it used |
| **Best for** | Teaching the model a new *style* or *domain* | Giving the model new *facts* |

### What's Next?

In the **v2 notebook**, you'll fine-tune the same 1.5B model on reformatted WikiDoc data (cleaned via GPT-4o-mini). You'll see how data quality is everything — clean data preserves the model's strengths while adding safety disclaimers.
