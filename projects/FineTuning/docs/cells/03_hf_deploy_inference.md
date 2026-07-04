# MODULE 3 — 03_hf_deploy_inference.ipynb

_Source: `03_hf_deploy_inference.ipynb`_


## Cell 1 — Markdown

# Module 3: Hugging Face Deployment & Inference

**What we're doing:**
- Load the fine-tuned adapter from Hugging Face Hub (pushed in Module 2)
- Run benchmark prompts through the deployed model
- Compare base vs fine-tuned outputs again from a clean environment
- Demonstrate the Hugging Face Inference API as an alternative

**Prerequisites:** Completed Module 2 (adapter pushed to HF Hub)

---

### ⚙️ How to Run This Notebook

1. **Open in Google Colab** (recommended — needs a GPU)
   - Go to [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook**
   - Set **Runtime → Change runtime type → T4 GPU**

2. **Set Your Hugging Face Repo ID**
   - In the **Configuration** cell below, change `HF_REPO_ID` from `"jeev1992/healthcare-assistant-lora"` to the repo you pushed in Module 2 (e.g. `"jeev1992/healthcare-assistant-lora"`)

3. **Run All Cells Top to Bottom**
   - Click **Runtime → Run all**, or `Shift+Enter` cell by cell
   - Takes **~5-10 minutes** (model download + inference)


## Cell 2 — Markdown

## 1. Install & Import


## Cell 3 — Code

```python
# ── Install Dependencies ──
# Lighter than the training notebooks — we don't need trl or datasets here.
#   transformers + accelerate: load and run the model
#   peft: attach LoRA adapter to base model
#   bitsandbytes: 4-bit quantization (same as training)
#   huggingface-hub: download adapter from HF Hub + Inference API client
#   pandas: comparison tables
!pip install -q transformers accelerate peft bitsandbytes huggingface-hub pandas

import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel                     # Load saved LoRA adapter onto base model
from huggingface_hub import InferenceClient    # Optional: call HF Inference API endpoints
```

## Cell 4 — Code

```python
# ── Log in to Hugging Face ──
# You need this to download your private adapter from HF Hub.
# Get a Read token at: https://huggingface.co/settings/tokens
from huggingface_hub import notebook_login
notebook_login()
```

## Cell 5 — Markdown

## 2. Configuration

Set your HF repo ID (same one you pushed in Module 2) and the benchmark prompts.


## Cell 6 — Code

```python
# ── Configuration ──
# MODEL_ID: the base model that was fine-tuned in Module 2.
# HF_REPO_ID: where you pushed the LoRA adapter in Module 2.
MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
HF_REPO_ID = "jeev1992/healthcare-assistant-lora-v2"   # ← your adapter repo from Module 2

SYSTEM_PROMPT = (
    "You are a helpful healthcare assistant. "
    "Provide accurate, evidence-based medical information. "
    "Always recommend consulting a healthcare professional for serious concerns."
)

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

print(f"Will load adapter from: {HF_REPO_ID}")
```

## Cell 7 — Markdown

## 3. Option A — Local Inference (Load from Hub)

Load the base model + LoRA adapter directly from Hub. This works on any
machine with enough VRAM (Colab T4 is fine).


## Cell 8 — Code

```python
# ── Load base model (4-bit quantized) ──
# Same quantization config as Module 2 training notebook.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

ft_model = PeftModel.from_pretrained(base_model, HF_REPO_ID)
ft_model.eval()
print(f"Fine-tuned model loaded from {HF_REPO_ID} ✓")
```

## Cell 9 — Markdown

## 4. Run Benchmark — Base vs Fine-Tuned


## Cell 10 — Code

```python
def generate_response(model, tokenizer, user_prompt, max_new_tokens=256):
    """
    Generate a response for a given user prompt.

    Pipeline:
      1. Build chat messages: system prompt + user question
      2. Apply the model's chat template (adds special tokens)
      3. Tokenize and send to GPU
      4. Generate new tokens with sampling (temperature=0.7, top_p=0.9)
      5. Decode only the newly generated tokens (skip the prompt)

    Same function used in Module 2 for consistency.
    """
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
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Base model inference ──
# PeftModel has a neat feature: disable_adapter_layers() temporarily removes
# the LoRA adapter, giving you pure base model behavior WITHOUT loading a
# second copy of the model. This saves VRAM and keeps the comparison fair.
print("Running base model...")
base_outputs = []
for prompt in BENCHMARK_PROMPTS:
    ft_model.disable_adapter_layers()    # ← temporarily remove LoRA adapter
    resp = generate_response(ft_model, tokenizer, prompt)
    base_outputs.append({"prompt": prompt, "response": resp})
    ft_model.enable_adapter_layers()     # ← re-enable LoRA adapter

# ── Fine-tuned model inference ──
# With adapters enabled (default), this generates using base + LoRA = fine-tuned behavior.
print("Running fine-tuned model...")
ft_outputs = []
for prompt in BENCHMARK_PROMPTS:
    resp = generate_response(ft_model, tokenizer, prompt)
    ft_outputs.append({"prompt": prompt, "response": resp})

# ── Comparison table ──
# Side-by-side: base model vs fine-tuned on all 10 prompts.
# This is the same comparison as Module 2, but loaded from HF Hub
# instead of a local adapter directory — proving deployment works.
rows = []
for b, f in zip(base_outputs, ft_outputs):
    rows.append({
        "Prompt": b["prompt"],
        "Base Model": b["response"][:200],
        "Fine-Tuned": f["response"][:200],
    })

df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 200)
df
```

## Cell 11 — Markdown

## 5. Option B — HF Inference API (No GPU Needed)

If you deployed an Inference Endpoint on Hugging Face, you can call it
via HTTP. This works from any machine — no local GPU required.

> **Note:** Free Inference API is rate-limited. For production, use a paid Inference Endpoint.


## Cell 12 — Code

```python
# ── HF Inference API example (requires a deployed Inference Endpoint) ──
#
# WHAT THIS IS:
# Instead of loading the model locally (needs GPU), you can deploy a model
# as an API endpoint on HF's infrastructure. Then you call it via HTTP
# from any machine — no GPU needed on your side. Think of it like calling
# the OpenAI API, but with your own fine-tuned model.
#
# HOW TO SET UP:
# 1. Go to huggingface.co → Inference Endpoints → Create new
# 2. Select your adapter repo (e.g., jeev1992/healthcare-assistant-lora)
# 3. Choose a GPU instance (starts at ~$0.60/hr)
# 4. Get the endpoint URL when it's ready
#
# Uncomment and set your endpoint URL if you have one:

# ENDPOINT_URL = "https://your-endpoint-id.us-east-1.aws.endpoints.huggingface.cloud"
# HF_TOKEN = "hf_your_token"
#
# client = InferenceClient(model=ENDPOINT_URL, token=HF_TOKEN)
#
# for prompt in BENCHMARK_PROMPTS[:3]:
#     response = client.text_generation(
#         prompt=f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
#         max_new_tokens=256,
#         temperature=0.7,
#     )
#     print(f"Q: {prompt}")
#     print(f"A: {response}\n")

print("Inference API example (uncomment above if endpoint is deployed)")
```

## Cell 13 — Markdown

## 6. Export Results for LangSmith Evaluation

Save outputs so Module 4 can pick them up without re-running inference.


## Cell 14 — Code

```python
# ── Export results for downstream use ──
# Save the base and fine-tuned outputs as JSON so Module 4 (LangSmith evaluation)
# can score them without re-running inference.
#
# This file is the "handoff" between Module 3 and Module 4:
#   Module 3 → inference_results.json → Module 4 (LangSmith eval)
results = {
    "model_id": MODEL_ID,
    "adapter_repo": HF_REPO_ID,
    "benchmark_prompts": BENCHMARK_PROMPTS,
    "base_outputs": base_outputs,
    "finetuned_outputs": ft_outputs,
}

with open("inference_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved inference_results.json ✓")
print("\nNext: → Module 4: Evaluate with LangSmith (langsmith_eval.ipynb)")
```