# Module 3 — Webinar Script (Speaker Notes)

> **Total time: ~30 min** • 15 min lecture + 15 min live demo
> Slides: `docs/Week9_FineTuning_Domain_Adaptation.pptx` — Section: *Deployment*

---

## 2:30 – 2:33 · Frame the module (3 min)

Slide title: **"A fine-tuned model that lives in a Colab notebook is useless."**

Deliver:

> "Training was half the job. Deployment is what turns a trained artifact into
> a product. And here's the surprise — for a LoRA adapter, deployment is
> almost trivial. Let me show you what actually got pushed to Hugging Face."

---

## 2:33 – 2:38 · What lives on HF Hub (5 min)

Open the attendee's HF repo (or the reference one: `jeev1992/healthcare-assistant-lora-v2`).
Show the files literally on screen.

```
your-username/healthcare-assistant-lora-v2/
├── adapter_config.json          (~1 KB)
├── adapter_model.safetensors    (~15 MB)
├── tokenizer.json
└── special_tokens_map.json
```

Message:

> "This is not a model. This is a **patch**. It's 15 MB — smaller than a movie
> poster. The 3 GB base model stays where Qwen already hosts it.
> Your adapter tells PEFT: *when you load Qwen, apply this diff on top*."

Open `adapter_config.json` in a preview tab:

```json
{
    "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
}
```

> "That one file — the config — tells any consumer of your adapter exactly
> where to apply which weight update. It's the manifest."

---

## 2:38 – 2:43 · The adapter-toggle pattern (5 min)

Draw or slide:

```
WITH ADAPTER ENABLED:
  output = base_layer(x) + lora_B(lora_A(x)) × (alpha/r)   ← adapter active

WITH ADAPTER DISABLED:
  output = base_layer(x)                                    ← pure base model
```

```python
ft_model.enable_adapter_layers()   # Fine-tuned behavior
ft_model.disable_adapter_layers()  # Base model behavior
```

Deliver:

> "One model object in memory. Two behaviors. You get both the base model
> and the fine-tuned model for the VRAM cost of one. This is *the* pattern
> for A/B testing before you ship."

---

## 2:43 – 2:48 · Generation parameters for healthcare (5 min)

| Param | Healthcare recommendation | Why |
|---|---|---|
| `temperature` | 0.1 – 0.3 | Medical info must be consistent; high temp → hallucinations |
| `top_p` | 0.5 – 0.7 | Restrict to high-probability tokens |
| `max_new_tokens` | 256 – 512 | Long enough for thorough answers, short enough to prevent rambling |
| `do_sample` | `True` (with low temp) | Not deterministic, but not chaotic either |

Point out on the slide:

> "For a *creative writing* app, temperature 0.9 makes sense. For a *healthcare*
> app, temperature 0.9 is a lawsuit. Default to low temperatures for anything
> factual and regulated."

---

## 2:48 – 3:00 · Live demo — load and benchmark (12 min)

Open [`notebooks/03_hf_deploy_inference.ipynb`](notebooks/03_hf_deploy_inference.ipynb).

1. **Login to HF** — run `notebook_login()` cell. Show the attendees where the token
   comes from (settings → tokens → Write).
2. **Load base model in 4-bit** with the *exact same* BitsAndBytesConfig as training.
   Say this out loud:

   > "The quantization config at inference **must match training**. The adapter
   > was tuned against 4-bit weights — load the base in fp16 and your adapter
   > corrections are calibrated against the wrong numbers."

3. **Load the adapter** with `PeftModel.from_pretrained(base_model, "your-username/...")`.
4. Run the benchmark loop with the toggle pattern:

   ```python
   for prompt in BENCHMARK_PROMPTS:
       ft_model.disable_adapter_layers()
       base_out = generate(ft_model, tokenizer, prompt)

       ft_model.enable_adapter_layers()
       ft_out   = generate(ft_model, tokenizer, prompt)

       results.append({"prompt": prompt, "base": base_out, "finetuned": ft_out})
   ```

5. Save `inference_results.json` — **explicitly note** that Module 4 will consume this file.

While it runs (~2–3 min), transition to the production section.

---

## 3:00 – 3:05 · Production reality check (5 min) — lecture only

Slide with this table (walk through it fast):

| Strategy | Examples | Cold start | Cost model | Best for |
|---|---|---|---|---|
| **Serverless** | HF Endpoints, SageMaker Serverless | 30–120s | Pay-per-request | <100 req/day |
| **Dedicated GPU** | SageMaker Real-time, Azure ML, Vertex AI | None | Pay-per-hour | Production w/ SLA |
| **Container (vLLM/TGI)** | ECS, EKS, EC2 with vLLM | None | Pay-per-hour | Multi-model, K8s teams |
| **Model-as-a-Service** | Bedrock, Azure OpenAI | None | Pay-per-token | Zero-ops (limited catalog) |

Key lines to hit:

- **Right-size your GPU.** 1.5B in 4-bit needs a T4 ($0.74/hr). An A100 ($37/hr) is 50× the cost for the same throughput. Overprovisioning is the #1 way teams waste money on inference.
- **vLLM is a game-changer.** Raw `model.generate()` handles one request at a time.
  vLLM with continuous batching = **2–4× throughput** on the same GPU.
- **Merge adapters for production.**
  ```python
  merged = ft_model.merge_and_unload()
  merged.save_pretrained("prod-model/")
  ```
  One fewer file to load, one fewer failure point. In class we keep the adapter
  separate for the toggle trick — in prod, merge it.
- **Healthcare compliance:** AWS/Azure/GCP all offer **HIPAA-eligible** configurations,
  but you must configure them (VPC, encryption, audit logs). RunPod, Lambda Labs,
  and generic GPU providers are **not** HIPAA-compliant by default.

---

## Wrap-up (last 30 sec of Module 3)

Recap in one slide:

1. Adapter = a 15 MB patch, not a model.
2. Toggle base ↔ fine-tuned in one load with `enable/disable_adapter_layers()`.
3. Use **identical quantization** at inference as at training.
4. For production: right-size the GPU, batch with vLLM, merge the adapter.

Transition:

> "We now have 10 prompts × 2 versions saved to `inference_results.json`.
> Time to actually *measure* which one is better."

---

## Presenter cheat-sheet

- If `PeftModel.from_pretrained()` errors with "cannot find adapter_config.json" —
  it means the attendee typed the wrong HF repo name. Trailing slashes, typos, wrong
  username. Have them re-open the repo in a browser and copy the exact ID from the URL.
- If the audience asks about serving on-device: current LoRA adapters can be
  applied to on-device runtimes (Apple MLX, ONNX Runtime with a converter, GGUF via
  llama.cpp with a merge step) — mention it, don't demo it, it's a separate webinar.
- **"Can I fine-tune on top of a fine-tuned model?"** — Yes: load base + adapter A,
  merge, then LoRA again to get adapter B. But you're baking A into the base — you
  lose the toggle. Only do this if adapter A is finalized.
