# Module 3 — Deployment & Inference

**Duration in webinar:** ~30 minutes · **GPU needed:** ✅ Yes (Colab T4)

## What this module answers

1. What actually lives on Hugging Face Hub when I push a "fine-tuned model"?
2. How do I load a base model + a LoRA adapter and switch between them at runtime?
3. What are the right generation parameters for a healthcare Q&A assistant?
4. What are my options when I need to deploy this to production?

## What lives on HF Hub

The adapter you pushed in Module 2 is a **patch**, not a full model.

```
your-username/healthcare-assistant-lora-v2/     (≈ 20–50 MB)
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json

Qwen/Qwen2.5-1.5B-Instruct                       (≈ 3 GB, stays at HF)
```

At runtime: download base (cached after first pull) + download adapter + apply = <1 second.

## The adapter-toggle pattern

```python
ft_model.enable_adapter_layers()   # Fine-tuned behavior
ft_model.disable_adapter_layers()  # Base model behavior — same model object!
```

You get **both models in one load** — saves ~1 GB VRAM vs loading them separately.

## Files

- [`webinar_script.md`](webinar_script.md) — presenter script (15 min lecture + 15 min live demo)
- [`notebooks/03_hf_deploy_inference.ipynb`](notebooks/03_hf_deploy_inference.ipynb) — load adapter from Hub, run inference, export results for Module 4

## Open in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/module_3_deployment_inference/notebooks/03_hf_deploy_inference.ipynb)

## Production reality check (covered in the lecture)

| Strategy | Example | Cold start | Cost model | Good for |
|---|---|---|---|---|
| Serverless | SageMaker Serverless, HF Endpoints | 30–120s | Pay-per-request | <100 req/day |
| Dedicated GPU | SageMaker Real-time, Azure ML | None | Pay-per-hour | Production w/ SLA |
| Container (vLLM/TGI) | ECS, EKS, EC2 | None | Pay-per-hour | Multi-model, custom |
| Model-as-a-Service | Bedrock, Azure OpenAI | None | Pay-per-token | Zero-ops |

**Rule of thumb for a 1.5B model in 4-bit:** a T4 ($0.74/hr on SageMaker) is enough.
Using an A100 ($37/hr) wastes 98% of the GPU.
