# ARC 1.0 agent model router

[`rayline-ai/ARC-1.0`](https://huggingface.co/rayline-ai/ARC-1.0) is Rayline’s 0.6B Apache-2.0 decision model (a fine-tune of [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)). It reads an agent conversation and predicts per-model success, token usage, PII presence, request categories, and answers developer-defined questions. It does not generate the next assistant turn. Policy is supplied by the caller: **Utility = V × predicted success − expected cost**, with card presets Sage V = 4.5 and Swift V = 0.1.

## Use cases

- **Route an agent across a model pool** when you already have list prices and want predicted success minus expected cost, without retraining when prices change.
- **Prefer quality or prefer cost** with the same predictors: Sage (V = 4.5) vs Swift (V = 0.1).
- **PII presence and request categorization** (operations and domains) on the same encoder readout, before a request leaves the box.
- **Ask a routing question** such as “Does this request need a stronger model?” via the question head.

## Not a good fit for

- **Writing replies or tool calls.** Pair ARC with a generator; this checkpoint scores conversations.
- **Drop-in `transformers` + `trust_remote_code` or stock vLLM.** The architecture is `ArcImagePolicyForPooling`; there is no Hub `modeling_*.py`. Stock vLLM cannot load it. The public [`vllm-arc`](https://github.com/rayline-ai/vllm-arc) plugin is Stage-1 only and needs a CUDA 13 vLLM fork, which a free Colab T4 does not ship.
- **Hosted Routes / Generation without a Rayline key.** This notebook is local and secret-free. Use [docs.rayline.ai](https://docs.rayline.ai) for the hosted APIs.
- **Bit-exact goldens or Stage-2 reasoning-effort serving.** Goldens were recorded on vLLM GATHER readouts. Stage-2 effort arms are in the weights; the public plugin does not serve them yet.

## Running it on a free Colab T4

Pick a T4 GPU. Load `Qwen/Qwen3.5-0.8B` in float16 with SDPA (eager fallback). The [NOTICE](https://huggingface.co/rayline-ai/ARC-1.0/blob/main/NOTICE) says text layers 0–11 are unchanged from that backbone; specialized heads sit on top. Combined download is about 4 GB: Qwen3.5-0.8B ~1.7 GB, ARC `model.safetensors` 1.3 GB (heads only; encoder tensors skipped in RAM), question-head upper layers ~1.0 GB. Peak VRAM on the short demos should stay around 2–5 GB of 15 GB (estimate, not a measured Colab run). Do not install `flash-attn`, `flash-linear-attention`, or `causal-conv1d`.

## What the notebook demos

- Check the T4 and pick float16.
- Load the Qwen3.5-0.8B backbone, overlay the question-head upper layers, and load Stage-1 / PII / topic heads.
- Route three sample conversations across a named model pool with the reference price card, showing Sage vs Swift.
- PII detection and request categorization, including a fictional PII example.
- A developer-defined noul / choice question via the bilinear pointer.

## Links

- Model card: [rayline-ai/ARC-1.0](https://huggingface.co/rayline-ai/ARC-1.0)
- Source post: [x.com/davidvgilmore](https://x.com/davidvgilmore/status/2120583960447029699)
- Docs: [docs.rayline.ai](https://docs.rayline.ai)
- Technical report: [ARC](https://static.rayline.ai/rayline/reports/arc/latest)
- Plugin: [rayline-ai/vllm-arc](https://github.com/rayline-ai/vllm-arc)
- CLI: [rayline-ai/rayline-cli](https://github.com/rayline-ai/rayline-cli)
- Notebook: [`colab/ARC-1.0-agent-model-router.ipynb`](../colab/ARC-1.0-agent-model-router.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/ARC-1.0-agent-model-router.ipynb)
- Related: [THX-01](THX-01-decision-routing.md), [d1-2B](d1-2B-inference-decisions.md), [elephant-1](elephant-1-inference-decisions.md)
