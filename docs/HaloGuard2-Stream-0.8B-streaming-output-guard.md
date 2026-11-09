# HaloGuard2-Stream 0.8B streaming output guard

[`astroware/HaloGuard2-Stream-0.8B`](https://huggingface.co/astroware/HaloGuard2-Stream-0.8B) is Astroware’s ~0.75B streaming output-safety model (Apache 2.0), built on Qwen3.5-0.8B plus a safety head. It emits P(unsafe) at every response token and a latched `first_fire_index` (max ≥ 0.975 or two consecutive ≥ 0.70). It is not a prompt/response classifier like GliGuard.

## Use cases

- **Cut a streaming completion** as soon as the safety head fires, instead of waiting for the full reply.
- **Log `first_fire_index`** so you can see which token tripped the guard.
- **On-box output moderation** next to a generator, with a small 0.8B head.

## Not a good fit for

- **Prompt-side jailbreak/toxicity labels.** Use [GliGuard](AIBackends-GLiGuard_LLMGuardrails-300M-moderation.md).
- **Generating the user-facing answer.** This model scores tokens; it does not write the reply.
- **Installing `flash-linear-attention` / `causal-conv1d` on free Colab.** Use the torch fallback.

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 with SDPA (eager if SDPA refuses the hybrid stack). `trust_remote_code`. About 1.5 GB of weights, roughly 2–4 GB peak for the short demos (estimate from the 1.5 GB shard plus a 2,048-token window, not a measured Colab run). The card says ≥ 4 GB VRAM. Do not install `flash-linear-attention` or `causal-conv1d`.

## What the notebook demos

- Load HaloGuard2-Stream 0.8B.
- Per-token P(unsafe) on streaming output.
- Latched `first_fire_index`.
- Comparison notes vs GliGuard.

## Links

- Model card: [astroware/HaloGuard2-Stream-0.8B](https://huggingface.co/astroware/HaloGuard2-Stream-0.8B)
- Source post: [x.com/xnavkum](https://x.com/xnavkum/status/2118010382992457822)
- Notebook: [`colab/HaloGuard2-Stream-0.8B-streaming-output-guard.ipynb`](../colab/HaloGuard2-Stream-0.8B-streaming-output-guard.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/HaloGuard2-Stream-0.8B-streaming-output-guard.ipynb)
- Related: [GliGuard moderation](AIBackends-GLiGuard_LLMGuardrails-300M-moderation.md)
