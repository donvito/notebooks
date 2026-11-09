# LFM3-1.7B Uncensored GGUF

[`AtomicChat/LFM3-1.7B-Uncensored-GGUF`](https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF) is a safety-ablated / refusal-reduced GGUF of LiquidAI/LFM3-1.7B (same LFM Open License v1.0). This notebook chats with official llama.cpp **b11960** (CUDA 12.8, `#31252` / `v0.8.0`), not the 0.3.36 `llama-cpp-python` wheel (that build predates `lfm3`).

## Use cases

- **Low-refusal creative demos** (adult fiction, dark-humor roast, villain monologue) versus the safety-tuned Transformers LFM3 notebook.
- **GGUF llama.cpp on a T4** when you need the ablated checkpoint rather than the official chat model.

## Not a good fit for

- **Safety-tuned assistants.** Use [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md).
- **The stock `llama-cpp-python` 0.3.36 wheel.** It predates `lfm3`; this notebook pins official llama.cpp b11960.
- **256K context.** Context is capped at 8,192 tokens.

## Running it on a free Colab T4

Pick a T4 GPU. Default weights are AD-Q4_K_M (~1.1 GB); optional AD-Q6_K (~1.6 GB). FlashAttention off. Weights plus the 8K cache land around 2–4 GB on 15 GB.

## What the notebook demos

- Install official llama.cpp b11960 CUDA 12.8.
- Load AD-Q4_K_M (optional Q6_K).
- Three low-refusal creative chats versus the Transformers LFM3 notebook.

## Links

- GGUF: [AtomicChat/LFM3-1.7B-Uncensored-GGUF](https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF)
- Base: [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)
- Source post: [x.com/atomic_chat_hq](https://x.com/atomic_chat_hq/status/2115183360138223808)
- Blog: [LFM3](https://www.liquid.ai/blog/lfm3)
- Notebook: [`colab/LFM3-1.7B-Uncensored-GGUF-llamacpp.ipynb`](../colab/LFM3-1.7B-Uncensored-GGUF-llamacpp.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LFM3-1.7B-Uncensored-GGUF-llamacpp.ipynb)
- Related: [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md)
