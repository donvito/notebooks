# LFM3-1.7B chat and tools

[`LiquidAI/LFM3-1.7B`](https://huggingface.co/LiquidAI/LFM3-1.7B) is Liquid AI’s ~1.73B text model (LFM Open License v1.0) for on-device chat, thinking, and tool calling. Native Transformers LFM3 support is still landing (`transformers#50036`), so the notebook uses `trust_remote_code`.

## Use cases

- **Short local chat** with optional thinking.
- **A mock tool-calling loop** before you wire real tools.
- **The safety-tuned Transformers path** to compare with the uncensored GGUF notebook.

## Not a good fit for

- **Low-refusal creative writing.** Use [LFM3-1.7B Uncensored GGUF](LFM3-1.7B-Uncensored-GGUF-llamacpp.md).
- **Vision, audio, or omni.** Use the VL / Audio / Omni notebooks.
- **256K context on a T4.** Keep generations short; that window does not fit a T4 KV cache.
- **Apache/MIT-only production.** Read LFM Open License v1.0.

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 (~3.5 GB of weights) with SDPA. `trust_remote_code`. No native BF16, no FlashAttention-2.

## What the notebook demos

- Load LFM3-1.7B.
- Text chat.
- Thinking.
- A mock tool-calling loop.

## Links

- Model card: [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)
- Source post: [x.com/liquidai](https://x.com/liquidai/status/2115128329003303107)
- Blog: [LFM3](https://www.liquid.ai/blog/lfm3)
- Notebook: [`colab/LFM3-1.7B-chat-tools.ipynb`](../colab/LFM3-1.7B-chat-tools.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LFM3-1.7B-chat-tools.ipynb)
- Related: [Uncensored GGUF](LFM3-1.7B-Uncensored-GGUF-llamacpp.md), [LFM3-VL-2B](LFM3-VL-2B-vision.md)
