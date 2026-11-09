# LFM2.5-2.6B llama-cpp-python

This notebook loads Liquid AI’s LFM2.5-2.6B Q4_K_M GGUF with CUDA `llama-cpp-python`, verifies chat, benchmarks inference, and runs a multi-tool request. License is LFM Open License v1.0. It is the lower-level llama.cpp walkthrough next to the aibackends LFM2.5 tool-calling notebook.

## Use cases

- **GPU GGUF chat** with full offload (`n_gpu_layers=-1`) and a 16,384-token context.
- **A quick throughput check** on a Colab T4 before you embed the model in an app.
- **Multi-tool loops** following Liquid’s tool-use workflow (describe functions, execute in Python, return tool messages).

## Not a good fit for

- **CPU-only Colab.** This walkthrough requires a T4 or newer.
- **The aibackends task API.** Use [AIBackends LFM2.5 tool calling](AIBackends-LFM2_5-tool-calling.md) for that wrapper.
- **LFM3 (newer family).** See the LFM3-1.7B notebooks.

## Running it on a free Colab T4

Pick a T4 GPU. Install the pinned CUDA 12.1 `llama-cpp-python` wheel from a direct URL so pip does not grab the CPU package. Restart the session if `llama_cpp` was already imported. Load Q4_K_M from `LiquidAI/LFM2.5-2.6B-GGUF`.

## What the notebook demos

- Inspect the Colab GPU.
- Install the CUDA wheel.
- Load LFM2.5-2.6B on the GPU.
- Verify basic chat generation.
- Benchmark inference.
- Run a multi-tool request.

## Links

- Model card: [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B)
- GGUF: [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)
- Tool-use docs: [Liquid AI](https://docs.liquid.ai/lfm/key-concepts/tool-use)
- Notebook: [`colab/LFM-2_5-2_6b-llamacpp-python.ipynb`](../colab/LFM-2_5-2_6b-llamacpp-python.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LFM-2_5-2_6b-llamacpp-python.ipynb)
- Related: [AIBackends LFM2.5 tool calling](AIBackends-LFM2_5-tool-calling.md), [LFM3-1.7B chat](LFM3-1.7B-chat-tools.md)
