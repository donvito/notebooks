# FrogNano-4B coding agent

[`microsoft/FrogNano-4B-2609`](https://huggingface.co/microsoft/FrogNano-4B-2609) is Microsoft's ~4.66B repository-level coding agent (Hub tag MIT; card also lists Apache 2.0 via Qwen3.5-4B). It is post-trained with RL on about 1,500 synthetic SWE tasks from Qwen3.5-4B and is meant to be driven through the Leaf harness: `Read`, `Write`, `Edit`, `Glob`, and `Bash`. This notebook is a mini Leaf loop, not a unified-diff agent.

## Use cases

- **Tiny local coding-agent experiments.** Embed a failing pytest repo and let the model inspect, patch, and rerun tests through typed tools.
- **Leaf-harness practice** before you scale to the official 150-step / ~131K-context eval setup.
- **A smaller alternative to Mellum** when Q4_K_M around 2.8 GB is a better fit than an 8 GB MoE GGUF.

## Not a good fit for

- **Full SWE-bench runs.** The demo caps the loop at 8 Leaf turns (about 2–3 inspect/edit/test cycles). Official evals use 150 steps and ~131K context, which this notebook does not allocate.
- **Native float16/BF16 on a free T4.** Native BF16 is about 9.3 GB before KV cache and is tight on 15 GB.
- **Unified-diff apply/rerun like Mellum.** Different harness.

## Running it on a free Colab T4

Pick a T4 GPU. Load bartowski Q4_K_M GGUF (~2.80 GB) with `llama-cpp-python`, 16,384-token context. No BF16, no FlashAttention-2 (`flash_attn=False`). Install the pinned CUDA 12.1 wheel plus `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` on current Colab images.

## What the notebook demos

- Install the CUDA `llama-cpp-python` wheel.
- GPU check and T4 caveats.
- Load Q4_K_M.
- In-process Leaf tools.
- Embed a tiny failing invoice repo.
- Mini Leaf loop (8 turns).
- Caveats.

## Links

- Model card: [microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609)
- GGUF: [bartowski/FrogNano-4B-2609-GGUF](https://huggingface.co/bartowski/FrogNano-4B-2609-GGUF)
- Leaf harness: [microsoft/FrogNano](https://github.com/microsoft/FrogNano)
- Notebook: [`colab/FrogNano-4B-coding-agent.ipynb`](../colab/FrogNano-4B-coding-agent.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/FrogNano-4B-coding-agent.ipynb)
- Related: [`Mellum2.1-12B-A2.5B-Thinking-coding-agent.ipynb`](../colab/Mellum2.1-12B-A2.5B-Thinking-coding-agent.ipynb)
