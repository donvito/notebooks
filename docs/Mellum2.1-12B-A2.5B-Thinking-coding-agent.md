# Mellum2.1 12B-A2.5B Thinking coding agent

[`JetBrains/Mellum2.1-12B-A2.5B-Thinking`](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) is JetBrains’ 12B MoE / ~2.5B-active coding model (Apache 2.0). This notebook is a mini SWE-style agent: embed a tiny failing pytest repo, let the model propose a unified diff, apply it, and rerun tests (1–2 iterations).

## Use cases

- **Local coding-agent smoke tests** that apply a diff and rerun pytest.
- **A thinking MoE coder** when you want unified diffs rather than FrogNano’s Leaf tools.
- **Q4_K_M on a T4** (~8.1 GB) instead of BF16.

## Not a good fit for

- **The trained 131K window.** This notebook allocates 32,768 tokens (16K also works).
- **BF16 / FlashAttention-2.** Not used here.
- **A five-tool Leaf harness.** That is [FrogNano](FrogNano-4B-coding-agent.md).

## Running it on a free Colab T4

Pick a T4 GPU. Official Q4_K_M GGUF (~8.1 GB) via `llama-cpp-python`, 32K context, `flash_attn=False`. Weights plus a 16–32K cache land around 9–11 GB on 15 GB.

## What the notebook demos

- Load official Q4_K_M GGUF.
- Embed a tiny failing pytest repo.
- Propose a unified diff, apply it, rerun tests (1–2 iterations).

## Links

- Model card: [JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
- GGUF: [Mellum2.1-12B-A2.5B-Thinking-GGUF](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF)
- Blog: [Mellum 2.1](https://blog.jetbrains.com/ai/2026/10/mellum2-1-gets-to-work-a-fast-open-model-for-coding-agents/)
- Notebook: [`colab/Mellum2.1-12B-A2.5B-Thinking-coding-agent.ipynb`](../colab/Mellum2.1-12B-A2.5B-Thinking-coding-agent.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Mellum2.1-12B-A2.5B-Thinking-coding-agent.ipynb)
- Related: [FrogNano-4B](FrogNano-4B-coding-agent.md)
