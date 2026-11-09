# Spark-X2.6-4B on-device agent

[`XHToken/Spark-X2.6-4B`](https://huggingface.co/XHToken/Spark-X2.6-4B) is SparkLLM’s 4.11B on-device agent (Apache 2.0, `spark2_5` architecture). The chat template takes `enable_thinking` plus `reasoning_effort` (`low` / `high`) and a `tools` list in OpenAI function form. This notebook is think-then-answer chat, a small coding task, and a tool-use loop.

When a Spark-X2.6-1.7B notebook lands, copy this page’s sections (use cases, T4 dtype, pins, GGUF option) and swap the Hub id, parameter count, and weight sizes.

## Use cases

- **Think-then-answer chat** with `enable_thinking` and `reasoning_effort`.
- **A small on-device coding task.**
- **A tool-use loop** in OpenAI function form.

## Not a good fit for

- **131,072 new tokens on a T4.** The card’s `max_new_tokens` does not fit; demos stay in the low hundreds.
- **FlashAttention 2.** Shipped modeling code is eager attention only.
- **Skipping the transformers pin.** The card pins `transformers==4.57.1` and `trust_remote_code`.

## Running it on a free Colab T4

Pick a T4 GPU. Pin `transformers==4.57.1` and `trust_remote_code`. About 4.11B parameters, ~8.2 GB in float16; falls back to 4-bit bitsandbytes if fp16 overflows. Optional [`GGUF`](https://huggingface.co/XHToken/Spark-X2.6-4B-GGUF) `Q4_K_M` (~3.54 GB). Restart after pinning transformers.

## What the notebook demos

- Load Spark-X2.6-4B (float16, 4-bit fallback).
- Think-then-answer chat.
- Coding task.
- Tool-use loop.
- Caveats.

## Links

- Model card: [XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B)
- GGUF: [Spark-X2.6-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.6-4B-GGUF)
- Notebook: [`colab/Spark-X2.6-4B-on-device-agent.ipynb`](../colab/Spark-X2.6-4B-on-device-agent.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Spark-X2.6-4B-on-device-agent.ipynb)
