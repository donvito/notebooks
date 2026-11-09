# AIBackends Transformers runtime tasks

This notebook runs the same aibackends built-in tasks you would run on llama.cpp, but through the Hugging Face Transformers runtime (`aibackends[transformers]`). Any Hugging Face causal LM works via `ModelRef(name="<hf repo id>")`. The walkthrough uses ungated [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) so it needs no Hub token. The registered `GEMMA3_270M_IT` profile also works, but `google/gemma-3-270m-it` is gated.

## Use cases

- **Swap runtimes without rewriting tasks.** Invoice extract, summarize, and classify use the same task API as the llama.cpp notebook.
- **Prototype on an ungated small LM.** Qwen2.5-1.5B-Instruct runs anywhere without accepting a gated license first.
- **Free-form chat on the same loaded model.** `get_runtime` exposes raw completions after the structured tasks.

## Not a good fit for

- **Gated Gemma without a token.** `google/gemma-3-270m-it` needs a license accept plus an `HF_TOKEN` Colab secret.
- **Vision document OCR.** Use the [vision receipt OCR](AIBackends-vision-receipt-OCR.md) notebook for images.
- **GGUF-only deployment.** That path is the [llama.cpp document tasks](AIBackends-llamacpp-document-tasks.md) notebook.

## Running it on a free Colab T4

It runs on a free CPU runtime. A T4 is faster and is detected automatically. On GPU the notebook loads weights in float16 to halve memory. `device` maps to Transformers `device_map` (`gpu` → `cuda`). The model is reused across tasks with identical config so it loads once.

## What the notebook demos

- Structured invoice extraction.
- Summarize and classify on the same loaded runtime.
- Raw chat completions via `get_runtime`.

## Links

- Model card: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Notebook: [`colab/AIBackends-Transformers-runtime-tasks.ipynb`](../colab/AIBackends-Transformers-runtime-tasks.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-Transformers-runtime-tasks.ipynb)
- Related: [llama.cpp document tasks](AIBackends-llamacpp-document-tasks.md), [custom tasks and batch](AIBackends-custom-tasks-pipelines-batch.md)
