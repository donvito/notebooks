# Aion 1.0 Instruct 2.3B on-device SLM

[`microsoft/Aion-1.0-Instruct-2.3B`](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) is Microsoft Windows Applied Sciences Group's **~2.29B MIT instruct SLM** for on-device text. The card tags it for **summarization, rewriting, and chat**, lists eight languages (English, German, Spanish, Portuguese, Italian, French, Chinese, Japanese), and a **32,768-token** context. Custom modeling code requires `trust_remote_code=True`. Hub weights are **bfloat16** (`model.safetensors` ~4.6 GB). Knowledge cutoff is **2025-12**. A further-optimized Windows OS SLM, paired with a local content-moderation model, is planned by the end of January 2027; this Hugging Face dump is the unquantized introduction.

## Use cases

- **Summarize a long email thread or multi-page document on-device** when the text should stay on the machine. The card's uses include document comprehension, content analysis, question answering, and processing lengthy email threads within the 32K window without sending them to a cloud model.
- **Rewrite a paragraph** (tone, length, or audience) over text you already have. Rewriting is a card tag; the [Windows sample](https://github.com/microsoft/Aion-Instruct-Preview-Sample) exposes rewrite as a first-class Text Intelligence task. This notebook does the same job through the instruct chat template.
- **Short chat in one of the eight listed languages**, including Spanish or Japanese. The card says output quality may still differ by language.
- **A T4-sized smoke of the official Transformers path** with the bundled `chat_template.jinja` (`<|system|>` / `<|user|>` / `<|assistant|>`, every turn closed by `<|end|>`, which is also EOS).

## Not a good fit for

- **World knowledge, open trivia, or long multi-step reasoning.** The card: the model was not trained to excel at world knowledge; at 2.3B it can miss fine details, misstate facts, or fail complex reasoning compared with larger models.
- **Medical diagnoses, legal judgments, or other automated high-stakes decisions.** Listed as out of scope on the card, along with harmful, violent, explicit, offensive, or misleading content.
- **The card quickstart unchanged on a T4 or on transformers 5.24.x.** `dtype="auto"` loads bfloat16 from `config.json`; a T4 has no native bf16. A [public report](https://x.com/prompt48/status/2124597425310695696) found the same Hub snippet on transformers **5.24.0** produced gibberish with the KV cache on, and decoded cleanly on the export pin **5.9.0**.
- **A full 32K context on a free T4.** That window is a card limit. Weights plus a long KV cache do not fit ~15 GB; keep prompts short.
- **The Windows Copilot+ NPU sample as this runtime.** [microsoft/Aion-Instruct-Preview-Sample](https://github.com/microsoft/Aion-Instruct-Preview-Sample) is a WinUI app on Snapdragon / Lunar Lake / Ryzen AI. This notebook is Transformers on a Colab T4.
- **FlashAttention-2 or native bfloat16 on this GPU.** T4 is compute capability 7.5.

## Running it on a free Colab T4

Pick a T4 GPU. Pin **`transformers==5.9.0`** (the `transformers_version` in `config.json` / `generation_config.json`). Load with `trust_remote_code=True`. Use **float16** (~4.6 GB of weights) and **SDPA** (eager if SDPA refuses the hybrid sliding-window stack). If a short probe is NaN or non-text, reload **float32** (~9.2 GB; still fits 15 GB). Do not pass `dtype="auto"`. Do not install `flash-attn`.

Chat formatting is `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. The template injects the system preamble, knowledge cutoff `2025-12`, and the current date. Hub `generate` is greedy (`max_new_tokens` only); `generation_config.json` sets `eos_token_id` to `[200020, 199999]`, `pad_token_id` 199999, and `use_cache: true`. The Windows sample's chat client uses `Temperature = 0.5`, `TopP = 0.9`, `TopK = 40`; this notebook uses those settings only on the multilingual chat turn. Peak VRAM is printed from `torch.cuda.max_memory_allocated`. Expected peak is about **5–7 GB** in float16, or about **10 GB** if the notebook falls back to float32 (estimate from weight size plus a short window, not a measured Colab run).

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Pinned transformers 5.9.0 (no `python -I`; Colab torch kept).
- Load with `trust_remote_code`, float16 then float32, SDPA then eager; a one-word banana probe to catch NaNs/gibberish.
- The bundled chat template rendered once so the `<|system|>` / `<|end|>` format is visible.
- Summarize a short multi-paragraph email thread (greedy).
- Rewrite a stiff paragraph as a friendly Slack message (greedy).
- One Spanish chat turn with the Windows sample's 0.5 / 0.9 / 40 sampler.
- Peak VRAM and cleanup.

## Links

- Model card: [microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B)
- Windows sample: [microsoft/Aion-Instruct-Preview-Sample](https://github.com/microsoft/Aion-Instruct-Preview-Sample)
- Prompt guide: [Improving Prompts for Aion](https://github.com/microsoft/Aion-Instruct-Preview-Sample/blob/main/docs/prompt-guide.md)
- Source post: [x.com/mswnlz](https://x.com/mswnlz/status/2123114277330395257)
- Public report on the 5.24.0 quickstart: [x.com/prompt48](https://x.com/prompt48/status/2124597425310695696)
- ONNX WebGPU INT4 sibling: [microsoft/Aion-1.0-Instruct-2.3B-ONNX-WebGPU](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B-ONNX-WebGPU)
- License: [MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE)
- Notebook: [`colab/Aion-1.0-Instruct-2.3B-on-device-slm.ipynb`](../colab/Aion-1.0-Instruct-2.3B-on-device-slm.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Aion-1.0-Instruct-2.3B-on-device-slm.ipynb)
- Related: [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md), [Granite 4.2 3B chat and tools](granite-4.2-3b-chat-tools.md), [Smaug Humanizer](Smaug-Humanizer-llamacpp.md)
