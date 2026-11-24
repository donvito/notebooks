# Hy-MT3 1.8B translation

[`tencent/Hy-MT3-1.8B`](https://huggingface.co/tencent/Hy-MT3-1.8B) is Tencent’s 1.8B checkpoint in the third-generation Hy multilingual translation family ([Apache-2.0](https://huggingface.co/tencent/Hy-MT3-1.8B/blob/main/LICENSE.txt)). The card lists three sizes — **1.8B**, **7B**, and **30B-A3B** (MoE) — and **102 languages** (38 from Hy-MT2 plus 64 new ones). Load is Transformers `AutoTokenizer` + `AutoModelForCausalLM` (`hunyuan_v1_dense`). Hub weights are one **3.6 GB** `model.safetensors` shard (`torch_dtype: bfloat16`). FP8 (`tencent/Hy-MT3-1.8B-FP8`) and GGUF (`tencent/Hy-MT3-1.8B-GGUF`) siblings are published; this notebook loads the full 1.8B safetensors in **float16**.

## Use cases

- **Text translation across the card’s 102 languages**, including pairs that are new in Hy-MT3 (for example Swahili) and ones already in Hy-MT2 (for example Filipino / `tl`). The card: use **full language names** in the prompt, English names with English prompts.
- **Constrained translation** when you must keep glossary terms, a style, delimiters, or structured data (keys / placeholders) intact. The card ships prompt templates for default translation, terminology, style, personalization, delimiters, and two structured-data layouts.
- **A T4-sized smoke of the Transformers path** before you take the card’s vLLM / SGLang serving recipes or the 7B / 30B-A3B checkpoints. The card’s industrial notes (game localization, OCR-cascaded image translation) describe the family; this notebook only runs short text prompts.

## Not a good fit for

- **7B or 30B-A3B on a free T4.** This notebook loads **1.8B** only.
- **The card’s `dtype=torch.bfloat16` on a T4.** A T4 (`sm_75`) has no native bfloat16. This notebook loads **float16**.
- **Speech translation.** Use [Granite Speech 5.0 2B](granite-speech-5.0-2b-asr-translation.md) for X→En / En→X from audio.
- **A rerun of published TransRubric / WildMTBench / IFMTBench numbers**, or the card’s comparison of 1.8B against commercial translation APIs. Those are the authors’ results, not this notebook.
- **FP8 or GGUF in this Colab.** The notebook does not load [`Hy-MT3-1.8B-FP8`](https://huggingface.co/tencent/Hy-MT3-1.8B-FP8) or [`Hy-MT3-1.8B-GGUF`](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF).
- **General chat, tools, or a default system prompt.** The card: these models have **no default system_prompt**. Prompts here are the translation templates only.

## Running it on a free Colab T4

Pick a T4 GPU. Keep Colab’s CUDA torch. Install **`transformers>=5.6.0`** (the card’s pin), `accelerate`, and **`jinja2>=3.1.6`** (needed for `apply_chat_template`). Do not install `flash-attn`. Load the public 1.8B safetensors with `trust_remote_code=True`, `device_map="auto"`, and **float16** (SDPA, then eager). First run downloads **3.6 GB**. Weights should fit easily on ~15 GB VRAM; peak allocated VRAM is printed at the end. The card’s 1.8B sampler is `temperature=0.7`, `top_p=0.6`, `top_k=20`, `repetition_penalty=1.05`; `max_tokens` on the card is 4096 — demos stay at a few hundred new tokens. Card `max_context` is **8192**.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card Transformers load in **float16** (`AutoTokenizer` + `AutoModelForCausalLM`).
- Default English prompt: a few sentences English→Chinese, Spanish, Japanese.
- The same default prompt on a less common pair from the supported list (English→Filipino and English→Swahili).
- Constrained translation with the card’s **terminology** template.
- Session `torch.cuda.max_memory_allocated` after cleanup.

## Links

- Model card: [tencent/Hy-MT3-1.8B](https://huggingface.co/tencent/Hy-MT3-1.8B)
- Collection: [tencent/hy-mt3](https://huggingface.co/collections/tencent/hy-mt3)
- GitHub: [Tencent-Hunyuan/Hy-MT3](https://github.com/Tencent-Hunyuan/Hy-MT3)
- Report: [arXiv:2611.27673](https://arxiv.org/abs/2611.27673)
- Source post: [x.com/TencentAI_News](https://x.com/TencentAI_News/status/2125177755943960580)
- GGUF: [tencent/Hy-MT3-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF)
- FP8: [tencent/Hy-MT3-1.8B-FP8](https://huggingface.co/tencent/Hy-MT3-1.8B-FP8)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/Hy-MT3-1.8B-translation.ipynb`](../colab/Hy-MT3-1.8B-translation.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Hy-MT3-1.8B-translation.ipynb)
- Related: [Granite Speech 5.0 2B ASR and translation](granite-speech-5.0-2b-asr-translation.md), [WeMM-Video-2B](WeMM-Video-2B-audio-visual-video-understanding.md)
