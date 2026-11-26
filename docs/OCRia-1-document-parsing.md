# OCRia-1 4B document parsing

[`llamaindex/OCRia-1`](https://huggingface.co/llamaindex/OCRia-1) is LlamaIndex’s 4B document-parsing vision-language model, fine-tuned from [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B). It reads one page image and writes Markdown (tables and chart data as HTML). The Hub file is the unoptimized BF16 weights (`model.safetensors`, 10.35 GB). License is `other` (OCRia-1 Model and Code License): free for research, and free for commercial use by organizations under $2M annual revenue and under $2M total funding. The card’s Transformers path is `AutoModelForImageTextToText` / `AutoProcessor` (Transformers 5.3 or later).

## Use cases

- **Whole-page transcription** of receipts, invoices, and forms into Markdown, including strikethrough, underline, checkboxes, and blank fields, without a separate layout detector or OCR engine.
- **Dense tables** where merged cells and spanning headers need `colspan` / `rowspan` in HTML (the card’s Texas insurance-rate example).
- **Charts as data tables.** Every chart is written as an HTML `<table>` of the plotted values, not a caption of the figure. The card reports **82.47** on ParseBench 1 charts (next-best open-weight model in that table: **69.14**).

## Not a good fit for

- **vLLM / LlamaParse serving, layout boxes, or GGUF.** This notebook is the card’s Hugging Face Transformers path on a free T4. GGUF builds live at [`llamaindex/OCRia-1-GGUF`](https://huggingface.co/llamaindex/OCRia-1-GGUF) (llama.cpp / Ollama). Layout boxes (`ocria_grounding`) are a separate experimental package on the card.
- **Matching the card’s 300 DPI / 24,576-token recipe on 16 GB.** The card renders PDF pages at 300 DPI and decodes with `max_tokens` 24,576. This notebook downscales to about 150 DPI / max side ~1600 px and caps `max_new_tokens` at 4096 so a T4 can finish.
- **Thinking mode.** The Qwen3.5 chat template turns thinking on by default; the card says the model then answers in the wrong format. Keep `enable_thinking=False`.
- **A rerun of ParseBench / olmOCR-bench numbers.** Those are the card’s scores, not this notebook.

## Running it on a free Colab T4

Pick a T4 GPU. T4 has no native bfloat16. Load **float16** (the card uses `dtype=torch.bfloat16`). If that does not fit, the notebook falls back to 8-bit then 4-bit bitsandbytes. Pin `transformers==5.20.1` (same Qwen3.5 VLM pin as the LightOnOCR-3 notebooks; the card’s floor is Transformers 5.3). Do not reinstall Colab’s torch. First run downloads **10.35 GB**. Peak VRAM is printed at the end; it has not been measured on a real T4 in this repo.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card Transformers load in **float16** (`AutoModelForImageTextToText` + `prompt.txt` as the system message, thinking off, greedy decoding).
- A SROIE receipt (same URL as the other OCR notebooks in this repo).
- Page 8 of Attention Is All You Need (Table 2), with HTML tables rendered in the notebook.
- The card’s line-chart example page, written as an HTML data table.
- Session `torch.cuda.max_memory_allocated` after cleanup.

## Links

- Model card: [llamaindex/OCRia-1](https://huggingface.co/llamaindex/OCRia-1)
- GGUF (llama.cpp / Ollama): [llamaindex/OCRia-1-GGUF](https://huggingface.co/llamaindex/OCRia-1-GGUF)
- Technical report: [arXiv:2611.00092](https://arxiv.org/abs/2611.00092)
- Source post: [x.com/llama_index](https://x.com/llama_index/status/2125315040723062872)
- License: [OCRia-1 Model and Code License](https://huggingface.co/llamaindex/OCRia-1/blob/main/LICENSE) (`other`)
- Notebook: [`colab/OCRia-1-document-parsing.ipynb`](../colab/OCRia-1-document-parsing.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/OCRia-1-document-parsing.ipynb)
- Related: [LightOnOCR-3 4B](LightOnOCR-3-4B-document-ocr.md), [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md), [Xiaomi-OCR-0](Xiaomi-OCR-0-document-ocr.md)
