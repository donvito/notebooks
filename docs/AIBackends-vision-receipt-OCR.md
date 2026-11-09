# AIBackends vision receipt OCR

This notebook extracts structured receipt data and describes images with local vision-language models on llama.cpp. aibackends downloads each GGUF and the matching `mmproj` projector. It compares Gemma 4 E2B (`unsloth/gemma-4-E2B-it-GGUF`), LFM2.5-VL-3B (`LiquidAI/LFM2.5-VL-3B-GGUF`), and Qwen3-VL-4B (`Qwen/Qwen3-VL-4B-Instruct-GGUF`).

## Use cases

- **Receipt fields from a photo.** `ImageIngestor` plus `VisionExtractor` validate the reply against a schema so you can compare headline fields across three local VLMs.
- **Image understanding.** Ask a local VLM what is in a picture without a hosted vision API.
- **Runtime bake-off.** Same workflow, three GGUFs, so you can pick a model for an on-box expense app.

## Not a good fit for

- **Grounded OCR with layout boxes.** Use a dedicated OCR notebook (LightOnOCR, MiniCPM-OCR, Xiaomi-OCR) when you need reading-order text, not schema JSON.
- **Fast CPU iteration.** Each model is about a 3 GB download; CPU runs take a few minutes per image.
- **Text-only invoices.** Use [llama.cpp document tasks](AIBackends-llamacpp-document-tasks.md) when you already have text.

## Running it on a free Colab T4

It runs on a free CPU runtime. A T4 is faster and is detected automatically. Each model is about a 3 GB download. CPU inference is slow (a few minutes per image).

## What the notebook demos

- Define `ImageIngestor` + `VisionExtractor` workflows.
- Receipt OCR with Gemma 4, LFM2.5-VL, and Qwen3-VL.
- Compare headline fields across models.
- Image understanding prompts.

## Links

- Models: [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) · [LiquidAI/LFM2.5-VL-3B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF) · [Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)
- Notebook: [`colab/AIBackends-vision-receipt-OCR.ipynb`](../colab/AIBackends-vision-receipt-OCR.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-vision-receipt-OCR.ipynb)
- Related: [llama.cpp document tasks](AIBackends-llamacpp-document-tasks.md), [`OvisOCR3-structured-json.ipynb`](../colab/OvisOCR3-structured-json.ipynb), [`LightOnOCR-3-0.8B-document-ocr.ipynb`](../colab/LightOnOCR-3-0.8B-document-ocr.ipynb)
