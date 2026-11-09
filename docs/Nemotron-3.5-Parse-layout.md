# Nemotron 3.5 Parse layout

[`nvidia/Nemotron-3.5-Parse`](https://huggingface.co/nvidia/Nemotron-3.5-Parse) is NVIDIA’s layout parser (OpenMDW-1.1, ~631M). It returns layout boxes, semantic classes, and reading-order Markdown/HTML. It is a layout complement to LightOnOCR / OvisOCR, not a full OCR replacement.

## Use cases

- **Reading-order Markdown/HTML** from a page image when you care about boxes and classes.
- **A layout pass** before or beside an OCR VLM.
- **A Transformers T4 path** instead of the card’s Hopper vLLM/FA3 recipe.

## Not a good fit for

- **Replacing OCR.** Pair with LightOnOCR / OvisOCR / MiniCPM-OCR for text.
- **vLLM, BF16, or FlashAttention 2 on a T4.** Not used here.

## Running it on a free Colab T4

Pick a T4 GPU. Transformers float16 + SDPA. Hub weights are about 2.5 GB on disk; float16 in VRAM is about 1.3 GB. A 2048×1664 page plus a few thousand generated tokens should stay well under 16 GB (estimate from parameter count and max canvas, not a measured Colab run).

## What the notebook demos

- Layout boxes and semantic classes.
- Reading-order Markdown/HTML.

## Links

- Model card: [nvidia/Nemotron-3.5-Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse)
- Notebook: [`colab/Nemotron-3.5-Parse-layout.ipynb`](../colab/Nemotron-3.5-Parse-layout.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Nemotron-3.5-Parse-layout.ipynb)
- Related: [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md), [OvisOCR3](OvisOCR3-structured-json.md)
