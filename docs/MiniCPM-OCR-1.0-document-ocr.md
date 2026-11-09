# MiniCPM-OCR 1.0 document OCR

[`openbmb/MiniCPM-OCR-1.0`](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) is OpenBMB’s ~1.3B document OCR model (Apache 2.0) with native `MiniCPMOcrForConditionalGeneration` in Transformers 5.21. Region prompts (`Text Recognition:`, `Table Recognition:`) produce OTSL then Markdown, plus receipt key-field JSON.

## Use cases

- **Region-prompt OCR** when you want text vs table recognition as separate instructions.
- **Receipt key fields as JSON** without a separate extract model.
- **An Apache-2.0 ~2.6 GB T4 path** that does not need PaddlePaddle or vLLM.

## Not a good fit for

- **Layout SDKs / PaddlePaddle / vLLM / DSpark.** Out of scope.
- **CPU 1M OCR.** Use [Indra](Indra-OCR-1M-document-ocr.md).
- **Grounded LightOnOCR text.** Different recipe; see [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md).

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA. About 2.6 GB of weights. A single page or crop with up to 1024 new tokens should stay well under 6 GB (estimate, not a measured Colab number).

## What the notebook demos

- Region-prompt text recognition.
- Table recognition → OTSL → Markdown.
- Receipt key-field JSON.

## Links

- Model card: [openbmb/MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)
- Source post: [x.com/OpenBMB](https://x.com/OpenBMB/status/2115124474882469955)
- Notebook: [`colab/MiniCPM-OCR-1.0-document-ocr.ipynb`](../colab/MiniCPM-OCR-1.0-document-ocr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/MiniCPM-OCR-1.0-document-ocr.ipynb)
- Related: [MiniCPM-V 4.7](MiniCPM-V-4.7-1B-vision.md), [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md)
