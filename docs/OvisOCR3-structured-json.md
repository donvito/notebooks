# OvisOCR3 structured JSON

[`ATH-MaaS/OvisOCR3`](https://huggingface.co/ATH-MaaS/OvisOCR3) is ATH-MaaS’s Apache 2.0 ~0.85B OCR VLM (Qwen3.5-0.8B). This notebook asks for prompt-driven structured JSON from a receipt and a table page — invoice fields and a two-page table rebuild, not grounded OCR text like LightOnOCR.

## Use cases

- **Receipt fields as JSON** from a photo.
- **Rebuild a table as structured data** rather than Markdown OCR.
- **A 1.7 GB T4 extract path** when you already know the schema in the prompt.

## Not a good fit for

- **Grounded OCR strings / layout boxes.** Use LightOnOCR or Nemotron Parse.
- **CPU 1M OCR.** Use Indra.

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA. About 0.85B / 1.7 GB of weights. A 2048 px page peaks at roughly 5–8 GB (estimate from the weight file plus a page, not a measured Colab run).

## What the notebook demos

- Receipt → structured JSON.
- Table page → structured JSON rebuild.

## Links

- Model card: [ATH-MaaS/OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3)
- Notebook: [`colab/OvisOCR3-structured-json.ipynb`](../colab/OvisOCR3-structured-json.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/OvisOCR3-structured-json.ipynb)
- Related: [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md), [Nemotron 3.5 Parse](Nemotron-3.5-Parse-layout.md)
