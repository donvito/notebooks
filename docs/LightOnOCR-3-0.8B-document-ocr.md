# LightOnOCR-3 0.8B document OCR

[`lightonai/LightOnOCR-3-0.8B`](https://huggingface.co/lightonai/LightOnOCR-3-0.8B) is LightOn’s Apache 2.0 grounded OCR model (~0.85B). The notebook OCRs a receipt, a handwritten letter, and a results table. Prefer this size for speed on simple pages; prefer 4B when layout quality matters.

## Use cases

- **Receipts, letters, and tables** when you want grounded OCR text rather than schema JSON.
- **A 1.7 GB T4 path** that still has headroom on a 2048 px page.
- **A baseline before you spend on the 4B quality upgrade.**

## Not a good fit for

- **Dense multi-column forms** where 4B quality matters. Use [LightOnOCR-3 4B](LightOnOCR-3-4B-document-ocr.md).
- **Prompted invoice JSON.** Use [`OvisOCR3-structured-json.ipynb`](../colab/OvisOCR3-structured-json.ipynb).
- **CPU-only 1M OCR.** Use [Indra OCR](Indra-OCR-1M-document-ocr.md).

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 with SDPA. About 1.7 GB of weights; a page whose longest edge is 2048 px peaks at roughly 5–8 GB (estimate from the weight file plus a 2048 px page, not a measured Colab run). No native BF16, no FlashAttention 2.

## What the notebook demos

- Receipt OCR.
- Handwritten letter.
- Results table.

## Links

- Model card: [lightonai/LightOnOCR-3-0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B)
- Notebook: [`colab/LightOnOCR-3-0.8B-document-ocr.ipynb`](../colab/LightOnOCR-3-0.8B-document-ocr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LightOnOCR-3-0.8B-document-ocr.ipynb)
- Related: [LightOnOCR-3 4B](LightOnOCR-3-4B-document-ocr.md), [Indra OCR](Indra-OCR-1M-document-ocr.md)
