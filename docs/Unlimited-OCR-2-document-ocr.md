# Unlimited-OCR-2 document OCR

[`baidu/Unlimited-OCR-2`](https://huggingface.co/baidu/Unlimited-OCR-2) is Baidu’s ~3.34B MIT OCR VLM (`UnlimitedOCRForCausalLM`). This notebook does single-page OCR for a receipt, a handwritten letter, and a results table via the official `infer()` path.

## Use cases

- **Single-page receipt / handwriting / table OCR** on a MIT 3.3B VLM.
- **The official `infer()` API** rather than a third-party serving stack.

## Not a good fit for

- **SGLang or long multi-page.** Out of scope.
- **The card’s BF16-on-A800 recipe.** This notebook uses float16 on a T4.
- **CPU 1M OCR.** Use Indra.

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + official `infer()`. About 6.7 GB of weights.

## What the notebook demos

- Receipt OCR.
- Handwritten letter.
- Results table.

## Links

- Model card: [baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2)
- Notebook: [`colab/Unlimited-OCR-2-document-ocr.ipynb`](../colab/Unlimited-OCR-2-document-ocr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Unlimited-OCR-2-document-ocr.ipynb)
- Related: [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md), [MiniCPM-OCR 1.0](MiniCPM-OCR-1.0-document-ocr.md)
