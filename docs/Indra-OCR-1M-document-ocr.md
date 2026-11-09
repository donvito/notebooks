# Indra OCR 1M document OCR

[`datalab-to/indra-ocr`](https://huggingface.co/datalab-to/indra-ocr) is Datalab’s ~1M-parameter ONNX document OCR (code Apache 2.0, weights OpenRAIL-M). It is built for CPU. The notebook OCRs a receipt, a multi-page PDF, handwriting, Korean, a rotated page, and math-as-LaTeX.

## Use cases

- **CPU OCR in a Colab without a GPU**, including multi-page PDFs.
- **Handwriting, rotation, and math-as-LaTeX** as a tiny-model smoke test.
- **Korean and other multilingual pages** without a 1B+ VLM.

## Not a good fit for

- **GPU-accelerated VLM OCR.** Indra stays on ONNX Runtime CPU even on a T4.
- **Unrestricted commercial use.** OpenRAIL-M here is modified: free for research, personal use, and startups under $20M funding/revenue; it cannot be used competitively with Datalab’s API. Read the LICENSE.
- **Grounded VLM OCR with layout JSON from a 0.8B–4B model.** Use LightOnOCR / MiniCPM-OCR / Xiaomi-OCR.

## Running it on a free Colab T4

A free Colab CPU runtime is enough. A T4 also works, but Indra does not use the GPU. First run downloads about 76 MB of default models.

## What the notebook demos

- Receipt OCR.
- Multi-page PDF.
- Handwriting.
- Korean.
- A rotated page.
- Math as LaTeX.

## Links

- Model card: [datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr)
- Notebook: [`colab/Indra-OCR-1M-document-ocr.ipynb`](../colab/Indra-OCR-1M-document-ocr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Indra-OCR-1M-document-ocr.ipynb)
- Related: [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md)
