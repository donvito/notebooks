# Xiaomi-OCR-0 document OCR

[`xiaomi-research/Xiaomi-OCR-0`](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) is Xiaomi’s ~0.8B / 873M document OCR VLM (Apache 2.0, Qwen3.5-0.8B-Base). Mirror: [`SeerRay-Lab/Xiaomi-OCR-0`](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0). Whole-page Markdown (OTSL tables, LaTeX formulas), receipt key-field JSON, and document VQA. Official Transformers path (`transformers==5.17.0`); no SGLang, vLLM, or PP-DocLayoutV3.

## Use cases

- **Whole-page Markdown** with OTSL tables and LaTeX formulas.
- **Receipt key-field JSON.**
- **Document VQA** on the same 0.8B checkpoint.

## Not a good fit for

- **SGLang / vLLM / PP-DocLayoutV3.** Not used.
- **Skipping the transformers pin.** Official path is `transformers==5.17.0`.

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA. About 1.7 GB of weights.

## What the notebook demos

- Whole-page Markdown parsing.
- Receipt key-field JSON.
- Document VQA.

## Links

- Model card: [xiaomi-research/Xiaomi-OCR-0](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0)
- GitHub: [SeerRay-Lab/Xiaomi-OCR-0](https://github.com/SeerRay-Lab/Xiaomi-OCR-0)
- Paper: [arXiv:2609.36136](https://arxiv.org/abs/2609.36136)
- Source post: [x.com/MiniMax__AI](https://x.com/MiniMax__AI/status/2114754291368034781)
- Notebook: [`colab/Xiaomi-OCR-0-document-ocr.ipynb`](../colab/Xiaomi-OCR-0-document-ocr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Xiaomi-OCR-0-document-ocr.ipynb)
- Related: [MiniCPM-OCR 1.0](MiniCPM-OCR-1.0-document-ocr.md), [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md)
