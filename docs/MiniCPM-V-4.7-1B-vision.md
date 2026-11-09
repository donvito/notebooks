# MiniCPM-V 4.7 1B vision

[`openbmb/MiniCPM-V-4.7-1B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) is OpenBMB’s Apache 2.0 on-device VLM (Hub ~1.55B). The notebook does image Q&A, receipt OCR, a written integral, and a short video. Optional llama.cpp GGUF section.

## Use cases

- **General image Q&A** on a ~3 GB T4 VLM.
- **Receipt OCR and a handwritten integral** as vision smokes, not a dedicated OCR stack.
- **Short video** when you can cap frames.

## Not a good fit for

- **128-frame video on a T4.** The notebook samples 16 frames so it stays on 16 GB.
- **Dedicated document OCR prompts / OTSL.** Use [MiniCPM-OCR 1.0](MiniCPM-OCR-1.0-document-ocr.md).
- **The card’s FlashAttention-2 / bf16 auto path.** A T4 uses float16 + SDPA.

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA. About 3.1 GB of weights, roughly 6–8 GB for an image. Video capped at 16 frames. Decoding uses PyAV (`av`) when `torchcodec` cannot load.

## What the notebook demos

- Image Q&A.
- Receipt OCR.
- A written integral.
- Short 16-frame video.
- Optional llama.cpp GGUF section.

## Links

- Model card: [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)
- Notebook: [`colab/MiniCPM-V-4.7-1B-vision.ipynb`](../colab/MiniCPM-V-4.7-1B-vision.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/MiniCPM-V-4.7-1B-vision.ipynb)
- Related: [MiniCPM-OCR 1.0](MiniCPM-OCR-1.0-document-ocr.md), [LFM3-VL-2B](LFM3-VL-2B-vision.md)
