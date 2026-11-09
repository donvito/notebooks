# LFM3-VL-2B vision

[`LiquidAI/LFM3-VL-2B`](https://huggingface.co/LiquidAI/LFM3-VL-2B) is Liquid AI’s ~1.92B vision-language model (LFM Open License v1.0) for image Q&A and short video. Needs `transformers>=5.21.0` and `trust_remote_code`.

## Use cases

- **Image question answering** on a small edge VLM.
- **A few frames of video → text** when you can cap at 8 frames / 2 fps.
- **The backbone behind d1-2B** if you want generative VL rather than decisions.

## Not a good fit for

- **Long video.** The processor default is up to 768 frames, which will OOM a T4. This notebook caps at 8 frames / 2 fps.
- **Calibrated boxes / yes-no.** Use [d1-2B](d1-2B-inference-decisions.md).
- **Document OCR as a product.** Use LightOnOCR / MiniCPM-OCR.

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 with SDPA. About 3.8 GB of weights. Image questions should sit around 5–8 GB. `av` writes and decodes the clip if `torchcodec` is missing.

## What the notebook demos

- Load LFM3-VL-2B.
- Image Q&A.
- A short 2 fps video (8 frames).

## Links

- Model card: [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B)
- Source post: [x.com/liquidai](https://x.com/liquidai/status/2115128329003303107)
- Blog: [LFM3](https://www.liquid.ai/blog/lfm3)
- Notebook: [`colab/LFM3-VL-2B-vision.ipynb`](../colab/LFM3-VL-2B-vision.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LFM3-VL-2B-vision.ipynb)
- Related: [d1-2B](d1-2B-inference-decisions.md), [LFM3-Omni-2B](LFM3-Omni-2B.md)
