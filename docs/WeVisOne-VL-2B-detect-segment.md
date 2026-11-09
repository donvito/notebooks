# WeVisOne-VL 2B detect and cut out

[`tencent/WeVisOne-VL-2B`](https://huggingface.co/tencent/WeVisOne-VL-2B) is Tencent’s Apache 2.0 Qwen3-VL model (Hub ~2.44B) for open-vocab boxes, counts, points, and alpha cutouts. Needs the `wevisone_vl` package.

## Use cases

- **Open-vocabulary boxes and counts** on a photo.
- **Point grounding.**
- **Alpha cutouts / image matting** from referring expressions.

## Not a good fit for

- **Plain image Q&A without detection.** Use MiniCPM-V or LFM3-VL.
- **Document OCR as a product.** Use LightOnOCR / Xiaomi-OCR.

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA. About 4.9 GB of weights, roughly 5–8 GB with a 1024 px image.

## What the notebook demos

- Open-vocab boxes.
- Counts and points.
- Alpha cutouts.

## Links

- Model card: [tencent/WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B)
- Notebook: [`colab/WeVisOne-VL-2B-detect-segment.ipynb`](../colab/WeVisOne-VL-2B-detect-segment.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/WeVisOne-VL-2B-detect-segment.ipynb)
