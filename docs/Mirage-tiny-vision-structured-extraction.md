# Mirage tiny vision structured extraction

[`Cactus-Compute/mirage`](https://huggingface.co/Cactus-Compute/mirage) is Cactus Compute’s Apache-2.0 on-device vision model. The [blog](https://cactuscompute.com/blog/mirage) gives **51.3 million parameters** in about **22 MB** (`mirage.cact` is 21.6 MB of 2-bit Cactus Quants; a companion `mirage.safetensors` is 216 MB and is not what the Python API loads). One pass returns a structured record: scene, open-vocab objects with boxes, OCR with confidence, segmentation masks, spatial relations, and schema-based decisions.

## Use cases

- **On-device scene records** for phones, wearables, or robots: names, boxes, and relations as JSON instead of a chat caption.
- **A tiny OCR + box pass** on signs or short printed labels, with a per-line score.
- **Constrained visual decisions** (`extract` with an `enum`, bounded integer, or string field) when you need a choice, not a paragraph.
- **Mask overlays** for the objects Mirage already named, without a separate segmenter.

## Not a good fit for

- **GPU VLMs / free-text image Q&A.** This notebook’s path is the Cactus CPU engine and structured JSON, not a chat VLM. Use [MiniCPM-V](MiniCPM-V-4.7-1B-vision.md) or [LFM3-VL](LFM3-VL-2B-vision.md).
- **Dedicated document OCR.** Use [LightOnOCR](LightOnOCR-3-0.8B-document-ocr.md) / [Indra OCR](Indra-OCR-1M-document-ocr.md) / [Xiaomi-OCR](Xiaomi-OCR-0-document-ocr.md). Mirage’s reader is a small crop OCR inside the same 22 MB file.
- **Languages outside the card list.** Descriptions and the speech/text stack are **en, de, fr, es, it, nl, pl**.
- **The 216 MB safetensors file as the runtime weights.** `cactus.Mirage()` loads `mirage.cact`.

## Running it on a free Colab T4

A free CPU runtime is enough. A T4 is the runtime this notebook is tested on, but **Mirage does not use the GPU** — inference is the Linux x86-64 Cactus engine. First use downloads the engine plus `mirage.cact` (~22 MB). Keep Colab’s torch; do not reinstall it. The notebook still prints `torch.cuda.is_available()`, the device name, and free VRAM.

The [blog](https://cactuscompute.com/blog/mirage) quotes 74 ms on a Galaxy S26 NPU and 107 ms on that phone’s CPU. Those are not Colab numbers. This notebook reports wall time per image on the runtime you actually have.

## What the notebook demos

- `cactus.Mirage().describe(...)` on a COCO indoor photo, with object boxes drawn on the image.
- OCR plus a polygon-mask overlay on the blog’s COCO stop-sign photo (including the documented `ARRÉT` accent).
- `extract(...)` schema / visual decision on a public SROIE receipt (`document_type` enum).

## Links

- Model card: [Cactus-Compute/mirage](https://huggingface.co/Cactus-Compute/mirage)
- Source post: [x.com/cactuscompute](https://x.com/cactuscompute/status/2118761482343313844)
- Blog: [Mirage: VLM + OCR + Detection + Segment + Decide in 22 MB](https://cactuscompute.com/blog/mirage)
- Python API: [Needle Python Docs](https://cactuscompute.com/blog/needle-python-docs)
- Runtime: [Cactus-Compute/cactus](https://huggingface.co/Cactus-Compute/cactus)
- Source: [cactus-compute/cactus](https://github.com/cactus-compute/cactus)
- Notebook: [`colab/Mirage-tiny-vision-structured-extraction.ipynb`](../colab/Mirage-tiny-vision-structured-extraction.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Mirage-tiny-vision-structured-extraction.ipynb)
- Related: [WeVisOne-VL 2B](WeVisOne-VL-2B-detect-segment.md), [MiniCPM-V 4.7 1B](MiniCPM-V-4.7-1B-vision.md), [Indra OCR](Indra-OCR-1M-document-ocr.md), [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md)
