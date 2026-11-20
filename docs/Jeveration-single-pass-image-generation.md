# Jeveration single-pass image generation

[`linlinlinlinsdada/Jeveration`](https://huggingface.co/linlinlinlinsdada/Jeveration) is Apache-2.0 LoRA adapters (rank 256) on a frozen [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) backbone. The card: the model returns a whole **512×512** image as one joint decision in a **single forward pass**. One checkpoint serves text-to-image, image-to-image (HED edge map or depth map to image, colorization, anime / ghibli / watercolor), and dense prediction (13-class NYUv2 segmentation, depth, surface normals). Weights are PyTorch state dicts (`jeveration.pt` 4.0 GB, `decoder512.pt` 1.2 GB, `source_decoder512.pt` 1.3 GB), loaded with `JeverationPipeline` from [Carol-lyh/Jeveration](https://github.com/Carol-lyh/Jeveration) — not a `transformers` or `diffusers` checkpoint.

## Use cases

- **A whole 512×512 image in one forward pass** when you want a System One / parallel-code generator instead of many autoregressive tokens or diffusion steps. The paper and project page time **78 ms** per image on one **H200** (0.74 GenEval; up to 122× vs Janus-Pro-7B on that table).
- **One checkpoint for three families of visual output:** text-to-image; structure-to-image / colorize / three trained styles; indoor seg + depth + normals.
- **A T4 smoke of the released pipeline** (`JeverationPipeline.text_to_image` / `image_to_image` / `dense_predict`) without standing up the Gradio demo.

## Not a good fit for

- **Treating 78 ms / 0.74 GenEval / 67.7 mIoU as free-Colab T4 figures.** Those are from the card, paper, and project page (H200 timings; NYUv2 654-image table for dense).
- **Arbitrary resolution, aspect ratio, or non-English prompts.** The card: images are 512×512; prompts and instructions are English.
- **Open image-to-image edits beyond the six trained tasks** (`hed`, `depth`, `colorize`, `anime`, `ghibli`, `watercolor`) with their trained instructions.
- **Dense prediction on outdoor scenes or odd aspect ratios.** The dense heads were trained on NYUv2 (indoor scenes, 4:3 pictures).
- **Loading both 512-px decoders plus the 4B backbone in float32 on a 15 GB T4.** The notebook loads float16 and only the decoder each demo needs.

## Running it on a free Colab T4

Pick a T4 GPU. The repo install is `pip install -r requirements.txt` (`torch==2.7.1`, `transformers==5.12.1`). Keep Colab’s CUDA torch; this notebook pins `transformers==5.12.1`, `open_clip_torch==3.3.0`, `omegaconf==2.3.0`, `einops==0.8.1`, and `flash-linear-attention==0.5.2`, and restarts so that transformers version is imported. A T4 has no native bfloat16 — the clone is patched to **float16** (backbone `device_map="cuda"`, LoRA created in the backbone dtype, autocast float16). First run downloads `jeveration.pt` (4.0 GB) plus the decoder used by that demo, the Qwen3.5-4B shards (~9.3 GB), TA-TiTok (`model.safetensors` 1.6 GB; only the codebook stays), and CLIP (`open_clip_model.safetensors` 1.7 GB; the vision tower is dropped). Peak VRAM is printed at the end. The 78 ms H200 number is not this runtime.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card install from [Carol-lyh/Jeveration](https://github.com/Carol-lyh/Jeveration) at commit `e0cc08bb8fa4a0c0c060cd189e55901010c2c92c`, keeping Colab’s torch.
- Text-to-image (README pine-tree prompt, seed 0) with wall-clock timing.
- Colorization of the README greyscale pancakes photo (source decoder only; GitHub example with a COCO fallback).
- Dense seg + depth on the README indoor room photo, one task at a time (13 NYUv2 classes).
- `torch.cuda.max_memory_allocated` peak and cleanup.

## Links

- Model card: [linlinlinlinsdada/Jeveration](https://huggingface.co/linlinlinlinsdada/Jeveration)
- Code: [Carol-lyh/Jeveration](https://github.com/Carol-lyh/Jeveration)
- Paper: [arXiv:2611.24738](https://arxiv.org/abs/2611.24738)
- Project page: [carol-lyh.github.io/Jeveration](https://carol-lyh.github.io/Jeveration/)
- Demo: [Hugging Face Space](https://huggingface.co/spaces/linlinlinlinsdada/Jeveration)
- Source post: [x.com/carol00892](https://x.com/carol00892/status/2123808797009543293)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/Jeveration-single-pass-image-generation.ipynb`](../colab/Jeveration-single-pass-image-generation.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Jeveration-single-pass-image-generation.ipynb)
- Related: [WeVisOne-VL 2B](WeVisOne-VL-2B-detect-segment.md), [MiniCPM-V 4.7 1B](MiniCPM-V-4.7-1B-vision.md), [LFM3-VL-2B](LFM3-VL-2B-vision.md), [Mirage](Mirage-tiny-vision-structured-extraction.md)
