# WePocketModel 1.0 4B — generation, embeddings, and decisions

[`tencent/WePocketModel1.0-4B`](https://huggingface.co/tencent/WePocketModel1.0-4B) is WeChat AI (Tencent)’s on-device vision-language checkpoint in the WePocketModel (WePM) family ([Apache-2.0](https://huggingface.co/tencent/WePocketModel1.0-4B/blob/main/LICENSE.txt)). One shared backbone and vision encoder support three capabilities: **generation** (OCR, captioning, VQA), **embedding** (L2-normalized text and image vectors; 4B full dim 2048, with MRL at 64 / 128 / 256 / 512 / 1024 / 2048), and **decision** (structured choices, ordinal scores, or binary judgments). The Hub architecture is Qwen3.5 (`Qwen3_5ForConditionalGeneration`, ~4.54B parameters). GitHub reports about **8 GB / ≈8.5 GiB of BF16 weight memory** for the 4B backbone; Hub shards plus `unimodel_aux.safetensors` are about **9.1 GB** on disk. Inference is the [WePM Transformers runtime](https://github.com/Tencent/WePocketModel), not a raw `transformers` generate loop. A [2B safetensors](https://huggingface.co/tencent/WePocketModel1.0-2B) sibling (~4.47 GB of BF16 weights) and [4B GGUF](https://huggingface.co/tencent/WePocketModel1.0-4B-GGUF) exist; this notebook uses the 4B Transformers checkpoint, loaded as **float16** on a T4.

## Use cases

- **One on-device model instead of three specialists.** The card: a personal assistant that needs multimodal retrieval, generation, and a typed next-action decision can keep a single resident backbone rather than separate generation, embedding, and decision models.
- **Local OCR / captioning / VQA.** Generation takes text, images, or interleaved text and images. The notebook reads a public receipt (OCR + a short visual question).
- **Multimodal retrieval and ranking.** Embedding encodes text and images into L2-normalized vectors. Use the full 2048-d vector, or a Matryoshka prefix (`--embedding-dim` 64–2048) when you want a shorter index.
- **Typed routing from application state.** Decision returns structured `choice`, ordinal `score`, or binary `noul` (yes/no) answers in the SystemOne-visible request format. Validate those outputs before they trigger actions — the card says so.

## Not a good fit for

- **The card’s CUDA BF16 path on a free T4.** GitHub: Transformers on NVIDIA GPU uses BF16 and recommends **16 GiB VRAM**. A T4 (`sm_75`) has no native bfloat16 and about **15 GB**. This notebook loads **float16**. Quality numbers on the card’s BF16 tables are not rerun here.
- **A drop-in `AutoModel.generate` chat loop.** Generation, embedding, and decision go through `UniModelTransformers` in the WePM repo (`model.generate` / `model.embed` / `model.decide`). Embedding appends a learned E slot; decision appends a J slot; generation is greedy AR with no soft slot.
- **GGUF / CPU-only deploys.** GGUF lives at [`tencent/WePocketModel1.0-4B-GGUF`](https://huggingface.co/tencent/WePocketModel1.0-4B-GGUF) (Q4_K_M language file plus shared multimodal weights). This notebook is the CUDA Transformers path.
- **Treating the GitHub evaluation figure or HCD comparison as a result of this notebook.** Those are project-reported figures; this file does not copy scores or claim a T4 measurement.
- **Open-vocab boxes / alpha cutouts.** Use [WeVisOne-VL 2B](WeVisOne-VL-2B-detect-segment.md). For audio-visual video, use [WeMM-Video-2B](WeMM-Video-2B-audio-visual-video-understanding.md).

## Running it on a free Colab T4

Pick a T4 GPU. GitHub install pins `torch==2.10.0` and `transformers==5.3.0` (`config.json` records `transformers_version: 5.3.0`). **Do not reinstall Colab’s CUDA torch.** The notebook keeps that wheel, **pins `transformers==5.3.0`**, installs `accelerate`, `huggingface_hub`, `safetensors`, `jinja2`, and `pillow`, clones [Tencent/WePocketModel](https://github.com/Tencent/WePocketModel), and restarts the session. The runtime hardcodes `dtype=torch.bfloat16` on CUDA; the load cell intercepts that assignment and loads **float16** on Turing. First run downloads the **~9.1 GB** 4B snapshot (five backbone shards + `unimodel_aux.safetensors`). GitHub waits for free CUDA memory equal to the checkpoint size plus 2 GiB before load (~11 GiB); the notebook passes `--min-free-gib` so a 15 GB T4 can start. If 4B still does not fit, it falls back to [`tencent/WePocketModel1.0-2B`](https://huggingface.co/tencent/WePocketModel1.0-2B) (~4.47 GB of BF16 weights) and prints that it did so. Peak VRAM on the short demos is printed at the end; it has **not** been measured on a real T4 in this repo. Weight memory alone is ≈8.5 GiB in float16, so peak with MGVB + a short generate is expected in the **~11–13 GB** range — an estimate from the card’s 16 GiB CUDA guidance, not a Colab log.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- WePM Transformers load in **float16** on T4 (`UniModelTransformers`, SDPA). 4B first; 2B only if 4B does not fit.
- **Generation:** OCR + VQA on a public SROIE receipt (`hf-internal-testing/fixtures_ocr`).
- **Embedding:** L2-normalized vectors for a few texts and images; rank them against a query (full 2048-d, plus a 256-d MRL prefix).
- **Decision:** one `choice`, one ordinal `score`, and one binary `noul` request in the card’s SystemOne-visible JSON format.
- Session `torch.cuda.max_memory_allocated` after cleanup.

## Links

- Model card: [tencent/WePocketModel1.0-4B](https://huggingface.co/tencent/WePocketModel1.0-4B)
- 2B fallback: [tencent/WePocketModel1.0-2B](https://huggingface.co/tencent/WePocketModel1.0-2B)
- 4B GGUF: [tencent/WePocketModel1.0-4B-GGUF](https://huggingface.co/tencent/WePocketModel1.0-4B-GGUF)
- Runtime: [Tencent/WePocketModel](https://github.com/Tencent/WePocketModel)
- Technical report: [arXiv:2611.27365](https://arxiv.org/abs/2611.27365)
- Collection: [tencent/wepocketmodel](https://huggingface.co/collections/tencent/wepocketmodel)
- Source post: [x.com/TencentAI_News](https://x.com/TencentAI_News/status/2125152448218034505)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/WePocketModel-4B-embed-vqa-decisions.ipynb`](../colab/WePocketModel-4B-embed-vqa-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/WePocketModel-4B-embed-vqa-decisions.ipynb)
- Related: [WeMM-Video-2B](WeMM-Video-2B-audio-visual-video-understanding.md), [WeVisOne-VL 2B](WeVisOne-VL-2B-detect-segment.md), [MiniCPM-V 4.7 1B](MiniCPM-V-4.7-1B-vision.md), [d1-2B decisions](d1-2B-inference-decisions.md), [Nemotron 3.5 Embed 1B](Nemotron-3.5-Embed-1B-visual-rag.md), [EmbeddingGemma 2](embeddinggemma-2-inference-similarity.md)
