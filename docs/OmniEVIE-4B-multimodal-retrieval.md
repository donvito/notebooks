# OmniEVIE-4B multimodal retrieval

[`tencent/OmniEVIE-4B`](https://huggingface.co/tencent/OmniEVIE-4B) is Tencent IMA Product Center and Tencent Youtu Lab’s 4.54B unified multi-modal, multi-vector retriever ([Apache-2.0](https://huggingface.co/tencent/OmniEVIE-4B/blob/main/LICENSE.txt)), fine-tuned from [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B). The card lists modalities **text, natural image, visual document, native video**, and agent / tool / GUI / memory. Retrieval is ColBERT-style MaxSim (one vector per token), with Prefix-MRL dimensions `{64, 128, 256, 512, 1024, 2048}`. Hub files are **9.1 GB** `model.safetensors` (bf16) plus `query_gate.safetensors`. This notebook uses the card’s **Sentence Transformers** `MultiVectorEncoder` path (`>=6.3.0`, `transformers>=5.13.1`, `trust_remote_code` not required), not the `omnievie` package.

## Use cases

- **Text-to-text search** when you want late-interaction MaxSim over sentences or short passages, not a single pooled vector.
- **Text-to-image / visual-document search** without an OCR step: pass PIL images, file paths, or URLs to `encode_document`, as the card’s image snippet does.
- **A T4-sized smoke of the Sentence Transformers export** before you take the card’s production `omnievie` runtime (ST-HAC `K128×d64 int8` compression, `QueryGate`, three-stage CPU index, page/video localization).

## Not a good fit for

- **The official `omnievie` package on a free T4.** That path requires `flash-attn>=2.7.0` and `attn_implementation="flash_attention_2"`. A T4 (`sm_75`) cannot run FlashAttention-2.
- **Matching the card’s printed bf16 scores.** This notebook loads **float16 + SDPA**.
- **Native video retrieval on a free T4.** The card supports video (`encode_query(..., prompt_name="query_video")`); this notebook does not run it.
- **Million-scale three-stage CPU search, QueryGate saliency, or page-box / timestamp localization.** Those APIs are on the `omnievie` package (`ThreeStageIndex`, `QueryGate`, `localize_image`), not this Sentence Transformers export.
- **Audio.** The card’s MMEB-Leaderboard claim is **excluding audio**.
- **A tiny dense MiniLM / EmbeddingGemma path, or sparse inverted-index search.** See [EmbeddingGemma 2](embeddinggemma-2-inference-similarity.md), [AIBackends MiniLM](AIBackends-MiniLM-embeddings-similarity.md), and [SlideOn](SlideOn-sparse-retrieval.md).
- **A rerun of published MMEB / ViDoRe / latency numbers.** Those are the authors’ results, not this notebook.

## Running it on a free Colab T4

Pick a T4 GPU. Keep Colab’s CUDA torch. Install **`sentence-transformers[image]>=6.3.0`** and **`transformers>=5.13.1`** (the card’s pins; the video extra is skipped). Do not install `flash-attn`. Load `MultiVectorEncoder("tencent/OmniEVIE-4B")` with **float16 + SDPA** (eager fallback) and `batch_size=1` (the card’s tip for Qwen3.5 linear-attention layers). First run downloads **~9.1 GB**. Weight VRAM is about **9.1 GB**; peak while encoding a few 768 px images is an **estimate of ~10–13 GB**. Images are capped on the long edge so the session stays under ~14 GB.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card Sentence Transformers load in **float16** (`MultiVectorEncoder`).
- Text-to-text MaxSim over the card’s three-sentence example, plus a `truncate_dim=64` Prefix-MRL shape print.
- Text-to-image MaxSim over three public Hub figure URLs from the card’s image snippet.
- A visual-document page query over three more Hub figures (ST-HAC, three-stage funnel, token alignment).
- Session `torch.cuda.max_memory_allocated` after cleanup.

## Links

- Model card: [tencent/OmniEVIE-4B](https://huggingface.co/tencent/OmniEVIE-4B)
- GitHub: [Tencent/OmniEVIE](https://github.com/Tencent/OmniEVIE)
- Report: [arXiv:2611.32203](https://arxiv.org/abs/2611.32203)
- Source post: [x.com/paperswithcodee](https://x.com/paperswithcodee/status/2125530235802100031)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/OmniEVIE-4B-multimodal-retrieval.ipynb`](../colab/OmniEVIE-4B-multimodal-retrieval.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/OmniEVIE-4B-multimodal-retrieval.ipynb)
- Related: [pplx-embed-v2 late](pplx-embed-v2-late-0.6b-page-search.md), [pplx-embed-v2 context](pplx-embed-v2-context-0.6b-contextual-rag.md), [Nemotron Embed visual RAG](Nemotron-3.5-Embed-1B-visual-rag.md), [EmbeddingGemma 2](embeddinggemma-2-inference-similarity.md)
