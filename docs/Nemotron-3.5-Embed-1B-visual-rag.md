# Nemotron 3.5 Embed 1B visual RAG

[`nvidia/Nemotron-3.5-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) is NVIDIA’s multimodal embedder (OpenMDW-1.1) for visual RAG over PDF page images and text. This notebook retrieves with FAISS, then answers with `Qwen/Qwen2.5-1.5B-Instruct`. Unlike EmbeddingGemma 2’s text-only notebook, this one embeds page images.

## Use cases

- **Search PDF pages as images** when you do not want an OCR step first.
- **A small RAG loop** (retrieve, then a 1.5B instruct answer).
- **Multilingual page retrieval** (card lists many languages).

## Not a good fit for

- **Matching the card’s BF16 printed scores.** T4 uses float16 + SDPA, not BF16 + FlashAttention-2.
- **Text-only MiniLM/EmbeddingGemma search.** Those notebooks do not embed page images.
- **Late-interaction MaxSim.** Use [pplx-embed-v2 late](pplx-embed-v2-late-0.6b-page-search.md).

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA (no BF16 compute, no FlashAttention-2). Hub file is about 3.1 GB of BF16; on T4 the tower sits around 3.5 GB in float16. Encoding a 1,120 px page should stay well inside 16 GB.

## What the notebook demos

- Embed PDF page images and text.
- FAISS retrieve.
- Short answer from Qwen2.5-1.5B-Instruct.

## Links

- Model card: [nvidia/Nemotron-3.5-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16)
- Blog: [Nemotron retriever](https://huggingface.co/blog/nvidia/nemotron-retriever-world-class-agentic-retrieval)
- Notebook: [`colab/Nemotron-3.5-Embed-1B-visual-rag.ipynb`](../colab/Nemotron-3.5-Embed-1B-visual-rag.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Nemotron-3.5-Embed-1B-visual-rag.ipynb)
- Related: [EmbeddingGemma 2](embeddinggemma-2-inference-similarity.md), [pplx-embed-v2 late](pplx-embed-v2-late-0.6b-page-search.md)
