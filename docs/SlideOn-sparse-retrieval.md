# SlideOn sparse retrieval

[`lightonai/SlideOn`](https://huggingface.co/lightonai/SlideOn) is LightOn’s 174M English sparse retriever (Apache 2.0). This notebook builds a tiny inverted index, compares BM25 and MiniLM, and optionally prunes Matryoshka dimensions. Distinct from dense EmbeddingGemma 2 and late-interaction pplx-embed.

## Use cases

- **Inverted-index search** you can inspect as terms and weights, not opaque dense vectors.
- **A BM25 vs MiniLM vs SlideOn bake-off** on a small English corpus.
- **Matryoshka prune** when you want a smaller sparse vector.

## Not a good fit for

- **Page-image search without OCR.** Use pplx-embed late or Nemotron Embed.
- **Multilingual sparse retrieval.** This checkpoint is English.

## Running it on a free Colab T4

T4 or CPU. Float16 + SDPA. About 0.32 GB in float16.

## What the notebook demos

- Build a tiny sparse index.
- BM25 and MiniLM comparison.
- Optional Matryoshka prune.

## Links

- Model card: [lightonai/SlideOn](https://huggingface.co/lightonai/SlideOn)
- Blog: [SlideOn](https://huggingface.co/blog/lightonai/slideon-mslideon)
- Notebook: [`colab/SlideOn-sparse-retrieval.ipynb`](../colab/SlideOn-sparse-retrieval.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/SlideOn-sparse-retrieval.ipynb)
- Related: [EmbeddingGemma 2](embeddinggemma-2-inference-similarity.md), [AIBackends MiniLM](AIBackends-MiniLM-embeddings-similarity.md)
