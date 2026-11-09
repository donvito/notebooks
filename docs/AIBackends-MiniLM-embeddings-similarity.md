# AIBackends + MiniLM embeddings and similarity

This notebook generates sentence embeddings and ranks similar texts with `aibackends` on the Transformers runtime. It uses [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (22.7M, Apache 2.0, English BERT MiniLM). The workflow covers `EmbedTask`, `EmbeddingSimilarityWorkflow`, and a small FAQ search.

## Use cases

- **Ticket similarity.** Embed a new support message and rank it against recent tickets so agents can reuse an existing answer.
- **Tiny FAQ search.** Encode questions and documents, then cosine-rank hits without standing up a vector database.
- **Downstream clustering or retrieval.** `EmbedTask` returns plain float lists, so you can feed them into any vector math or index.

## Not a good fit for

- **Multilingual or domain-heavy retrieval.** MiniLM-L6 is a small English general-purpose encoder. For page images or contextual chunks, use the pplx-embed or Nemotron Embed notebooks.
- **Sparse inverted-index search.** See [`SlideOn-sparse-retrieval.ipynb`](../colab/SlideOn-sparse-retrieval.ipynb).
- **Long-document late interaction.** See [`pplx-embed-v2-late-0.6b-page-search.ipynb`](../colab/pplx-embed-v2-late-0.6b-page-search.ipynb).

## Running it on a free Colab T4

It runs on a free CPU runtime. A T4 is faster and is detected automatically. The notebook does not document a weight or VRAM figure; MiniLM-L6 is a small encoder and is intended to stay light.

## What the notebook demos

- Embed a single text (`EmbedTask`).
- Rank similar support tickets (`EmbeddingSimilarityWorkflow`).
- Semantic search over a small FAQ.

## Links

- Model card: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Notebook: [`colab/AIBackends-MiniLM-embeddings-similarity.ipynb`](../colab/AIBackends-MiniLM-embeddings-similarity.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-MiniLM-embeddings-similarity.ipynb)
- Related: [PDF redact and extract](AIBackends-PDF-redact-extract-workflows.md), [`embeddinggemma-2-inference-similarity.ipynb`](../colab/embeddinggemma-2-inference-similarity.ipynb), [`SlideOn-sparse-retrieval.ipynb`](../colab/SlideOn-sparse-retrieval.ipynb)
