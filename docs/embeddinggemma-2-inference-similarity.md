# EmbeddingGemma 2 similarity and search

[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) is Google DeepMind's Apache 2.0 multimodal embedding model (740M total: a 270M text model plus vision and audio encoders) that maps text, images, video, and audio into one 768-dimensional space. This notebook is text-only with `sentence-transformers`. It drops the vision and audio encoders to save RAM.

## Use cases

- **FAQ or doc search.** Encode queries with `SearchQuery` prefixes and documents with `Document` prefixes, then cosine-rank.
- **Inspect 768-dim vectors** from the text encoder before you stand up an index.
- **A small in-notebook semantic search** without a vector database.

## Not a good fit for

- **Image, audio, or video embeddings.** This notebook skips those encoders. For page images see [`Nemotron-3.5-Embed-1B-visual-rag.ipynb`](../colab/Nemotron-3.5-Embed-1B-visual-rag.ipynb) or the pplx-embed notebooks.
- **Sparse inverted-index retrieval.** See [`SlideOn-sparse-retrieval.ipynb`](../colab/SlideOn-sparse-retrieval.ipynb).
- **A tiny MiniLM path.** See [AIBackends MiniLM](AIBackends-MiniLM-embeddings-similarity.md).

## Running it on a free Colab T4

A T4 is fastest. A CPU runtime also works. First load downloads the full checkpoint (~1.5 GB) even in text-only mode. Restart after upgrading transformers or you get `Transformers does not recognize this architecture` for `embedding_gemma2`.

## What the notebook demos

- Load a text-only EmbeddingGemma 2.
- Encode with task prompts.
- Rank documents by cosine similarity.
- Semantic search over a small FAQ.

## Links

- Model card: [EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) · [Hub](https://huggingface.co/google/embeddinggemma-2)
- Announcement: [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)
- Notebook: [`colab/embeddinggemma-2-inference-similarity.ipynb`](../colab/embeddinggemma-2-inference-similarity.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/embeddinggemma-2-inference-similarity.ipynb)
- Related: [AIBackends MiniLM](AIBackends-MiniLM-embeddings-similarity.md), [`SlideOn-sparse-retrieval.ipynb`](../colab/SlideOn-sparse-retrieval.ipynb)
