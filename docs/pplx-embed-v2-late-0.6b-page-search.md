# pplx-embed-v2 late 0.6B page search

[`perplexity-ai/pplx-embed-v2-late-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b) is Perplexity’s MIT late-interaction (ColBERT-style MaxSim) embedder for search over page images with no OCR step. Text and images must be encoded in separate calls. The query/document marker belongs at the start of the text (`[Q] ` / `[D] `).

## Use cases

- **Search page screenshots** without running OCR first.
- **Late-interaction ranking** (MaxSim) instead of a single dense vector.
- **A 1.2 GB MIT T4 path** for visual document search.

## Not a good fit for

- **Mixed text+image batches.** Encode them in separate calls.
- **PyLate’s second-position marker.** Wrong format for this Sentence Transformers export.
- **Contextual overlapping-chunk RAG.** Use [pplx-embed-v2 context](pplx-embed-v2-context-0.6b-contextual-rag.md).

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA. About 1.2 GB in float16; encoding a few page images peaks at roughly 3–5 GB.

## What the notebook demos

- Encode page images (no OCR).
- Late-interaction search with `[Q]` / `[D]` markers.

## Links

- Model card: [perplexity-ai/pplx-embed-v2-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b)
- Notebook: [`colab/pplx-embed-v2-late-0.6b-page-search.ipynb`](../colab/pplx-embed-v2-late-0.6b-page-search.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/pplx-embed-v2-late-0.6b-page-search.ipynb)
- Related: [pplx-embed-v2 context](pplx-embed-v2-context-0.6b-contextual-rag.md), [Nemotron Embed visual RAG](Nemotron-3.5-Embed-1B-visual-rag.md)
