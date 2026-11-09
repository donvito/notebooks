# pplx-embed-v2 context 0.6B contextual RAG

[`perplexity-ai/pplx-embed-v2-context-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b) is Perplexity’s 0.6B contextual chunk embedder for text and visual documents. The Hub card lists **no license tag** (the late 0.6B sibling is MIT). Queries use `encode_queries`; chunks use `encode` and must be non-overlapping. Shares the 1024-dim MRL prefix of the 9B contextual model. Distinct from the late-interaction page-search notebook (ColBERT-style MaxSim, MIT).

## Use cases

- **Contextual RAG chunks** that see surrounding document context, not isolated sentences.
- **Text and visual documents** with the same 0.6B encoder.
- **A 1.2 GB T4 embedder** that matches the 9B model’s 1024-dim MRL prefix.

## Not a good fit for

- **Overlapping chunks.** The card requires non-overlapping chunks.
- **ColBERT MaxSim page search.** Use [pplx-embed-v2 late](pplx-embed-v2-late-0.6b-page-search.md).
- **Assuming MIT.** There is no license tag on this card; review it before any use beyond the demo.

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA, `transformers>=5.4.0`, `trust_remote_code`. Hub file is ~2.4 GB of float32; about 1.2 GB in float16. A short text encode should stay around 2–4 GB. Image encodes keep `max_pixels` modest.

## What the notebook demos

- Contextual chunk embeddings over text.
- Visual document chunks.
- Query vs chunk encode APIs.

## Links

- Model card: [perplexity-ai/pplx-embed-v2-context-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b)
- Source post: [x.com/tomaarsen](https://x.com/tomaarsen/status/2117618555210641590)
- Notebook: [`colab/pplx-embed-v2-context-0.6b-contextual-rag.ipynb`](../colab/pplx-embed-v2-context-0.6b-contextual-rag.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/pplx-embed-v2-context-0.6b-contextual-rag.ipynb)
- Related: [pplx-embed-v2 late](pplx-embed-v2-late-0.6b-page-search.md)
