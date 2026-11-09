# AIBackends + GLiNER2.5 extraction

Zero-shot extraction with Fastino's [GLiNER2.5](https://huggingface.co/fastino/gliner2.5-base-v1) through `aibackends[extraction]` (aibackends v0.7.0). You name entity types, classification tasks, or relations at call time; the encoder finds them without fine-tuning. Default checkpoint is `fastino/gliner2.5-base-v1` (Apache 2.0, English). Variants: `small`, `base`, and multilingual `fastino/gliner2.5-multi-v1`.

## Use cases

- **Ad-hoc NER for any label list.** Pull products, orgs, dates, or domain types from tickets and docs without a training set.
- **Span attributes.** Classify each extracted span (for example product + sentiment) in the same pass.
- **Zero-shot text classification.** Single-label, multi-label, and constrained classification from a `tasks` map — useful for intent or topic tags next to extraction.
- **Knowledge-graph sketches.** Entities and relations in one pass for lightweight KG construction from prose.
- **Long documents.** `long_document=True` chunk-and-merges when text exceeds the model context.

## Not a good fit for

- **Open-ended generation.** This is span extraction and classification, not a chat model.
- **Image or audio input.** Text only. For page images, OCR first (LightOnOCR / MiniCPM-OCR notebooks).
- **PII-specific redaction UX.** Use the dedicated [PII notebook](AIBackends-GLiNER-PII-redaction.md) when you need placeholder maps and the NVIDIA PII label set.

## Running it on a free Colab T4

Runs on a **free CPU runtime**; a T4 is optional. First run of the default variant downloads ~800 MB of weights. Extra: `aibackends[extraction]` pulls `gliner2[local]` and `protobuf`. Default confidence `threshold` is 0.5.

## What the notebook demos

- Model variants (`small` / `base` / `multi`).
- Load once, reuse everywhere.
- Zero-shot entity extraction, including label descriptions.
- Structured entity attributes.
- Batching, thresholds, and long documents.
- Text classification (single, multi-label, constrained).
- Knowledge-graph extraction.
- Async calls and CLI.

## Links

- Model cards: [gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1) · [gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1)
- Notebook: [`colab/AIBackends-GLiNER2_5-extraction.ipynb`](../colab/AIBackends-GLiNER2_5-extraction.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-GLiNER2_5-extraction.ipynb)
- Related: [advanced extraction](AIBackends-GLiNER2_5-advanced-extraction.md), [`GLiNER2_5-Decide-classification.ipynb`](../colab/GLiNER2_5-Decide-classification.ipynb)
