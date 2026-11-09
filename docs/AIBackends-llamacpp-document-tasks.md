# AIBackends llama.cpp document tasks

This notebook runs structured extraction, classification, and summarization with local GGUF models through the aibackends `LLAMACPP` runtime. It uses Gemma 4 E2B (`unsloth/gemma-4-E2B-it-GGUF`, Q4_K_M, Apache 2.0). Tasks accept a file path or a string and return validated Pydantic models.

## Use cases

- **Invoice fields from text.** `ExtractInvoiceTask` returns a validated `InvoiceOutput`.
- **Document classification and summaries.** `ClassifyTask` and `SummarizeTask` share the same loaded GGUF.
- **Your own schema.** `ExtractTask` constrains output to any Pydantic model’s JSON schema.
- **Sales-call and video-ad reports.** Built-in `SalesCallReport` / `VideoAdReport` schemas with explicit field guidance so small local models follow them more reliably.

## Not a good fit for

- **Image receipts.** Use [vision receipt OCR](AIBackends-vision-receipt-OCR.md) for photos.
- **PII-first PDF pipelines.** Use [PDF redact and extract](AIBackends-PDF-redact-extract-workflows.md) when you need redaction before the LLM.
- **Hugging Face Transformers instead of GGUF.** See [Transformers runtime tasks](AIBackends-Transformers-runtime-tasks.md).

## Running it on a free Colab T4

It runs on a free CPU runtime. A T4 is faster and is detected automatically. `GEMMA4_E2B` downloads Q4_K_M from the Hub on first use. `device=DEVICE` offloads all layers to CUDA on a GPU runtime.

## What the notebook demos

- Invoice extraction (`ExtractInvoiceTask`).
- Document classification (`ClassifyTask`).
- Summarization (`SummarizeTask`).
- Custom schema extraction (`ExtractTask`).
- Sales-call and video-ad reports.
- The same tasks from the CLI.

## Links

- Model: [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
- Notebook: [`colab/AIBackends-llamacpp-document-tasks.ipynb`](../colab/AIBackends-llamacpp-document-tasks.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-llamacpp-document-tasks.ipynb)
- Related: [Transformers runtime tasks](AIBackends-Transformers-runtime-tasks.md), [PDF redact and extract](AIBackends-PDF-redact-extract-workflows.md), [vision receipt OCR](AIBackends-vision-receipt-OCR.md)
