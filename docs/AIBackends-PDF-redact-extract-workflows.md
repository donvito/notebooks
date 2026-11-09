# AIBackends PDF redact and extract workflows

This notebook composes privacy-first document pipelines in `aibackends`: read a PDF, redact PII locally with GLiNER, then send only the redacted text to a local LLM. Generation uses Gemma 4 E2B through llama.cpp (`unsloth/gemma-4-E2B-it-GGUF`, Q4_K_M, Apache 2.0). It covers invoice extract, resume summarize, and resume-to-role matching.

## Use cases

- **Invoice PDFs.** Extract text, redact PII, then fill a structured invoice schema so vendors never send raw personal data to the LLM.
- **Hiring-manager summaries.** Redact a resume, then summarize it for a hiring manager without leaking phone numbers or emails.
- **Role matching.** Redact, summarize, then classify the candidate against open roles with `ClassifyTask` and label descriptions.

## Not a good fit for

- **Scanned page images with no text layer.** This path expects extractable PDF text. For photos of pages, use a vision OCR notebook first.
- **Guaranteed compliance.** GLiNER PII can miss or over-redact; keep a human review step for regulated pipelines.
- **Cloud document APIs.** Everything here is local aibackends tasks.

## Running it on a free Colab T4

It runs on a free CPU runtime. A T4 is faster and is detected automatically. `GEMMA4_E2B` downloads the Q4_K_M GGUF from the Hub on first use. Passing `device=DEVICE` offloads llama.cpp layers to CUDA when a GPU is present.

## What the notebook demos

- Invoice PDF: extract text, redact, then structured invoice JSON.
- Resume: redact, then a hiring-manager summary.
- Resume: redact, summarize, then match to open roles.

## Links

- Models: [nvidia/gliner-pii](https://huggingface.co/nvidia/gliner-pii) · [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
- Notebook: [`colab/AIBackends-PDF-redact-extract-workflows.ipynb`](../colab/AIBackends-PDF-redact-extract-workflows.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-PDF-redact-extract-workflows.ipynb)
- Related: [GLiNER PII redaction](AIBackends-GLiNER-PII-redaction.md), [llama.cpp document tasks](AIBackends-llamacpp-document-tasks.md), [custom tasks and batch](AIBackends-custom-tasks-pipelines-batch.md)
