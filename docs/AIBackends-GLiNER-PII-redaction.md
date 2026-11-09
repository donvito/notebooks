# AIBackends + GLiNER PII redaction

Local PII detection and redaction through `aibackends[pii]`, backed by NVIDIA's [nvidia/gliner-pii](https://huggingface.co/nvidia/gliner-pii) (GLiNER span classifier, NVIDIA Open Model License). The notebook wraps the one-call helper, a reusable `RedactPIITask` with custom labels, and batch redaction that loads the model once.

## Use cases

- **Redact support tickets or emails before they hit an LLM or a ticket store.** One call returns redacted text, every entity, and a placeholder-to-original map so you can restore values later if needed.
- **Custom label lists.** GLiNER is zero-shot, so you can keep only the entity types your policy cares about (names, account numbers, and so on) without retraining.
- **Batch document scrubbing.** `redact_text_batch.py` loads the model once and walks many strings, which fits nightly inbox or log-redaction jobs.
- **Privacy-first pipelines.** Pair with PDF ingest and a local LLM so only redacted text leaves the redaction step — see [PDF redact and extract workflows](AIBackends-PDF-redact-extract-workflows.md).

## Not a good fit for

- **Guaranteed compliance.** Span NER can miss or over-redact; the NVIDIA card recommends validation and human review for high-stakes HIPAA/GDPR workflows.
- **Scanned PDFs or images.** This notebook redacts text. For page images, use a vision OCR notebook first, then redact the transcript.
- **Generative rewriting.** It substitutes placeholders; it does not paraphrase around PII.

## Running it on a free Colab T4

Works on a **free CPU runtime**; a T4 is faster and is detected automatically. Exact weight size is not documented in the notebook. Device is cached per process so later calls reuse the GPU copy.

## What the notebook demos

- Load the GLiNER PII model (GPU when available).
- One-call `redact_pii`.
- A reusable task with custom labels (`redact_text.py`).
- Batch redaction with one model load (`redact_text_batch.py`).
- The same tasks from the CLI.

## Links

- Model card: [nvidia/gliner-pii](https://huggingface.co/nvidia/gliner-pii)
- Notebook: [`colab/AIBackends-GLiNER-PII-redaction.ipynb`](../colab/AIBackends-GLiNER-PII-redaction.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-GLiNER-PII-redaction.ipynb)
- Related: [PDF redact and extract](AIBackends-PDF-redact-extract-workflows.md), [Whisper then redact](AIBackends-Whisper-audio-transcription.md), [GLiNER2.5 extraction](AIBackends-GLiNER2_5-extraction.md)
