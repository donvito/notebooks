# AIBackends custom tasks, pipelines, and batch

This notebook shows how to extend aibackends: register a custom `BaseTask`, assemble a `Pipeline` from ingest / enrich / validate steps, and fan a workflow out with `run_batch` or `run_async`. The custom pipeline uses Gemma 4 E2B on llama.cpp (`unsloth/gemma-4-E2B-it-GGUF`, Q4_K_M) for schema extraction. Batch processing mirrors the built-in `SalesCallAnalyser` (ingest, transcribe if audio, redact PII, analyse, validate).

## Use cases

- **Register your own task.** Subclass `BaseTask`, register it, and look it up by name like a built-in, including tasks that need no model.
- **Compose a typed pipeline.** `FileIngestor` → `LLMAnalyser` → `PydanticValidator` keeps file ingest, LLM extract, and schema validation in one graph.
- **Batch many inputs with bounded concurrency.** `run_batch` plus `on_error="collect"` returns failures alongside results instead of aborting the job.
- **Async handlers.** `run_async` fits a web worker that already awaits other aibackends tasks.

## Not a good fit for

- **A first look at built-in tasks.** Start with [llama.cpp document tasks](AIBackends-llamacpp-document-tasks.md) or [PDF workflows](AIBackends-PDF-redact-extract-workflows.md).
- **Distributed queues.** `run_batch` is in-process concurrency, not a cluster scheduler.

## Running it on a free Colab T4

It runs on a free CPU runtime. A T4 is faster and is detected automatically. The custom-task demo needs no model. The pipeline and batch sections download Gemma 4 E2B Q4_K_M on first use. `device=DEVICE` flows into every LLM step.

## What the notebook demos

- A custom task (`task_interface.py`).
- A custom pipeline (`custom_pipeline.py`).
- Batch processing (`batch_processing.py`).
- Async `run_async`.

## Links

- Model: [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
- Notebook: [`colab/AIBackends-custom-tasks-pipelines-batch.ipynb`](../colab/AIBackends-custom-tasks-pipelines-batch.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-custom-tasks-pipelines-batch.ipynb)
- Related: [llama.cpp document tasks](AIBackends-llamacpp-document-tasks.md), [PDF redact and extract](AIBackends-PDF-redact-extract-workflows.md), [Whisper transcription](AIBackends-Whisper-audio-transcription.md)
