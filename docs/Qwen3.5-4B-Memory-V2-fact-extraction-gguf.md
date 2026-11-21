# Qwen3.5-4B-Memory-V2 fact extraction (GGUF)

[`IMLIULIAN/qwen3.5-4b-memory-v2`](https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2) is an Apache-2.0 LoRA fine-tune (`r=16`, `alpha=32`) of [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) for two local AI memory-pipeline jobs: **structured fact extraction** (strict JSON, segment evidence IDs `s0, s1, …`, record dates kept separate from event effective dates) and **pre-write fact admission / revision** (per-slot `pass` / `revise` / `drop` / `hold`). The Hub ships GGUF files (**Q8_0 4.48 GB recommended**, Q4_K_M 2.71 GB) plus the LoRA adapter — there is no merged Transformers checkpoint. Runtime on the card is Ollama or llama.cpp; this notebook uses a prebuilt CUDA `llama-cpp-python` wheel.

## Use cases

- **Extract grounded atomic facts from a conversation or note** when each claim must cite supplied segment IDs and keep the record timestamp off the event’s effective date. The card’s extraction schema is `facts[]` with `text`, `type` (`world` | `experience` | `observation`), `occurred_start` / `occurred_end`, `where`, `entities`, and `evidence`.
- **Audit candidate facts before they are written to a local memory store.** Admission returns, per candidate ID, `decision`, `evidence`, and `replacement`. The GitHub usage guide: V2 is optimized for `pass` and faithful `revise`; when a candidate contradicts a short evidence segment it often prefers `revise` with a grounded replacement rather than `drop`.
- **A T4-sized smoke of the GGUF path** with the official system prompts (`prompts/fact_extraction_system.txt`, `prompts/fact_admission_system.txt`) and the public synthetic cases in [`examples/public_synthetic_eval.jsonl`](https://github.com/lifdjgfndgn/qwen3.5-4b-memory-v2/blob/main/examples/public_synthetic_eval.jsonl).

## Not a good fit for

- **General-purpose chat, creative writing, or general math/coding benches.** The GitHub usage guide: this is a specialized backend model for those two memory jobs.
- **A rerun of the card’s 50-case frozen suite.** The Hub table (extraction schema 100%, 0 record-date leaks, admission slot accuracy 97.3% on Q8_0, and the rest) is the authors’ unconstrained-JSON eval, not this notebook.
- **Merging the LoRA in Transformers on this T4 walkthrough.** The Hub has GGUF + `lora/adapter_model.safetensors` (81.0 MB), not a full merged weight dump. The adapter is for researchers who want to merge onto `Qwen/Qwen3.5-4B` themselves.
- **Compiling llama.cpp from source in Colab.** Source builds disconnect the free T4 runtime. Use the pinned prebuilt CUDA wheel.
- **Leaving thinking on.** Card / Modelfile: `think=false`. Thinking mode wraps the JSON in a reasoning trace.

## Running it on a free Colab T4

Pick a T4 GPU. Install **`llama-cpp-python` 0.3.37** from the CUDA 12.1 wheel URL (plus `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12`). Do not compile. `qwen35` is in llama.cpp from **b7990**; 0.3.17 is the first Python release with Qwen 3.5 support, and 0.3.37 vendors llama.cpp `@a956a852`. Load **Q8_0 (4.48 GB)** with `n_gpu_layers=-1`, `n_ctx=16384`, `flash_attn=False` (T4 has no FlashAttention-2). GitHub reports **~5.2 GB VRAM** for Q8_0 at that context (~3.4 GB for Q4_K_M). Sampler: `temperature=0`, `presence_penalty=0`, `top_p=0.95`, `top_k=20`. Prompts and the public JSONL are fetched from GitHub raw (jsDelivr fallback, then the same files embedded in the notebook). Peak VRAM is printed from `nvidia-smi` because llama.cpp does not go through `torch.cuda.memory_allocated`.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Pinned CUDA `llama-cpp-python` 0.3.37 wheel (no source compile).
- Official prompts + `public_synthetic_eval.jsonl` (download with fallback).
- Q8_0 GGUF load; print `general.architecture` and require `qwen35`.
- Fact extraction from public case `pub-ext-02` (two-turn cache-timeout conversation).
- Fact admission on `pub-adm-01` (expected `pass`) and `pub-adm-02` (expected `revise`; the sample_io.md connection-pool example).
- `nvidia-smi` used/free/total after load, after generate, and after cleanup.

## Links

- Model card: [IMLIULIAN/qwen3.5-4b-memory-v2](https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2)
- Prompts, Modelfiles, public cases: [lifdjgfndgn/qwen3.5-4b-memory-v2](https://github.com/lifdjgfndgn/qwen3.5-4b-memory-v2)
- Source post: [x.com/Wrongvqxi](https://x.com/Wrongvqxi/status/2123784817078866265)
- Base model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/Qwen3.5-4B-Memory-V2-fact-extraction-gguf.ipynb`](../colab/Qwen3.5-4B-Memory-V2-fact-extraction-gguf.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Qwen3.5-4B-Memory-V2-fact-extraction-gguf.ipynb)
- Related: [Qwen3.5-4B decision model: email triage](qwen3_5_4b_decision_email_triage.md), [Ternary Bonsai 2 4B](Ternary-Bonsai-2-4B-llamacpp.md), [GLiNER2.5 extraction](AIBackends-GLiNER2_5-extraction.md)
