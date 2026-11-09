# Nemotron-Labs Unified ASR 0.6B multilingual

[`nvidia/nemotron-labs-unified-asr-0.6b`](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b) is NVIDIA’s ~0.6B FastConformer-RNNT (OpenMDW-1.1) for offline ASR in 25 European languages with a language-ID prompt (`target_lang="de"` or `"auto"`), batching, and word timestamps.

## Use cases

- **Offline European-language ASR** with an explicit or auto language-id prompt.
- **Word timestamps** without a separate aligner.
- **Batched files** on a T4.

## Not a good fit for

- **Streaming ASR or speech translation.** Not covered.
- **PyPI NeMo 3.0.0.** That release breaks `target_lang`; this notebook pins NeMo to a `main` commit.

## Running it on a free Colab T4

Pick a T4 GPU. Float32. About 2.5 GB `.nemo`. Short clips should need roughly 3–4 GB of VRAM (estimate from checkpoint size, not a measured Colab number).

## What the notebook demos

- Language-ID prompt (`de` / `auto`).
- Batching.
- Word timestamps.

## Links

- Model card: [nvidia/nemotron-labs-unified-asr-0.6b](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b)
- Source post: [x.com/aitrackerbot](https://x.com/aitrackerbot/status/2115106347511423216)
- License: [OpenMDW-1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1)
- Notebook: [`colab/Nemotron-Labs-Unified-ASR-0.6B-multilingual.ipynb`](../colab/Nemotron-Labs-Unified-ASR-0.6B-multilingual.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Nemotron-Labs-Unified-ASR-0.6B-multilingual.ipynb)
- Related: [Granite Speech](granite-speech-5.0-2b-asr-translation.md), [Whisper](AIBackends-Whisper-audio-transcription.md)
