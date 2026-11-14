# Fluid Speech Pro multilingual ASR

[`FluidVoice-Labs/fluid-speech-pro`](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) is FluidVoice Labs’ 0.63B FastConformer-TDT (CC BY 4.0) for offline speech-to-text in 25 European languages. It is built on NVIDIA Parakeet TDT 0.6B v3, detects the language automatically (no prompt), and returns punctuation and capitalization. Word/segment timestamps and optional clean mode are documented on the NeMo path, not on the 🤗 Transformers pipeline this notebook uses.

## Use cases

- **Dictation, meeting notes, captions and subtitles** in the 25 European languages on the card, when you want punctuation in the transcript without a second model.
- **Call and media analytics / transcription services** that need a small open CC BY 4.0 checkpoint rather than a 1B+ ASR model.
- **Clips whose language is unknown**, because the card’s Transformers call does not take a language-ID prompt.

## Not a good fit for

- **Non-European languages.** The card lists 25 European languages only (en, bg, hr, cs, da, nl, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk).
- **Word timestamps or clean mode** on this Transformers notebook. The card documents timestamps on NeMo, and clean mode on NeMo / ONNX / MLX.
- **Speaker diarization or overlapping talk.** Pair with [`Nemotron-3-Diarization.ipynb`](../colab/Nemotron-3-Diarization.ipynb). The card says the model makes more mistakes on overlapping speakers and heavy noise.

## Running it on a free Colab T4

Pick a T4 GPU. The card needs Transformers 5.22 or newer. Saved weights are float32 (`model.safetensors` about 2.5 GB); a free T4 has no native bfloat16, so the notebook loads **float16** (float32 if a transcript looks degraded). Short clips should need a few GB of VRAM — that is an estimate from the checkpoint size, not a measured Colab number.

## What the notebook demos

- The card’s 🤗 Transformers `pipeline("automatic-speech-recognition", model="FluidVoice-Labs/fluid-speech-pro")`.
- An English LibriSpeech-style clip with punctuation.
- One German FLEURS clip with no language prompt.
- A longer concatenated clip, with seconds of audio vs wall time.
- Why timestamps are skipped (NeMo-only API; heavier Colab install).

## Links

- Model card: [FluidVoice-Labs/fluid-speech-pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro)
- Source post: [x.com/FluidVoiceApp](https://x.com/FluidVoiceApp/status/2121308183314796772)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — credit **Fluid Speech Pro by FluidVoice**
- Notebook: [`colab/Fluid-Speech-Pro-multilingual-asr.ipynb`](../colab/Fluid-Speech-Pro-multilingual-asr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Fluid-Speech-Pro-multilingual-asr.ipynb)
- Related: [Nemotron-Labs Unified ASR](Nemotron-Labs-Unified-ASR-0.6B-multilingual.md), [Granite Speech](granite-speech-5.0-2b-asr-translation.md), [Whisper](AIBackends-Whisper-audio-transcription.md)
