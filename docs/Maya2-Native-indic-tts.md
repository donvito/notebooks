# Maya 2 Native Indic TTS

[`maya-research/maya2-native`](https://huggingface.co/maya-research/maya2-native) is Maya Research’s 2.50B streaming 24 kHz TTS (Apache 2.0) for 11 Indic languages plus English, with 20 built-in voices and zero-shot cloning. The published package is still `maya3` / `Maya3TTS`. Not gated.

## Use cases

- **Indic + English product voice** from one 2.5B checkpoint (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, English).
- **Streaming 24 kHz playback** with `max_seats=1`.
- **Zero-shot cloning of a voice you have rights to.**

## Not a good fit for

- **Tiny English-only CPU TTS.** Use [Phonon 31M](phonon-31m-en-streaming-tts.md).
- **A100-scale multi-speaker drama.** Use [Drama 3](Drama-3-tts-voice-clone.md).
- **Calling the package `maya2`.** The published API is still `maya3`.

## Running it on a free Colab T4

Pick a T4 GPU. Float16 backbone (T4 has no native BF16), gather attention (not Triton), backbone CUDA graphs off, `max_seats=1`, 1024-token KV cache. Documented single-seat load peak ~9.2 GB. If a waveform is NaN/Inf, the overflow-prone depth stack is rebuilt in float32.

## What the notebook demos

- Load Maya 2 Native with T4-safe flags.
- Streaming Indic + English TTS.
- Built-in voices.
- Zero-shot cloning.

## Links

- Model card: [maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native)
- Code: [MayaResearch/maya2-native](https://github.com/MayaResearch/maya2-native)
- Notebook: [`colab/Maya2-Native-indic-tts.ipynb`](../colab/Maya2-Native-indic-tts.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Maya2-Native-indic-tts.ipynb)
- Related: [Phonon 31M](phonon-31m-en-streaming-tts.md), [Drama 3](Drama-3-tts-voice-clone.md)
