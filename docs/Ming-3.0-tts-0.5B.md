# Ming-3.0-tts 0.5B

[`inclusionAI/Ming-3.0-tts-0.5B`](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) is inclusionAI’s multilingual TTS (MIT) with zero-shot cloning, Voice Design, and inline tags. Hub reports ~1.48B parameters under a 0.5B name. Languages include Chinese, English, German, Spanish, French, Korean, and Japanese.

## Use cases

- **Multilingual TTS with inline tags** on a MIT checkpoint that fits a T4 if you upcast generate to float32.
- **Zero-shot cloning** of a voice you have permission to use.
- **Voice Design** prompts when you want a described voice rather than a clip.

## Not a good fit for

- **Float16 generate on a T4.** fp16 LLM hidden states overflow and NaN the waveform even when DiT/VAE weights are already float32. The notebook runs the entire generate path in float32.
- **Cloning people without consent.**
- **Tiny English CPU TTS.** Use [Phonon](phonon-31m-en-streaming-tts.md).

## Running it on a free Colab T4

Pick a T4 GPU. Weights load as float16 (~3.0 GB), then the full generate path (LLM, DiT/flow, VAE, heads) runs in float32 + SDPA. Roughly 6 GB allocated after the upcast and 8–12 GB peak for a short clip. Autocast is off below compute capability 8.0. On L4/A100 the card’s bfloat16 path can be used.

## What the notebook demos

- Load Ming-3.0-tts with the T4 float32 generate path.
- Multilingual TTS.
- Zero-shot cloning.
- Voice Design and inline tags.

## Links

- Model card: [inclusionAI/Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)
- Notebook: [`colab/Ming-3.0-tts-0.5B.ipynb`](../colab/Ming-3.0-tts-0.5B.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Ming-3.0-tts-0.5B.ipynb)
- Related: [Phonon 31M](phonon-31m-en-streaming-tts.md), [Audio8](Audio8-TTS-Realistic-Air-voice-clone.md)
