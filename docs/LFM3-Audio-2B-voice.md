# LFM3-Audio-2B voice

[`LiquidAI/LFM3-Audio-2B`](https://huggingface.co/LiquidAI/LFM3-Audio-2B) is Liquid AI’s ~2.00B audio model (LFM Open License v1.0) for ASR, short TTS, and audio understanding. It needs `transformers>=5.20.2` and `trust_remote_code`.

## Use cases

- **Short local ASR** without a dedicated Whisper stack.
- **Short TTS** with the card’s audio sampler.
- **Audio understanding** (describe or answer questions about a clip).

## Not a good fit for

- **Long-form transcription.** Keep clips to a few seconds and `max_new_tokens` small.
- **Omni image+video+audio in one smoke.** Use [LFM3-Omni-2B](LFM3-Omni-2B.md).
- **Keyword-biased ASR / speech translation.** Use [Granite Speech](granite-speech-5.0-2b-asr-translation.md).

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 with SDPA. About 4.0 GB of backbone weights plus LiquidCodec.

## What the notebook demos

- Load LFM3-Audio-2B.
- ASR.
- Short TTS.
- Audio understanding.

## Links

- Model card: [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B)
- Source post: [x.com/liquidai](https://x.com/liquidai/status/2115128329003303107)
- Blog: [LFM3](https://www.liquid.ai/blog/lfm3)
- Notebook: [`colab/LFM3-Audio-2B-voice.ipynb`](../colab/LFM3-Audio-2B-voice.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LFM3-Audio-2B-voice.ipynb)
- Related: [LFM3-Omni-2B](LFM3-Omni-2B.md), [AIBackends Whisper](AIBackends-Whisper-audio-transcription.md)
