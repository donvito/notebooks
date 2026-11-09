# Nemotron-3 Diarization

[`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) is NVIDIA’s ~99M speaker-diarization model (OpenMDW-1.1) for podcast/meeting timelines and RTTM, up to 8 speakers. It does not transcribe.

## Use cases

- **Speaker timelines** for meetings or podcasts.
- **RTTM export** to pair with any ASR (Whisper, Granite Speech, Unified ASR).
- **A ≪1 GB model** that also runs on CPU.

## Not a good fit for

- **ASR.** Complements [Whisper](AIBackends-Whisper-audio-transcription.md), which has no speaker IDs.
- **More than 8 speakers**, per the card.

## Running it on a free Colab T4

T4 or CPU. Float16 on T4, float32 on CPU. Weights ≪1 GB; a short clip stays well under 4 GB. No FlashAttention-2.

## What the notebook demos

- Podcast/meeting speaker timeline.
- RTTM output.

## Links

- Model card: [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- License: [OpenMDW-1.1](https://openmdw.ai/license/1-1/)
- Notebook: [`colab/Nemotron-3-Diarization.ipynb`](../colab/Nemotron-3-Diarization.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Nemotron-3-Diarization.ipynb)
- Related: [Whisper](AIBackends-Whisper-audio-transcription.md), [Granite Speech](granite-speech-5.0-2b-asr-translation.md)
