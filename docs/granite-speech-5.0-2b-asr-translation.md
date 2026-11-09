# Granite Speech 5.0 2B ASR and translation

[`ibm-granite/granite-speech-5.0-2b`](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) is IBM’s ~2.3B speech model (Apache 2.0) for keyword-biased multilingual ASR and speech translation (X→En / En→X). Languages on the card include English, French, German, Spanish, Portuguese, and Japanese.

## Use cases

- **Keyword-biased ASR** when domain terms matter more than a generic transcript.
- **Speech translation** without a separate MT model (X→En or En→X).
- **An Apache-2.0 complement** to the Whisper aibackends notebook, which is plain faster-whisper ASR plus PII redaction.

## Not a good fit for

- **Speaker diarization.** Pair with [`Nemotron-3-Diarization.ipynb`](../colab/Nemotron-3-Diarization.ipynb).
- **Long unsegmented audio.** Split into segments of up to about 2 minutes.
- **Text chat.** That is [Granite 4.2 3B](granite-4.2-3b-chat-tools.md).

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 on T4 (bfloat16 on L4/A100). About 2.3B parameters / 4.6 GB of weights. Decode audio in segments of up to about 2 minutes.

## What the notebook demos

- Load Granite Speech 5.0 2B.
- Keyword-biased multilingual ASR.
- Speech translation X→En and En→X.
- Segment-length caveat.

## Links

- Model card: [ibm-granite/granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b)
- Notebook: [`colab/granite-speech-5.0-2b-asr-translation.ipynb`](../colab/granite-speech-5.0-2b-asr-translation.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/granite-speech-5.0-2b-asr-translation.ipynb)
- Related: [AIBackends Whisper](AIBackends-Whisper-audio-transcription.md), [Granite 4.2 3B](granite-4.2-3b-chat-tools.md)
