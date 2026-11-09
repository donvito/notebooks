# AIBackends Whisper audio transcription

This notebook transcribes audio locally with aibackends `AudioIngestor` + `WhisperTranscriber` (`aibackends[audio]`, powered by faster-whisper). It then chains the transcript into GLiNER PII redaction so audio can sit in a larger privacy pipeline. The Whisper size is not pinned in the notebook; `device="auto"` uses CUDA on a GPU runtime and CPU otherwise.

## Use cases

- **Meeting or voicemail transcripts on-box.** Transcribe an MP3 without sending audio to a hosted ASR API.
- **Redact after transcribe.** Pipe the transcript into PII redaction before you store or summarize it.
- **Front end for speaker-unaware ASR.** Pair later with a diarization notebook if you need speaker IDs.

## Not a good fit for

- **Speaker labels.** Whisper here does not diarize. See [`Nemotron-3-Diarization.ipynb`](../colab/Nemotron-3-Diarization.ipynb).
- **Speech translation or keyword biasing.** See [`granite-speech-5.0-2b-asr-translation.ipynb`](../colab/granite-speech-5.0-2b-asr-translation.ipynb).
- **Streaming ASR.** This workflow is file ingest, not a streaming decoder.

## Running it on a free Colab T4

It runs on a free CPU runtime. A T4 is faster and is detected automatically. The notebook does not document a Whisper checkpoint size or VRAM figure.

## What the notebook demos

- Transcribe an MP3.
- Transcribe, then redact PII in one pipeline.

## Links

- Notebook: [`colab/AIBackends-Whisper-audio-transcription.ipynb`](../colab/AIBackends-Whisper-audio-transcription.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-Whisper-audio-transcription.ipynb)
- Related: [GLiNER PII redaction](AIBackends-GLiNER-PII-redaction.md), [`granite-speech-5.0-2b-asr-translation.ipynb`](../colab/granite-speech-5.0-2b-asr-translation.ipynb), [`Nemotron-3-Diarization.ipynb`](../colab/Nemotron-3-Diarization.ipynb)
