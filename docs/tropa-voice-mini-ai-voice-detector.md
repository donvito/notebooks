# tropa-voice-mini AI voice detector

[`wasitaigeneratedcom/ai-voice-detector-small`](https://huggingface.co/wasitaigeneratedcom/ai-voice-detector-small) is wasitaigenerated’s open **tropa-voice-mini** detector (Apache-2.0). It is a fine-tuned [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m) with an attention head. Load it with the repo’s `detector.py`; `det(path)` returns a calibrated `ai_score`, a verdict (`ai` / `uncertain` / `human`), and 6 s / 3 s hop segment scores. It does not transcribe.

## Use cases

- **Screen a voicemail, call recording, or uploaded clip** for commercial TTS and voice clones before a human listens.
- **Score a posted clip** before you amplify it. The notebook uses a few files from the public [Commercial-Voices-2026-mini](https://huggingface.co/datasets/wasitaigeneratedcom/commercial-voices-2026-mini) set the card points at.
- **Read the per-window timeline** when only part of a file may be synthetic (6 s windows, 3 s hop).
- **Label speech you just generated** with a local TTS (the notebook optionally scores an `espeak-ng` clip).

## Not a good fit for

- **A courtroom or forensic sign-off.** The card’s numbers are the authors’ own runs; keep a human in the loop.
- **Clips shorter than about 2 s**, heavy background music, or phone-quality audio — the card lists these as harder for every detector it compared.
- **Transcription or diarization.** Use [Whisper](AIBackends-Whisper-audio-transcription.md) or [Nemotron-3 Diarization](Nemotron-3-Diarization.md).
- **Matching the hosted tropa-voice API on every bench.** The card says that larger hosted system is ahead on four of five reported sets.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). The card also calls the model CPU-friendly (~0.06 s for a 30 s clip on GPU vs ~10 s on 4 CPU cores). Weights are one **1.26 GB** F32 `model.safetensors`. This notebook keeps F32, as `detector.py` does. Peak VRAM for the short demos should be a few GB on ~15 GB. No FlashAttention-2, no extra CUDA kernels. Public weights; `HF_TOKEN` is optional. ffmpeg is needed for mp3/m4a/mp4 (FLAC works through soundfile).

## What the notebook demos

- Load tropa-voice-mini via the Hub `detector.py` (`VoiceDetector.from_pretrained`).
- Score a few human and AI clips from Commercial-Voices-2026-mini (skipped if a file is missing).
- Print the per-segment timeline and `serving_head.json` thresholds.
- Optionally score a clip synthesised with `espeak-ng` (same helper as the Granite Speech and Nemotron-3 Diarization notebooks).

## Links

- Model card: [wasitaigeneratedcom/ai-voice-detector-small](https://huggingface.co/wasitaigeneratedcom/ai-voice-detector-small)
- Test set: [commercial-voices-2026-mini](https://huggingface.co/datasets/wasitaigeneratedcom/commercial-voices-2026-mini)
- Benchmark code: [voice-deepfake-benchmark](https://github.com/wasitaigenerated/voice-deepfake-benchmark)
- Source post: [x.com/wiagenerated](https://x.com/wiagenerated/status/2119776813237096523)
- Notebook: [`colab/tropa-voice-mini-ai-voice-detector.ipynb`](../colab/tropa-voice-mini-ai-voice-detector.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/tropa-voice-mini-ai-voice-detector.ipynb)
- Related: [HaloGuard2-Stream output guard](HaloGuard2-Stream-0.8B-streaming-output-guard.md), [Whisper transcription](AIBackends-Whisper-audio-transcription.md), [Phonon 31M TTS](phonon-31m-en-streaming-tts.md)
