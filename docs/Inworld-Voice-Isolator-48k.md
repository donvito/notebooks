# Inworld Voice Isolator v1 (48 kHz)

[`inworld/voice-isolator`](https://huggingface.co/inworld/voice-isolator) is Inworld's **Voice Isolator v1** (Apache-2.0). It ships two 48 kHz checkpoints that share one architecture: `voice-isolator-v1` (**64.0M**, `denoiser.safetensors` 256.3 MB) for recorded audio, and `voice-isolator-v1-realtime` (**58.2M**, `denoiser_realtime.safetensors` 232.8 MB) for live audio. Offline, every frame sees its whole 8 s chunk; long inputs render in 8 s chunks overlapping by 2 s. The realtime model is a causal stream with a trained look-ahead of 0, 20 or 40 ms (algorithmic latency 63 ms at 20 ms of look-ahead). Version 1 does denoising (noise suppression): it removes what surrounds the voice and restores nothing the recording lost. It is not a speaker selector.

Install is `pip install inworld-voice-isolation`. The Python API is `from inworld import voice_isolation` then `VoiceIsolator("inworld/voice-isolator")`, with `mode="streaming", lookahead_frames=2` for the realtime path.

## Use cases

- **Clean a recording before you play or publish it.** Interviews, voice notes, and dataset clips where the speaker's identity, level, timing, and non-verbal sounds (laughter, breaths, sighs) should survive.
- **Pull the voice out of mixed audio.** Subtract the isolated voice from the input to hear the removed background, or a backing-track residual for vocals/music separation.
- **A live stream or call.** The realtime checkpoint takes 48 kHz mono in pieces of any size; the card's snippet pushes 20 ms (960 samples) at a time with 20 ms of look-ahead.
- **A step before speech-to-text.** Isolate first, then send the voice to an ASR notebook. Ask `isolate(..., output_sample_rate=16000)` if the transcriber wants 16 kHz.

## Not a good fit for

- **Picking one speaker out of several.** The card: when two people speak at once, the louder voice is kept and one a few dB quieter is removed; voices at similar levels are usually both kept. Use [Nemotron-3 Diarization](Nemotron-3-Diarization.md) for who-spoke-when.
- **Restoring bandwidth or a bad microphone.** Version 1 does not invent missing band detail. Restoration is planned for version 2.
- **Transcribing or detecting synthetic speech.** Pair with [Rumble-2-Flash](Rumble-2-Flash-noisy-speech-to-text.md) / [Whisper](AIBackends-Whisper-audio-transcription.md) for words, or [tropa-voice-mini](tropa-voice-mini-ai-voice-detector.md) for AI-voice scoring.
- **A VIBE leaderboard run.** Inworld scores systems on an internal evaluation set that is not distributed.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). The card uses CUDA when available and CPU otherwise. Inworld tested Ampere, Hopper and Blackwell GPUs (A100, H100, B200, B300) and CPU; **smaller GPUs are not tested yet**. This notebook still runs on a T4 via CUDA.

Default `precision="auto"` is bfloat16 autocast on CUDA. A T4 (`sm_75`) has no native bfloat16, so the notebook passes `precision="fp16"`. The STFT, iSTFT and selective scan stay in float32. Do not install `inworld-voice-isolation[fast]` (CUDA Mamba kernels need Ampere+). Do not install `flash-attn`. First run downloads both checkpoints (~0.5 GB together). Public weights; `HF_TOKEN` is optional. Peak VRAM for the short demos should be a small fraction of ~15 GB — an estimate from the two shards, not a measured Colab run.

## What the notebook demos

- Install `inworld-voice-isolation`, keeping Colab's torch.
- Print `torch.cuda.is_available()`, GPU name, and free VRAM (CPU fallback).
- Load the offline and realtime checkpoints from `inworld/voice-isolator`.
- Offline isolation of the repo's [`sample_14.wav`](https://raw.githubusercontent.com/inworld-ai/voice-isolator/main/demo/examples/sample_14.wav): play the noisy input, cleaned voice, and removed background.
- Streaming isolation of the same clip in 20 ms chunks with wall-clock timing.
- Optional upload / local path for your own file (`UPLOAD_OWN_AUDIO` stays false so Run all does not block).

## Links

- Model card: [inworld/voice-isolator](https://huggingface.co/inworld/voice-isolator)
- Code: [github.com/inworld-ai/voice-isolator](https://github.com/inworld-ai/voice-isolator)
- Official cleanup notebook: [examples/audio_cleanup.ipynb](https://github.com/inworld-ai/voice-isolator/blob/main/examples/audio_cleanup.ipynb)
- Blog: [Introducing Voice Isolator](https://inworld.ai/blog/introducing-voice-isolator)
- Source post: [x.com/inworld](https://x.com/inworld/status/2123120546908926300)
- Notebook: [`colab/Inworld-Voice-Isolator-48k.ipynb`](../colab/Inworld-Voice-Isolator-48k.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Inworld-Voice-Isolator-48k.ipynb)
- Related: [Rumble-2-Flash noisy speech-to-text](Rumble-2-Flash-noisy-speech-to-text.md), [Whisper transcription](AIBackends-Whisper-audio-transcription.md), [Nemotron-3 Diarization](Nemotron-3-Diarization.md), [tropa-voice-mini AI voice detector](tropa-voice-mini-ai-voice-detector.md)
