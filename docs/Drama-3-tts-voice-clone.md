# Fish Audio Drama 3 TTS

[`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3) is Fish Audio's mixture-of-experts text-to-speech model (35B total, 3B active) for multi-speaker scenes, voice cloning, and inline delivery tags such as `[sighs]`. This notebook serves the official int4 checkpoint [`fishaudio/drama-3-int4`](https://huggingface.co/fishaudio/drama-3-int4) with SGLang-Omni. License is Fish Audio Community License v1.0, not Apache or MIT.

## Use cases

- **Multi-speaker drama or audiobook lines** with inline delivery tags.
- **High-quality cloning of a voice you have rights to.** The notebook clones the model's published 3.9 s demo clip, not a real person.
- **Pro-tier Colab exploration of Drama 3** when a T4/L4 path is not claimed to work.

## Not a good fit for

- **Free Colab T4 or L4.** GPTQ-Marlin needs compute capability 8.0+ (A100). A T4 is 7.5 and cannot run this path.
- **Hosted TTS/cloning as a service, or cloning people without consent.** The community license forbids those uses. Commercial use is allowed only while annual revenue stays under USD 500,000.
- **Training another speech model on the output.** The license forbids it.
- **Community GGUF / audio.cpp on Colab L4.** The notebook does not claim that configuration.

## Running it on a free Colab T4

It does not run on a free T4. Use Colab Pro A100, 40 GB or larger. Int4 engine weights are 20.8 GB. On a 40 GB A100 the launch budget is about 33 GB (engine ~25 GB, vocoder ~5 GB, 3 GiB preprocessing reserve). The ~37 GB bf16 checkpoint does not fit a 40 GB A100. SGLang-Omni is pinned to commit `c45d6182e8d19c38408b7f580721d7c078eacf33` in its own virtualenv (CUDA 13 stack). First run downloads that stack; the server later downloads the ~22 GB int4 checkpoint. `/health` can take up to half an hour for download plus CUDA-graph capture.

## What the notebook demos

- Refuse the wrong GPU.
- Install SGLang-Omni in a venv.
- Start the server and wait on `/health`.
- Text to speech (`/v1/audio/speech`).
- Voice cloning from the published demo reference.
- Stop the server.

## Links

- Model cards: [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) · [fishaudio/drama-3-int4](https://huggingface.co/fishaudio/drama-3-int4)
- License: [Fish Audio Community License v1.0](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE)
- Cookbook: [SGLang-Omni Drama 3](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)
- Notebook: [`colab/Drama-3-tts-voice-clone.ipynb`](../colab/Drama-3-tts-voice-clone.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Drama-3-tts-voice-clone.ipynb)
- Related: [Audio8-TTS-Realistic-Air](Audio8-TTS-Realistic-Air-voice-clone.md), [`phonon-31m-en-streaming-tts.ipynb`](../colab/phonon-31m-en-streaming-tts.ipynb)
