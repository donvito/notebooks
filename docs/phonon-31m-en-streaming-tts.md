# Phonon 31M English streaming TTS

[`gradium/phonon-31m-en`](https://huggingface.co/gradium/phonon-31m-en) is Gradium’s Apache 2.0 English on-device TTS, run here through `ptts`. The q8 GGUF is 64.5 MB and the runtime is CPU, so VRAM is unused. This is the free-Colab contrast to A100-only Drama 3. Do not use `phonon-40m-*` (CC-BY-NC).

## Use cases

- **Tiny English streaming TTS** on CPU or T4 without touching VRAM.
- **A 64.5 MB q8 voice** for previews and IVR-style lines.
- **A permissive Apache path** vs the 40m CC-BY-NC family.

## Not a good fit for

- **Multilingual or Indic TTS.** Use Maya 2, Audio8, or Ming.
- **High-end cloning / drama.** Use Drama 3 on a Pro A100.
- **`phonon-40m-*`.** Different license; not used here.

## Running it on a free Colab T4

T4 or CPU. `ptts` 0.4.2’s Linux wheel reports `available_devices() == ["cpu"]`. First run downloads the 64.5 MB q8 file plus the Easton voice from `gradium/phonon-voices`. `ptts` is MIT OR Apache-2.0.

## What the notebook demos

- Install `ptts` and load q8 Phonon 31M.
- Streaming English TTS.
- Notes vs Drama 3 and vs phonon-40m.

## Links

- Model card: [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en)
- ptts: [PyPI](https://pypi.org/project/ptts/)
- Notebook: [`colab/phonon-31m-en-streaming-tts.ipynb`](../colab/phonon-31m-en-streaming-tts.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/phonon-31m-en-streaming-tts.ipynb)
- Related: [Drama 3](Drama-3-tts-voice-clone.md), [Maya 2 Native](Maya2-Native-indic-tts.md)
