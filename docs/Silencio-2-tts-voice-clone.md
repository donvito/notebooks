# Silencio-2 TTS voice clone

[`BlueRoseLabs/Silencio-2`](https://huggingface.co/BlueRoseLabs/Silencio-2) is a small English TTS model (CC-BY-4.0) with sound/emotion tags, streaming, and zero-shot cloning. It is a fine-tune of `kyutai/pocket-tts`. About 18.2M parameters generate speech; 27.9M including the Mimi encoder cloning needs. All weights sit in one ~112 MB `model.safetensors`. Inference uses the Hub-shipped `silencio2` package (not PyPI, not `pocket-tts`).

## Use cases

- **English TTS with inline sound and emotion tags** on a 112 MB file.
- **Streaming chunks** and a real-time factor check.
- **Zero-shot clone of a clip you own.** The notebook clones audio it just synthesized.

## Not a good fit for

- **Impersonation.** Clone only with permission; label synthetic speech.
- **Indic / Nepali / multilingual catalogs.** Use Maya 2, Swar, or Audio8.
- **Installing `silencio2` from PyPI.** Snapshot the Hub repo and import locally.

## Running it on a free Colab T4

T4 is fastest; CPU also works. The package runs float32. First run downloads ~112 MB plus bundled voice WAVs.

## What the notebook demos

- Load from a Hub snapshot.
- Bundled voices.
- Sound and emotion tags.
- Clone from a notebook-made clip.
- Streaming chunks and real-time factor.

## Links

- Model card: [BlueRoseLabs/Silencio-2](https://huggingface.co/BlueRoseLabs/Silencio-2)
- Source post: [x.com/julian_inada](https://x.com/julian_inada/status/2116987775467835840)
- Notebook: [`colab/Silencio-2-tts-voice-clone.ipynb`](../colab/Silencio-2-tts-voice-clone.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Silencio-2-tts-voice-clone.ipynb)
- Related: [Phonon 31M](phonon-31m-en-streaming-tts.md), [Swar](Swar-voice-clone-tts.md)
