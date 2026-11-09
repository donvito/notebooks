# Swar Nepali/English voice-clone TTS

[`Aarjan/swar`](https://huggingface.co/Aarjan/swar) is Naamche Labs’ F5-TTS Base for Nepali, English, and mixed-language zero-shot cloning. Weights are [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (share-alike; commercial OK). Code is MIT. Default `podvoices_ema.pt` is about 1.35 GB. Maya/Aarav presets are model-made audio, not recordings of real people. Only clone voices you have permission to use.

## Use cases

- **Nepali and mixed Nepali/English cloning** from a short reference.
- **English F5-TTS** when you want the same checkpoint as the Nepali path.
- **Presets (Maya/Aarav)** that are synthetic, not celebrity clones.

## Not a good fit for

- **Cloning a real person without consent.**
- **Tiny English-only CPU TTS.** Use Phonon or Silencio-2.
- **Indic 11-language streaming.** Use Maya 2 Native.

## Running it on a free Colab T4

Pick a T4 GPU. Official PyPI `f5-tts==1.1.22`, float32 EMA (fp16 NaNs). Optional ~0.49 GB Kriti ONNX CTC is not loaded here.

## What the notebook demos

- Load Swar F5-TTS float32 EMA.
- Nepali, English, and mixed-language cloning.
- Synthetic Maya/Aarav presets.
- Consent/license notes.

## Links

- Model card: [Aarjan/swar](https://huggingface.co/Aarjan/swar)
- Code: [Naamche-Labs/swar](https://github.com/Naamche-Labs/swar)
- Source post: [x.com/ArjanChaudharyy](https://x.com/ArjanChaudharyy/status/2115809203218755809)
- Notebook: [`colab/Swar-voice-clone-tts.ipynb`](../colab/Swar-voice-clone-tts.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Swar-voice-clone-tts.ipynb)
- Related: [Maya 2 Native](Maya2-Native-indic-tts.md), [Silencio-2](Silencio-2-tts-voice-clone.md), [Phonon 31M](phonon-31m-en-streaming-tts.md)
