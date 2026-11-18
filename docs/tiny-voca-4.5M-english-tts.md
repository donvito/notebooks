# tiny-voca 4.5M English TTS

[`n4ze3m/tiny-voca`](https://huggingface.co/n4ze3m/tiny-voca) is a small English text-to-speech model with two voices, F2 (female) and M2 (male). The complete model, including the waveform generator, has 4,537,291 parameters (`tiny-voca.pt` 18.2 MB, PyTorch float32). Output is 24 kHz mono, American English. It was trained from scratch on about 18 hours of synthetic speech. Apache-2.0. The [card](https://huggingface.co/n4ze3m/tiny-voca) says it runs about 70× faster than real time on four CPU threads.

The [announcement](https://x.com/n4ze3m/status/2121799813679419515) is the author's post from 15 November 2026. Code: [`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca). Phonemizer: [`n4ze3m/dizione`](https://github.com/n4ze3m/dizione). Browser Space: [`n4ze3m/tiny-voca-studio`](https://huggingface.co/spaces/n4ze3m/tiny-voca-studio).

## Use cases

- **Tiny English TTS on CPU or a free T4.** Two built-in voices, no extra vocoder, 18.2 MB of PyTorch weights (plus 21.4 MB ONNX if you want that path).
- **A phone / browser-sized checkpoint.** The card ships ONNX for `onnxruntime` (and notes the same file runs in onnxruntime-web) and GGUF variants on the Hub. This notebook uses PyTorch and ONNX Runtime on Colab.
- **A permissive Apache-2.0 path** with a dictionary phonemizer (dizione) instead of eSpeak.

## Not a good fit for

- **Voice cloning, multilingual TTS, or streaming.** Two fixed American English voices only. The card: no streaming. Use [Silencio-2](Silencio-2-tts-voice-clone.md) or [ZONOS3-Mini](ZONOS3-Mini-voice-clone-tts.md) for cloning; [Phonon 31M](phonon-31m-en-streaming-tts.md) for streaming English.
- **Studio-natural or variable delivery.** The card: the voice can sound slightly synthetic; F2 sounds younger than intended; the same text is always read the same way; speech is slightly faster than the training recordings at the default speed. Training audio is itself synthetic.
- **Non-English text, or names the phonemizer does not know.** The card: English only; foreign names and unusual spellings are often mispronounced; numbers, abbreviations, and homographs depend on dizione.

## Running it on a free Colab T4

CPU is enough; a T4 is optional. The card install is `pip install git+https://github.com/n4ze3m/dizione git+https://github.com/n4ze3m/tiny-voca`, then `hf download n4ze3m/tiny-voca --local-dir tiny-voca` and `export DIZIONE_DATA=tiny-voca/phonemizer`. The notebook keeps Colab's torch, uses `%pip` (not `python -I`), and downloads `tiny-voca.pt` (18.2 MB), `onnx/model.onnx` (21.4 MB), `config.json`, and `phonemizer/` (about 5.4 MB). No Hugging Face token. The PyTorch path uses CUDA when it is available; the ONNX demo stays on CPU. Peak VRAM is printed if CUDA is used.

## What the notebook demos

- Early runtime check (`torch.cuda.is_available()`, device name, free VRAM).
- Card install of dizione + tiny-voca, keeping Colab's torch, plus CPU onnxruntime.
- Hub download and `DIZIONE_DATA`.
- Both voices (F2 and M2) on the same line.
- `--speed`, `--pitch-shift` (0.058 ≈ one semitone), and `--pitch-range`.
- A long multi-sentence paragraph with wall time and real-time factor on CPU, and on GPU when CUDA is available.
- The card's ONNX Runtime path on CPU (phoneme ids + stress → 24 kHz waveform, peak scaled to 0.99).
- Caveats and cleanup, with peak VRAM if CUDA was used.

## Links

- Model card: [n4ze3m/tiny-voca](https://huggingface.co/n4ze3m/tiny-voca)
- Source post: [x.com/n4ze3m](https://x.com/n4ze3m/status/2121799813679419515)
- Code: [github.com/n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca)
- Phonemizer: [github.com/n4ze3m/dizione](https://github.com/n4ze3m/dizione)
- Space: [n4ze3m/tiny-voca-studio](https://huggingface.co/spaces/n4ze3m/tiny-voca-studio)
- Notebook: [`colab/tiny-voca-4.5M-english-tts.ipynb`](../colab/tiny-voca-4.5M-english-tts.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/tiny-voca-4.5M-english-tts.ipynb)
- Related: [Phonon 31M](phonon-31m-en-streaming-tts.md), [Silencio-2](Silencio-2-tts-voice-clone.md)
