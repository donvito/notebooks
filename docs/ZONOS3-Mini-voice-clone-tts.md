# ZONOS3-Mini TTS voice clone

[`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) is Zyphra's compact multilingual text-to-speech model packaged for Transformers (Apache-2.0, not gated). The Hub file holds about 1.04B parameters in a 2.4 GB `model.safetensors`. The card describes ~100 million active parameters per token and an 800 million backbone, 44.1 kHz mono output, 43 languages, 27 release voices, and cloning through the bundled MECT-B2 speaker encoder. Native weights: [`Zyphra/ZONOS3-Mini`](https://huggingface.co/Zyphra/ZONOS3-Mini).

The [announcement](https://x.com/ZyphraAI/status/2119914895861076151) presents it as open-source TTS with voice cloning and word-level control that can run real-time on a phone. Word-level steering, accents, speech editing, LACI, and streaming decode are **not** in this Transformers checkpoint — the card lists those on the [native runtimes](https://github.com/Zyphra/ZONOS3-Mini).

## Use cases

- **43-language product voice from one checkpoint.** Named release voices (Savannah, Ximena, Ananya, and 24 others) plus a SuperBPE tokenizer. `language` selects text normalization, not a language token in the prompt.
- **Zero-shot clone of a voice you own.** Condition on a short clip through the speaker encoder (`speaker_audio` / `embed_speaker`). The notebook clones a clip it just synthesised, not a real person.
- **Local Transformers preview without a native server.** `AutoModel.from_pretrained(..., trust_remote_code=True)` then `model.generate(...)`. No separate codec download.

## Not a good fit for

- **Cloning a real person without consent.** The notebook and card forbid impersonation. The demo reuses notebook-made audio.
- **Word-level steering, accents, LACI, streaming decode, or speech editing.** Those are native-runtime features. This Hub repo does not include them.
- **Tiny English-only on-device TTS.** See [`phonon-31m-en-streaming-tts.ipynb`](../colab/phonon-31m-en-streaming-tts.ipynb) or [`Silencio-2-tts-voice-clone.ipynb`](../colab/Silencio-2-tts-voice-clone.ipynb).
- **The Transformers `text-to-speech` pipeline.** The card says it tokenizes before the model sees the text and skips normalization. Call `generate` with the raw text.
- **Colab-free huge multi-speaker drama.** Drama 3 needs a Pro A100.

## Running it on a free Colab T4

Pick a T4 GPU. The checkpoint's LM is bfloat16; a T4 has no native BF16, so this notebook loads **float16** (bfloat16 on L4/A100). The card keeps the codec decoder and speaker encoder in float32. First run downloads about 2.4 GB `model.safetensors` plus the tokenizer and 27 voices. Pin `transformers>=4.57`. Do not replace Colab's torch, do not install `flash-attn`. Optional `zyphra-textnorm` reads numbers, dates, currencies, and units aloud for English, Mandarin, Japanese, Korean, German, Spanish, French, Italian, and Portuguese. Short clips should stay well under the T4's ~15 GB; the notebook prints measured peak VRAM.

## What the notebook demos

- Early GPU check.
- Card Transformers stack, keeping Colab's torch, plus optional `zyphra-textnorm`.
- Runtime check and T4 dtype.
- Load with `trust_remote_code`.
- Built-in Savannah voice in English (card snippet).
- The same voice in Spanish (`language="es"`).
- Clone from the English WAV the notebook just made.
- Caveats and cleanup, with peak VRAM.

## Links

- Transformers card: [Zyphra/ZONOS3-Mini-hf](https://huggingface.co/Zyphra/ZONOS3-Mini-hf)
- Base card: [Zyphra/ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini)
- Source post: [x.com/ZyphraAI](https://x.com/ZyphraAI/status/2119914895861076151)
- Blog: [zyphra.com/our-work/zonos3-mini](https://www.zyphra.com/our-work/zonos3-mini)
- Technical report: [arXiv:2611.16525](https://arxiv.org/abs/2611.16525)
- Code: [Zyphra/ZONOS3-Mini](https://github.com/Zyphra/ZONOS3-Mini)
- Notebook: [`colab/ZONOS3-Mini-voice-clone-tts.ipynb`](../colab/ZONOS3-Mini-voice-clone-tts.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/ZONOS3-Mini-voice-clone-tts.ipynb)
- Related: [Audio8](Audio8-TTS-Realistic-Air-voice-clone.md), [Silencio-2](Silencio-2-tts-voice-clone.md), [Ming-3.0-tts 0.5B](Ming-3.0-tts-0.5B.md), [Phonon 31M](phonon-31m-en-streaming-tts.md)
