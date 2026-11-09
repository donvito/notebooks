# Audio8-TTS-Realistic-Air

[`Edge0/Audio8-TTS-Realistic-Air`](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) is Edge0's 0.6B multilingual text-to-speech model (Apache 2.0). It is a DualAR transformer (Fish Audio S2 Pro–style) with a bundled 44.1 kHz neural codec. The checkpoint is 601M parameters and covers 23 languages. Inference uses Transformers `AutoProcessor` + `AutoModel` with `trust_remote_code=True`.

## Use cases

- **Multilingual product voice.** Generate English, Chinese, Portuguese, and other listed languages from one 0.6B checkpoint instead of a language-specific TTS stack.
- **Zero-shot cloning of a voice you own.** Condition on a short reference clip. The notebook clones a clip it just synthesized, not a real person.
- **Local preview of a 23-language catalog.** Quality varies by language; use the card's evaluation tables rather than assuming uniform quality.

## Not a good fit for

- **Cloning a real person without consent.** The notebook and card forbid impersonation. The demo reuses notebook-made audio.
- **Tiny English-only on-device TTS.** See [`phonon-31m-en-streaming-tts.ipynb`](../colab/phonon-31m-en-streaming-tts.ipynb) or [`Silencio-2-tts-voice-clone.ipynb`](../colab/Silencio-2-tts-voice-clone.ipynb).
- **SGLang Omni / OpenAI speech API serving.** That path is not installed here.
- **Colab-free huge multi-speaker drama.** Drama 3 needs a Pro A100.

## Running it on a free Colab T4

Pick a T4 GPU. The card's CUDA default is bfloat16; a T4 has no native BF16, so this notebook loads float16 with SDPA (bfloat16 on L4/A100). First run downloads about 1.2 GB `model.safetensors` plus 1.3 GB `codec.pth`. Peak VRAM for a short clip should stay well under 8 GB. Pin `transformers>=4.57.0,<5`. Do not replace Colab's torch, do not install `flash-attn`, and restart the session after the install cell.

## What the notebook demos

- Early GPU check.
- Card Transformers stack, keeping Colab's torch.
- Runtime check and T4 dtype.
- Load with `trust_remote_code`.
- Multilingual TTS without a reference (English and Chinese).
- Zero-shot clone from a clip the notebook just made.
- Portuguese save and playback.
- Caveats and cleanup.

## Links

- Model card: [Edge0/Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air)
- Source post: [x.com/SamuelZengML](https://x.com/SamuelZengML/status/2115881504102551856)
- Code: [Edge0-AI/Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS)
- Notebook: [`colab/Audio8-TTS-Realistic-Air-voice-clone.ipynb`](../colab/Audio8-TTS-Realistic-Air-voice-clone.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Audio8-TTS-Realistic-Air-voice-clone.ipynb)
- Related: [Drama 3](Drama-3-tts-voice-clone.md), [`phonon-31m-en-streaming-tts.ipynb`](../colab/phonon-31m-en-streaming-tts.ipynb), [`Silencio-2-tts-voice-clone.ipynb`](../colab/Silencio-2-tts-voice-clone.ipynb), [`Swar-voice-clone-tts.ipynb`](../colab/Swar-voice-clone-tts.ipynb)
