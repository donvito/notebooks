# oruk-loop-153M looped English ASR

[`oruk/oruk-loop-153M`](https://huggingface.co/oruk/oruk-loop-153M) is Oruk AI’s looped-transformer English ASR (153M stored params, distilled from Qwen3-ASR-1.7B-hf; encoder block ×24, decoder block ×28). Custom code is MIT (`trust_remote_code`). **Weights are not standard open source:** PolyForm Small Business 1.0.0 or PolyForm Noncommercial 1.0.0; other commercial use needs a license from Oruk AI. Required notice: Copyright (c) 2026 Oruk AI (https://oruk.ai). The card reports mean WER 7.12% (beam 4) on the 7 English Open ASR Leaderboard sets.

## Use cases

- **Compact English ASR** when 153M stored weights matter more than a 1.7B teacher.
- **A T4/CPU alternative** to Whisper or Granite Speech for English-only files.

## Not a good fit for

- **Unrestricted commercial weights.** Read LICENSE-WEIGHTS.md.
- **Multilingual ASR / translation.** English looped ASR only.
- **Assuming float16 quality matches the card.** If transcripts look degraded, the notebook says to reload in float32.

## Running it on a free Colab T4

Pick a T4 GPU (CPU works in float32, slower than real time). Float16; ~0.58 GB stored weights, a few GB merged. Card CUDA default is bf16 autocast plus ~4.2 GB merged; this notebook uses float16 + `merge=True`.

## What the notebook demos

- Load the looped ASR with `trust_remote_code`.
- English transcription.
- License/notice caveats.

## Links

- Model card: [oruk/oruk-loop-153M](https://huggingface.co/oruk/oruk-loop-153M)
- Weights license: [LICENSE-WEIGHTS.md](https://huggingface.co/oruk/oruk-loop-153M/blob/main/LICENSE-WEIGHTS.md)
- Notebook: [`colab/oruk-loop-153M-looped-asr.ipynb`](../colab/oruk-loop-153M-looped-asr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/oruk-loop-153M-looped-asr.ipynb)
- Related: [Whisper](AIBackends-Whisper-audio-transcription.md), [Granite Speech](granite-speech-5.0-2b-asr-translation.md)
