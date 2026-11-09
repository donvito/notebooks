# d1-Omni-2B decisions

[`LiquidAI/d1-Omni-2B`](https://huggingface.co/LiquidAI/d1-Omni-2B) is Liquid AI's ~2.17B decision head on [`LiquidAI/LFM3-Omni-2B`](https://huggingface.co/LiquidAI/LFM3-Omni-2B) (LFM Open License v1.0). Hub spelling is capital O. It uses the same `system_one` API as d1-2B, plus audio clips. It is not a chat model and writes no transcript or speech. Prefer d1-2B when you have no audio. This notebook does not change the older [d1-omni-600M](d1-omni-600M-inference-decisions.md) notebook.

## Use cases

- **The same text triage as d1-2B / d1-3B**, when you want one checkpoint that can also take audio later.
- **Image boxes** on a photo passed through `media=`.
- **Voice-note routing.** Score a short spoken request (16 kHz mono, a few seconds) without transcribing it first.
- **Non-speech audio checks.** The notebook includes the card's synthetic 880 Hz beeps as an API demo.

## Not a good fit for

- **ASR or TTS.** It decides; it does not transcribe or speak.
- **Long audio.** Keep clips to a few seconds; audio is 16 kHz mono.
- **Text-only or image-only production.** The card says to prefer d1-2B without audio.

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 (about 4.3 GB of weights) with SDPA. Card default is bfloat16. Restart the runtime after installing transformers.

## What the notebook demos

- Check the T4 and pick a dtype.
- Load d1-Omni-2B.
- Text triage (same email as d1-2B / d1-3B).
- Image boxes (COCO cats).
- Audio: beeps plus a voice-note route (espeak-ng, not a recorded voice).
- Caveats.

## Links

- Model card: [LiquidAI/d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B)
- Source post: [x.com/liquidai](https://x.com/liquidai/status/2115128329003303107)
- Blog: [LFM3](https://www.liquid.ai/blog/lfm3)
- Notebook: [`colab/d1-Omni-2B-inference-decisions.ipynb`](../colab/d1-Omni-2B-inference-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/d1-Omni-2B-inference-decisions.ipynb)
- Related: [d1-2B](d1-2B-inference-decisions.md), [d1-omni-600M](d1-omni-600M-inference-decisions.md), [d1-3B](d1-3B-inference-decisions.md)
