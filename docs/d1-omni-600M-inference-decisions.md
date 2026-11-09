# d1-omni-600M decisions

[`LiquidAI/d1-omni-600M`](https://huggingface.co/LiquidAI/d1-omni-600M) is Liquid AI's experimental 587M encoder-based decision model (LFM Open License v1.0). It is post-trained from LFM2.5-Encoder-350M with a SigLIP2 vision tower and a 17-layer FastConformer audio encoder. It answers yes/no, choice, and score questions about text plus an image, or text plus a speech clip, in one forward pass. For new work, prefer [d1-Omni-2B](d1-Omni-2B-inference-decisions.md). This notebook is left as the 600M speech-routing sibling of d1-3B.

Liquid published this as an early research checkpoint. On Decision Index 0.2.1 the card reports 15.95, well below d1-3B (48.57). Treat answers as an API demo, not a quality stand-in for d1-3B.

## Use cases

- **Tiny multimodal decisions on the edge.** Sub-billion encoder with image or audio, not both in one request.
- **Document photo plus text triage** when you cannot run d1-3B.
- **Voice-note routing** for English speaker↔assistant style clips (utterance type, topic, intent), which d1-3B does not cover the same way.

## Not a good fit for

- **Production quality vs d1-3B / d1-Omni-2B.** The card's Decision Index gap is large.
- **Images and audio in the same call.** Passing both raises `ValueError`.
- **Clips longer than 30 s, or stereo.** Audio is one 16 kHz mono clip, cut at 30 s.
- **Apache/MIT-only commercial use.** LFM Open License v1.0 limits commercial use to organizations under a $10M annual-revenue threshold; read the license.

## Running it on a free Colab T4

Pick a T4 GPU. The card recommends float16 on GPU (trained in float32). Do not use bfloat16: the card says it flipped the top answer on 0.8% of text and 1.7% of audio rows. Weights are about 1.2 GB in float16 (Hub file is 2.3 GB of float32). Peak on the short demos should stay a few GB. No FlashAttention-2. `trust_remote_code=True`. Restart after upgrading transformers.

## What the notebook demos

- Load d1-omni-600M.
- Text plus image: document triage and a photo defect.
- Text plus audio: voice-note routing.
- Caveats (experimental quality, modality pairing, license).

## Links

- Model card: [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
- Blog: [Open d1](https://huggingface.co/blog/LiquidAI/open-d1)
- License: [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-omni-600M/blob/main/LICENSE)
- Notebook: [`colab/d1-omni-600M-inference-decisions.ipynb`](../colab/d1-omni-600M-inference-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/d1-omni-600M-inference-decisions.ipynb)
- Related: [d1-Omni-2B](d1-Omni-2B-inference-decisions.md), [d1-3B](d1-3B-inference-decisions.md), [d1-2B](d1-2B-inference-decisions.md)
