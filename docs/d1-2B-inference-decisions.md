# d1-2B decisions

[`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B) is Liquid AI's ~1.92B decision model post-trained from [`LiquidAI/LFM3-VL-2B`](https://huggingface.co/LiquidAI/LFM3-VL-2B) (LFM Open License v1.0). It answers yes/no, choice, score, extraction, and box questions about text, JSON, images, or short video in one forward pass with zero output tokens. It is not a chat model. Prefer this over d1-3B for new work; the d1-3B notebook stays unchanged.

## Use cases

- **Support email triage.** Several named questions over one ticket in one pass: queue, priority, and related yes/no checks with calibrated probabilities.
- **Agent tool-call guardrails.** Score a JSON tool call before it executes.
- **Batch routing.** Route many messages without generating text.
- **Image boxes.** Return normalized boxes on a picture (the notebook uses a COCO-style demo).
- **Structured invoice fields.** Extract values such as VAT; missing fields come back as `None` rather than hallucinated text.

## Not a good fit for

- **Writing replies, transcripts, or chain-of-thought.** `system_one` reads answers off logits; it does not generate tokens.
- **Audio decisions.** Use [d1-Omni-2B](d1-Omni-2B-inference-decisions.md) when the state includes speech.
- **A drop-in chat LLM.** Pair it with a generator if you need language output.

## Running it on a free Colab T4

Pick a T4 GPU. The card uses bfloat16; a T4 has no native BF16, so this notebook loads float16 with SDPA. About 3.8 GB of weights. Peak on the short demos should stay well under 15 GB. Load with `trust_remote_code=True`. Restart the runtime after upgrading transformers.

## What the notebook demos

- Check the T4 and pick a dtype.
- Load d1-2B.
- Support triage (same tickets as d1-3B).
- JSON state: an agent guardrail.
- Batch routing plus image boxes.
- Structured invoice extraction.
- Caveats.

## Links

- Model card: [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B)
- Source post: [x.com/liquidai](https://x.com/liquidai/status/2115128329003303107)
- Blog: [LFM3](https://www.liquid.ai/blog/lfm3)
- Notebook: [`colab/d1-2B-inference-decisions.ipynb`](../colab/d1-2B-inference-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/d1-2B-inference-decisions.ipynb)
- Related: [d1-3B](d1-3B-inference-decisions.md), [d1-Omni-2B](d1-Omni-2B-inference-decisions.md), [d1-omni-600M](d1-omni-600M-inference-decisions.md)
