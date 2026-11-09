# d1-3B decisions

[`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) is Liquid AI's 3.1B decision model post-trained from LFM2.5-VL-3B (LFM Open License v1.0). It answers yes/no, multiple-choice, and rating questions about text, JSON, or images in one forward pass with zero output tokens. It is not a chat model. For new work, Liquid's card points at [d1-2B](d1-2B-inference-decisions.md); this notebook is left as the original 3B recipe.

## Use cases

- **Support triage.** Several named questions over one email in one pass.
- **Agent guardrails.** Check a JSON tool call before execution, with `noul` criteria spelling out yes and no.
- **Batch routing.** Score many messages at once without generating text.
- **Image questions.** Ask calibrated questions about a picture (visual inspection, not captioning).
- **LLM-as-a-judge style scores.** `score` questions return an expected level plus probabilities.

## Not a good fit for

- **Generating text.** There is no decoder output.
- **Audio.** Use [d1-Omni-2B](d1-Omni-2B-inference-decisions.md) or [d1-omni-600M](d1-omni-600M-inference-decisions.md).
- **Invoice extraction fields.** That demo lives on [d1-2B](d1-2B-inference-decisions.md).

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 (about 6.2 GB of VRAM, about 7 GB peak with an image). GPUs with native bfloat16 use bfloat16, the card default. `trust_remote_code=True`. Restart the runtime after the transformers upgrade.

## What the notebook demos

- Load d1-3B.
- Several questions about one state.
- JSON state: an agent guardrail.
- Batch: route many messages at once.
- Image state.
- Next steps.

## Links

- Model card: [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B)
- Blog: [d1 open](https://www.liquid.ai/blog/d1-open)
- Notebook: [`colab/d1-3B-inference-decisions.ipynb`](../colab/d1-3B-inference-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/d1-3B-inference-decisions.ipynb)
- Related: [d1-2B](d1-2B-inference-decisions.md), [d1-Omni-2B](d1-Omni-2B-inference-decisions.md), [d1-omni-600M](d1-omni-600M-inference-decisions.md)
