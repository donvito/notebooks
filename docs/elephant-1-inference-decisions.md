# elephant-1 decisions

[`baselabs/elephant-1`](https://huggingface.co/baselabs/elephant-1) is Base Labs / Baseten's 9B decision model (MIT), post-trained from Qwen3.5-9B. It answers yes/no, choice, and score questions through the System One API (`qev.DecisionEngine`) in one shared-prefix prefill with zero output tokens. It is not a chat model. The 27B sibling mammoth-1 does not fit a T4.

## Use cases

- **Support triage.** Same tickets as the d1-3B notebook, so you can compare a 9B 4-bit decision model with 3.1B float16 d1.
- **Agent tool-call guardrails.** Score a JSON tool call before it runs.
- **Batch routing.** Many messages, calibrated probabilities, no generated text.
- **LLM-as-a-judge scores.** Named `score` questions with a probability for every level.

## Not a good fit for

- **Writing replies.** There is no decoder output.
- **Production vision.** The technical report says the vision encoder was frozen and training was on text; image/video training is planned. The notebook tries COCO questions and skips if the runtime cannot take pixels.
- **Community GGUF.** llama.cpp does not implement `slot_lm` or the block-bidirectional mask, so this notebook does not use GGUF.

## Running it on a free Colab T4

Pick a T4 GPU. The Hub checkpoint is about 18.8 GB of bfloat16 and does not fit. The notebook loads 4-bit NF4 with float16 compute. About 5–8 GB of 4-bit weights, roughly 8–12 GB peak. No FlashAttention-2. Restart after upgrading transformers.

## What the notebook demos

- Load elephant-1 in 4-bit via `qev`.
- Support triage (same tickets as d1-3B).
- JSON agent guardrail.
- Batch routing.
- Optional image questions (skipped if unsupported).

## Links

- Model card: [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1)
- Technical report: [Mammoth-1 and Elephant-1](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)
- Notebook: [`colab/elephant-1-inference-decisions.ipynb`](../colab/elephant-1-inference-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/elephant-1-inference-decisions.ipynb)
- Related: [d1-3B](d1-3B-inference-decisions.md), [IOTA Decide 4B](IOTA-Decide-4B-typed-decisions.md)
