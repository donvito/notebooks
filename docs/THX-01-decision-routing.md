# THX-01 decision routing

[`doofz/THX-01`](https://huggingface.co/doofz/THX-01) is a 322M phone/edge-scale decision model (Apache 2.0, mmBERT + decision head). Install with `pip install thx01`. Demos cover intent/queue classification, priority scores, multilingual tickets, and calibrated auto-route thresholds. Distinct from the d1-3B vision notebook and the Unsloth email-triage trainers.

## Use cases

- **On-phone or CPU routing** when 322M is the budget.
- **Multilingual tickets** (card lists many languages including Azerbaijani).
- **Calibrated auto-route thresholds** without generating text.

## Not a good fit for

- **Image or audio decisions.** Text encoder. Use d1 / Kanana for vision.
- **Training a 4B Unsloth decision head.** Different recipe.

## Running it on a free Colab T4

T4 or CPU. Exact VRAM is not documented; 322M is intended to stay light.

## What the notebook demos

- Intent/queue classification.
- Priority scores.
- Multilingual tickets.
- Calibrated auto-route thresholds.

## Links

- Model card: [doofz/THX-01](https://huggingface.co/doofz/THX-01)
- Blog: [THX-01](https://huggingface.co/blog/doofz/thx-01-we-open-sourced-a-322m-decision-model-that)
- Notebook: [`colab/THX-01-decision-routing.ipynb`](../colab/THX-01-decision-routing.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/THX-01-decision-routing.ipynb)
- Related: [d1-3B](d1-3B-inference-decisions.md), [GLiNER2.5-Decide](GLiNER2_5-Decide-classification.md)
