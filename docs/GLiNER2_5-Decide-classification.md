# GLiNER2.5-Decide classification

[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) is Fastino’s English operational classifier (Apache 2.0, GLiNER2.5 family). The notebook describes it as a 340M DeBERTa-v3-large encoder. You pass any label set at call time and it scores one or more decision heads in a single forward pass — no prompt template, no generated tokens, no fine-tuning. It does not reason, explain, or answer open questions.

## Use cases

- **Support intent and ticket routing**, including several heads in one call (email triage).
- **Sentiment and product aspects** without training a new classifier.
- **Agent gates.** Handoff-to-human and “did the agent finish?” checks.
- **Moderation and spam**, plus incident severity or urgency scores.
- **Ordinal scores and thresholds** when you need a 0–10 style head plus confidence.

## Not a good fit for

- **Open-ended chat or explanations.** It only scores the labels you supply.
- **Image or audio decisions.** Text only. Use d1-Omni / Kanana for multimodal `system_one`.
- **Calibrated LLM decision heads you train yourself.** That is the Unsloth email-triage notebooks.

## Running it on a free Colab T4

It runs on CPU or GPU. The notebook is written for both. A T4 is optional. Exact VRAM is not documented; this is a sub-billion encoder.

## What the notebook demos

- Intent classification (support, banking, travel, clinic).
- Sentiment and product aspects.
- Topic, document type, and genre.
- Email triage and ticket routing.
- Agent gates, moderation, severity and urgency.
- Multi-head calls, questions over a passage, label descriptions.
- Ordinal scores, confidence thresholds, schema builder, batching.
- A message router that puts the pieces together.

## Links

- Model card: [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
- Notebook: [`colab/GLiNER2_5-Decide-classification.ipynb`](../colab/GLiNER2_5-Decide-classification.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/GLiNER2_5-Decide-classification.ipynb)
- Related: [AIBackends GLiNER2.5 advanced extraction](AIBackends-GLiNER2_5-advanced-extraction.md), [d1-2B](d1-2B-inference-decisions.md)
