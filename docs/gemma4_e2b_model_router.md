# Gemma 4 E2B model router

This notebook QLoRA-tunes gated [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) (Apache 2.0) with Unsloth so it routes a software task to `luna` (simple / bounded) or `astra` (complex / security-sensitive / architectural). It starts from the ModernBERT routing tasks and adds harder ones. ModernBERT learns a classification head; this notebook teaches Gemma to answer with the route name.

## Use cases

- **Train a small generative router** when you want the model to emit a route name instead of a classifier logit.
- **Compare with ModernBERT** on the same luna/astra policy, with a richer task set.
- **Export a LoRA adapter** for a local routing service.

## Not a good fit for

- **A tiny CPU classifier.** Use [ModernBERT](ModernBERT_train_classify.md) if you only need a binary head.
- **Calibrated multi-head email triage.** Use the Unsloth decision-model notebooks (Gemma 4 E4B / Qwen3.5-4B).
- **Running without accepting the Gemma license.** Weights are gated; a 401 means you have not accepted the card or signed in.

## Running it on a free Colab T4

A free T4 is enough for 4-bit QLoRA. Accept the license on `google/gemma-4-E2B-it` and sign in if the download returns 401.

## What the notebook demos

- Install the Gemma 4 / Unsloth stack.
- Dataset and configuration.
- Load Gemma 4 E2B in 4-bit.
- Score the base model.
- Add LoRA, train on the route name only.
- Compare base vs fine-tuned routes.
- Route a new task, save the adapter, optional exports.

## Links

- Model card: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- Notebook: [`colab/gemma4_e2b_model_router.ipynb`](../colab/gemma4_e2b_model_router.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/gemma4_e2b_model_router.ipynb)
- Related: [Gemma 4 E2B tweet style](gemma4_e2b_tweet_style_colab.md), [`ModernBERT_train_classify.ipynb`](../colab/ModernBERT_train_classify.ipynb)
