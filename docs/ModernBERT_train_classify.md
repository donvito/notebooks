# ModernBERT classifier

This notebook fine-tunes [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) (149.7M, Apache 2.0, English encoder) as a tiny binary classifier for `luna` (simple / bounded software task) vs `astra` (complex / security-sensitive / architectural). The point is that a small encoder can replace an LLM for a narrow routing task.

## Use cases

- **Train a cheap task router** on a handful of labeled examples instead of calling a 4B LLM.
- **A baseline** before you QLoRA Gemma 4 E2B as a generative router on the same luna/astra policy.
- **Show encoder classification** (a classification head, not generated route names).

## Not a good fit for

- **Generative route names or calibrated multi-head decisions.** Use [Gemma 4 E2B model router](gemma4_e2b_model_router.md) or the Unsloth email-triage notebooks.
- **Zero-shot labels you invent at call time.** Use [GLiNER2.5-Decide](GLiNER2_5-Decide-classification.md).
- **A production dataset.** The notebook builds a tiny demo set.

## Running it on a free Colab T4

Enable a T4 GPU before training. Exact VRAM is not documented; ModernBERT-base is 150M and is intended to be light.

## What the notebook demos

- Create a tiny demo dataset.
- Load ModernBERT-base.
- Fine-tune the classifier and plot graphs.
- Evaluate.
- Classify new tasks and differently worded examples.

## Links

- Model card: [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
- Notebook: [`colab/ModernBERT_train_classify.ipynb`](../colab/ModernBERT_train_classify.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/ModernBERT_train_classify.ipynb)
- Related: [Gemma 4 E2B model router](gemma4_e2b_model_router.md)
