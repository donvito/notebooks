# Gemma 4 E4B decision model: email triage

This notebook trains [`unsloth/gemma-4-E4B-it`](https://huggingface.co/unsloth/gemma-4-E4B-it) (Apache 2.0) as an Unsloth decision model for support email triage. A decision model does not generate text: a Clef-style head scores options. Each email gets `category` (8 queues) and `priority` (low / normal / urgent) with calibrated probabilities. It uses the shared [support email triage dataset](../colab/data/support_email_triage) (1,600 train / 400 test) so you can compare it with the Qwen3.5-4B notebook.

## Use cases

- **Train a calibrated email router** that returns a queue and a priority score instead of a generated reply.
- **Compare Gemma 4 E4B vs Qwen3.5-4B** on the identical split.
- **Serve with the Decision API** (Jev / Laya compatible) after you save the LoRA and merged model.

## Not a good fit for

- **Generative support replies.** This head classifies; it does not write the customer email.
- **A guaranteed free-T4 train.** The notebook’s table lists T4 16 GB as ~12–15 GB peak, tight (batch 2 × accum 8). A 24 GB card is the comfortable target.
- **Inference-only d1-style APIs.** Use the d1 / IOTA / elephant notebooks if you are not training.

## Running it on a free Colab T4

Settings follow Unsloth’s guide: 4-bit LoRA, rank 16, learning rate 2e-4, 2 epochs. The new head trains at 1e-4. Fits a 24 GB RTX 3090 at batch 4 × accum 4 (~12–16 GB estimated). T4 16 GB is listed as tight (~12–15 GB).

## What the notebook demos

- Install Unsloth and load the shared dataset.
- Load Gemma 4 E4B Instruct as a decision model.
- Baseline, train, calibrate.
- Accuracy after training and triage of new emails.
- Save LoRA and merged model.
- Optional Decision API serving notes.

## Links

- Model: [unsloth/gemma-4-E4B-it](https://huggingface.co/unsloth/gemma-4-E4B-it)
- Guide: [Train your own Decision Model](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
- Notebook: [`colab/gemma4_e4b_decision_email_triage.ipynb`](../colab/gemma4_e4b_decision_email_triage.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/gemma4_e4b_decision_email_triage.ipynb)
- Related: [`qwen3_5_4b_decision_email_triage.ipynb`](../colab/qwen3_5_4b_decision_email_triage.ipynb), [GLiNER2.5-Decide](GLiNER2_5-Decide-classification.md)
