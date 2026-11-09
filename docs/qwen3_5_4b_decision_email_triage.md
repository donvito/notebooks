# Qwen3.5-4B decision model: email triage

This notebook trains [`unsloth/Qwen3.5-4B`](https://huggingface.co/unsloth/Qwen3.5-4B) (Apache 2.0, ~4.66B) as an Unsloth decision model (`FastDecisionModel` + `DecisionTrainer`, 4-bit LoRA r=16) that routes support emails to 8 queues and scores priority, with calibrated probabilities. It uses the shared [support email triage dataset](../colab/data/support_email_triage) (1,600 train / 400 test) so you can compare it with the Gemma 4 E4B notebook. A decision model does not generate text.

## Use cases

- **Train a calibrated email router** (queue + priority) instead of prompting a chat model.
- **Compare Qwen3.5-4B vs Gemma 4 E4B** on the identical split.
- **Serve with the Decision API** (Jev / Laya compatible) after saving LoRA and the merged model.

## Not a good fit for

- **Writing the customer reply.** Classification only.
- **A guaranteed free-T4 train.** The companion Gemma notebook lists 24 GB as comfortable and T4 as tight; this recipe is the same 4-bit LoRA r=16 settings.
- **In-weight x86 tools.** That is [Qwenx86-4B compact](Qwenx86-4B-compact-in-weight-cpu.md).

## Running it on a free Colab T4

Settings follow Unsloth’s guide: 4-bit LoRA, rank 16, learning rate 2e-4, 2 epochs. Fits a 24 GB RTX 3090.

## What the notebook demos

- Load the shared dataset.
- Load Qwen3.5-4B as a decision model.
- Baseline, train, calibrate.
- Accuracy and triage of new emails.
- Save LoRA and merged model; Decision API notes.

## Links

- Model: [unsloth/Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B)
- Guide: [Train your own Decision Model](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) · [serving](https://unsloth.ai/docs/models/decision-laya)
- Notebook: [`colab/qwen3_5_4b_decision_email_triage.ipynb`](../colab/qwen3_5_4b_decision_email_triage.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/qwen3_5_4b_decision_email_triage.ipynb)
- Related: [Gemma 4 E4B email triage](gemma4_e4b_decision_email_triage.md)
