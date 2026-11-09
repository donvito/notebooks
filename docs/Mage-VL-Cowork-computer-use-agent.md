# Mage-VL-Cowork computer-use agent

[`microsoft/Mage-VL-Cowork`](https://huggingface.co/microsoft/Mage-VL-Cowork) is Microsoft’s 4B computer-use policy (Apache 2.0, Qwen3-4B-Instruct-2507 + Mage-ViT). It predicts screenshot-grounded `computer_use` clicks plus a `bash` tool in one policy. The card reports 66.0% OSWorld-Verified. This notebook is a safe single-step demo: synthetic PIL windows only; predicted clicks are drawn and predicted bash is printed, not executed.

## Use cases

- **Inspect a CUA policy** without giving it a real desktop.
- **Compare click predictions** on synthetic windows before you wire OSWorld.
- **A 4B Apache-2.0 computer-use checkpoint** that also emits bash.

## Not a good fit for

- **Executing actions.** Nothing is clicked or run.
- **The card’s SGLang / FlashInfer / 1920×1080 / bf16 recipe.** That does not fit a T4.
- **transformers 5.x.** Colab 5.21’s processor omits `patch_positions`; this notebook pins 4.57.x.

## Running it on a free Colab T4

Pick a T4 GPU. Pin `transformers>=4.57.3,<5` and `trust_remote_code`. Load float16 + SDPA (about 9.5 GB, same as bf16); screenshots capped at 1280×720 / `max_pixels≈0.92M`; `max_new_tokens=640`. 4-bit NF4 fallback if float16 overflows ~15 GB (about 4.8 GB packed). Do not install `flash-attn`.

## What the notebook demos

- Load Mage-VL-Cowork (float16, 4-bit fallback).
- Single-step click prediction on synthetic PIL windows (drawn, not executed).
- Predicted bash printed, not run.

## Links

- Model card: [microsoft/Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork)
- Code: [mage_vl_cowork](https://github.com/microsoft/Mage/tree/main/mage_vl_cowork) · [project](https://microsoft.github.io/Mage/cowork/)
- Notebook: [`colab/Mage-VL-Cowork-computer-use-agent.ipynb`](../colab/Mage-VL-Cowork-computer-use-agent.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Mage-VL-Cowork-computer-use-agent.ipynb)
- Related: [NorthCUA-Nano](NorthCUA-Nano-computer-use-agent.md)
