# NorthCUA-Nano computer-use agent

[`beipo-ai/NorthCUA-Nano`](https://huggingface.co/beipo-ai/NorthCUA-Nano) is a 9B dense Qwen3.5 computer-use policy (Apache 2.0). Coordinates are normalized 0–1000 in the official NorthCUA Harness format. The report lists 78.90% OSWorld-Verified / 67.16% WebArena-Verified (not tied to this exact checkpoint). This notebook is a safe single-step demo: synthetic PIL windows plus one official demo frame; predicted clicks are drawn, nothing is executed.

## Use cases

- **Inspect NorthCUA-format tool calls** (desktop/browser) without a live OS.
- **Normalized 0–1000 click coordinates** as a harness smoke test.
- **A 9B CUA** next to 4B Mage-VL-Cowork.

## Not a good fit for

- **Executing clicks.** Nothing is run.
- **BF16 on a T4.** BF16 is ~18.8 GB and does not fit.
- **Tying this checkpoint to the report’s OSWorld numbers.** The notebook says those scores are not tied to this exact checkpoint.

## Running it on a free Colab T4

Pick a T4 GPU. 4-bit NF4 + float16 compute (~6–7 GB packed). Screenshots capped at 1280 px long side; `max_new_tokens=768`. Expected peak ~8–12 GB (estimate). Pin transformers 5.20.2. Stock Transformers classes — no `trust_remote_code`. Do not install `flash-attn`.

## What the notebook demos

- Load NorthCUA-Nano in 4-bit.
- Single-step policy on synthetic PIL windows and one official demo frame.
- Draw predicted clicks; execute nothing.

## Links

- Model card: [beipo-ai/NorthCUA-Nano](https://huggingface.co/beipo-ai/NorthCUA-Nano)
- Harness: [beipo-ai/NorthCUA](https://github.com/beipo-ai/NorthCUA)
- Report: [arXiv:2611.02102](https://arxiv.org/abs/2611.02102)
- Source post: [x.com/PtVHVQ2FPQ51994](https://x.com/PtVHVQ2FPQ51994/status/2117229952231702676)
- Notebook: [`colab/NorthCUA-Nano-computer-use-agent.ipynb`](../colab/NorthCUA-Nano-computer-use-agent.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/NorthCUA-Nano-computer-use-agent.ipynb)
- Related: [Mage-VL-Cowork](Mage-VL-Cowork-computer-use-agent.md)
