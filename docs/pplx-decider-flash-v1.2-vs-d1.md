# pplx-decider-flash vs d1-3B

This notebook is an inference-only side-by-side of [`perplexity-ai/pplx-decider-flash-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) (~4.54B, Apache 2.0) and [`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) on the same support-triage, agent-guardrail, and priority questions. It is not a training notebook. The two checkpoints do not stay resident together: run pplx, free it, then load d1.

## Use cases

- **Compare two decision models** on identical tickets and guardrail JSON.
- **See 4-bit pplx vs float16 d1** on a free T4 without training anything.
- **Pick a checkpoint** before you standardize on d1, IOTA, or elephant.

## Not a good fit for

- **Training a decision head.** Use the Unsloth email-triage notebooks.
- **Loading both models at once.** Sequential load only.
- **pplx at full float16 on a T4.** Float16 is ~9.5 GB; the notebook defaults to 4-bit NF4 (~3.5 GB).

## Running it on a free Colab T4

Pick a T4 GPU. Sequential load. 4-bit NF4 + SDPA for pplx (~3.5 GB; the official loader hardcodes CUDA bfloat16 and is patched here) or float16 (~9.5 GB). Float16 + SDPA for d1 (~6.2 GB). No FlashAttention-2.

## What the notebook demos

- Load pplx-decider-flash-v1.2 (4-bit).
- Support triage, agent guardrail, priority.
- Free pplx, load d1-3B, repeat the same questions.

## Links

- pplx card: [perplexity-ai/pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2)
- d1 notebook: [d1-3B](d1-3B-inference-decisions.md)
- Notebook: [`colab/pplx-decider-flash-v1.2-vs-d1.ipynb`](../colab/pplx-decider-flash-v1.2-vs-d1.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/pplx-decider-flash-v1.2-vs-d1.ipynb)
