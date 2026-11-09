# IOTA Decide 4B typed decisions

[`macrocosm-os/iota-decide-4b-v0.4`](https://huggingface.co/macrocosm-os/iota-decide-4b-v0.4) is Macrocosmos’s Apache-2.0 Qwen3.5-4B finetune (~4.2B) for typed `noul` / `choice` / `score` decisions. Answers come from one forward pass with zero output tokens. v0.4 uses bidirectional attention on full-attention layers, so you must use the bundled `letter_decide.py` loader — plain `generate` / stock vLLM / SGLang / llama.cpp scores lower.

## Use cases

- **Refund-ticket routing** with calibrated probabilities.
- **Agent-trace failure diagnosis** over a JSON trace.
- **Policy compliance checks** on a request plus a policy snippet.
- **Same job as d1 / elephant**, with an Apache-2.0 4B checkpoint and a custom loader.

## Not a good fit for

- **Chat replies.** It does not generate text.
- **High-stakes medical, legal, credit, or hiring calls without a human.** The notebook says keep a human in the loop.
- **Stock causal inference.** Use the bundled loader.

## Running it on a free Colab T4

Pick a T4 GPU. Hub file is one ~8.4 GB bf16 shard. The loader defaults to bfloat16; this notebook passes float16. Weights stay about 8.4 GB; short demos should peak around 10–12 GB of 15 GB. SDPA (eager fallback). Do not install `flash-attn`, `flash-linear-attention`, or `causal-conv1d` — those extras have killed free Colab host RAM in this repo.

## What the notebook demos

- Check the GPU.
- Load with the bundled loader.
- Refund-ticket routing.
- Agent-trace failure diagnosis.
- Policy compliance check.
- Caveats and cleanup.

## Links

- Model card: [macrocosm-os/iota-decide-4b-v0.4](https://huggingface.co/macrocosm-os/iota-decide-4b-v0.4)
- Source post: [x.com/MacrocosmosAI](https://x.com/MacrocosmosAI/status/2118028289813971098)
- Notebook: [`colab/IOTA-Decide-4B-typed-decisions.ipynb`](../colab/IOTA-Decide-4B-typed-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/IOTA-Decide-4B-typed-decisions.ipynb)
- Related: [d1-3B](d1-3B-inference-decisions.md), [elephant-1](elephant-1-inference-decisions.md), [Kanana-2-v-decide](Kanana-2-v-decide-typed-decisions.md)
