# Lev-3B log anomaly decisions

[`cisco-ai/lev-3b`](https://huggingface.co/cisco-ai/lev-3b) is Cisco’s compact Apache-2.0 **probabilistic decision model** on [`ibm-granite/granite-4.1-3b-base`](https://huggingface.co/ibm-granite/granite-4.1-3b-base) (pinned revision `dacb9cb9157bec98e99b09f285c92a4d58405c96`). It is a LoRA adapter plus a trained pointer head (`head.pt`). You supply a shared **state** (for example a window of log lines) and named questions (`noul` / `choice` / `score`); it returns a **probability vector over the alternatives you listed**. It does not generate answer text. Ordinary `AutoModelForCausalLM` or PEFT-only loading is not enough: the card requires the [Lev runtime](https://github.com/splunk/lev).

## Use cases

- **Log-window anomaly decisions.** Ask a `noul` question such as “Is this log session anomalous?” over caller-defined false/true descriptions. The card’s own example uses two lines (`INFO service started` / `ERROR request failed after repeated retries`) and prints `{"false": …, "true": …}`. The second probability is P(yes).
- **ATT&CK parent-technique matching over a closed set you supply.** A `choice` question scores named alternatives in dictionary insertion order. The paper’s ATT&CK eval offered 130 parent techniques; this notebook uses a handful. The card: ATT&CK probabilities are **conditional on the offered techniques**; they are not probabilities that a session is malicious.
- **Ordinal severity.** A `score` question with 2–255 ordered descriptions returns a probability for each zero-based level. The card: those probabilities can be used to calculate the expected level.
- **Research on bounded log decisions.** Joint log adaptation combined BGL / Spirit / Thunderbird anomaly examples, general-log questions, and ATT&CK parent-technique examples from [Splunk attack-data](https://github.com/splunk/attack_data). Questions and alternatives stay caller-defined; they are not restricted to those tasks.

## Not a good fit for

- **Writing text.** There is no decoder output: no incident write-up, no chat, no JSON generation.
- **`AutoModelForCausalLM` or PEFT-only loading.** The card: adapters alone are not the model. `head.pt` (pointer head plus backbone / tokenizer / execution / dtype metadata) is required, and so is Lev’s tokenizer with added delimiter tokens.
- **Autonomous remediation or a production detector.** The card and the Lev README call this a research model. Validate locally; keep human-reviewed action policies.
- **Treating ATT&CK scores as P(malicious).** Conditional on the techniques you offered.
- **Assuming HDFS transfer is solved.** The card: HDFS transfer remains weak despite high numbers on represented anomaly sources (BGL, Spirit, Thunderbird). The model can miss anomalies with high confidence; the fitted temperature does not remove that failure mode.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). A free T4 has about 15 GB of VRAM, compute capability 7.5: **no native bfloat16** and **no FlashAttention 2**.

The card’s serving recipe is Linux, Python 3.12, `uv`, and `load(..., dtype=torch.bfloat16)`. Colab already ships Python, torch, and CUDA, so this notebook **does not create a uv venv**. It clones [`splunk/lev`](https://github.com/splunk/lev), runs `scripts/setup_kev.py` (pinned, patched Kev), and `pip install -e`s `kev` into Colab’s environment. Keep Colab’s torch. Do not install `flash-attn`, `flash-linear-attention`, or `causal-conv1d` (those are the CUDA 12.8 / PyTorch 2.8 extra).

The Lev loader forces bfloat16 when `head.pt` records `weights_dtype=bf16`. This notebook clears that force so the backbone loads in **float16**. LoRA adapters, structural-token rows, and the pointer head stay **float32**, matching the card. Float32 for the whole 3B backbone is about 12 GB and is a last resort if float16 misbehaves.

Download is about **7 GB**: Granite-4.1-3B-Base shards (~4.9 + ~1.9 GB) plus Lev adapter / tokenizer / `head.pt` (~0.14 GB). Peak VRAM on these short demos is an estimate around **8–10 GB** of 15 GB, not a measured Colab run. Weights are public; no Hugging Face token.

Raw probabilities use `T=1`. The optional fitted scalar `T=1.8234449771164334` in `temperature.json` is **not** applied on load; set `model.head.temperature` yourself. Positive temperature scaling does not change the argmax.

## What the notebook demos

- Check the T4 and pick float16.
- Install Lev/Kev into Colab’s Python (no uv venv) and load with the pointer head.
- **`noul`:** the card’s two-line log window (raw `T=1` and the supplied fitted temperature), plus a routine window for contrast.
- **`choice`:** which of a handful of MITRE ATT&CK parent techniques best matches a short suspicious PowerShell log snippet.
- **`score`:** severity levels 0–3 on the card’s error window; print the vector, argmax, and expected level.

## Links

- Model card: [cisco-ai/lev-3b](https://huggingface.co/cisco-ai/lev-3b)
- Paper: [Lev-3B: A Data Recipe for Compact Probabilistic Log Decision Models](https://arxiv.org/abs/2610.38915)
- Runtime: [splunk/lev](https://github.com/splunk/lev)
- Decision-model implementation: [Kev](https://github.com/jaredpalmer/kev) (Lev applies integration patches; compatibility with upstream Kev alone is not guaranteed)
- Backbone: [ibm-granite/granite-4.1-3b-base](https://huggingface.co/ibm-granite/granite-4.1-3b-base)
- Source post: [x.com/LiangGou](https://x.com/LiangGou/status/2124705890242838645)
- Notebook: [`colab/Lev-3B-log-anomaly-decisions.ipynb`](../colab/Lev-3B-log-anomaly-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Lev-3B-log-anomaly-decisions.ipynb)
- Related: [d1-3B](d1-3B-inference-decisions.md), [ARC 1.0](ARC-1.0-agent-model-router.md), [elephant-1](elephant-1-inference-decisions.md), [Granite 4.1 3B LLMON-preview](granite-4.1-3b-llmon-preview-prompt-injection-guard.md)
