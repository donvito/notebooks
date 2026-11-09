# Nemotron-3.5 Bolt E4B tool agent

This notebook runs a multi-turn device-assistant tool-calling loop with NVIDIA’s Gemma-4-E4B post-train [`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) (OpenMDW-1.1). Same Gemma 4 E4B family as the email-triage trainer; this one is a tool agent, not triage.

## Use cases

- **On-box device-assistant tools** (grounded function calls) from NVIDIA’s 4GB GGUF.
- **Avoid the 15.9 GB BF16 shard** that kills free-Colab host RAM when quantized on the fly.
- **Compare Gemma 4 E4B as an agent** versus the Unsloth email-triage decision head.

## Not a good fit for

- **Loading BF16 + bitsandbytes on Colab.** The 15.9 GB shard plus host-RAM spike dies with no traceback. This notebook does not download it.
- **Email triage training.** Use [Gemma 4 E4B decision email triage](gemma4_e4b_decision_email_triage.md).

## Running it on a free Colab T4

Pick a T4 GPU. Official Q4_K_M GGUF (~6.2 GB) via `llama-cpp-python`. Tokenizer and chat template come from the BF16 repo as small files only. After load, trust `nvidia-smi` (llama.cpp allocates outside PyTorch).

## What the notebook demos

- Load the 4GB Q4_K_M GGUF.
- Multi-turn device-assistant tool calling.
- Notes on why BF16 is not downloaded.

## Links

- 4GB GGUF: [NVIDIA-Nemotron-3.5-Bolt-E4B-4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB)
- BF16 card: [NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16)
- Blog: [Nemotron 3.5 Bolt](https://research.nvidia.com/labs/nemotron/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/)
- Notebook: [`colab/nemotron-3.5-bolt-e4b-tool-agent.ipynb`](../colab/nemotron-3.5-bolt-e4b-tool-agent.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/nemotron-3.5-bolt-e4b-tool-agent.ipynb)
- Related: [Gemma 4 E4B email triage](gemma4_e4b_decision_email_triage.md)
