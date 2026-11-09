# Carbon-X DNA chat

[`HuggingFaceBio/Carbon-X`](https://huggingface.co/HuggingFaceBio/Carbon-X) is Hugging Face Bio's 14.8B decoder-only DNA chat model (Apache 2.0, not gated). It is a fine-tune of [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B). You paste raw DNA in a user message and it answers in English. A hybrid tokenizer encodes DNA as 6-mers and needs `trust_remote_code=True`. The architecture is stock `Qwen3ForCausalLM`. This walkthrough is inference only.

Outputs are research demos. Carbon-X has not been clinically validated. Do not use it to diagnose, treat, or counsel anyone.

## Use cases

- **Organism-group questions on a pasted sequence.** Chat over raw DNA when you want an English explanation, not a separate DNA encoder API.
- **Benign vs pathogenic variant discussion for research.** The notebook demos this as a research prompt, not a clinical call.
- **Next-base prediction.** Ask what base comes next as a capability check of the hybrid DNA tokenizer.
- **Local exploration of a DNA chat LM.** The Hub checkpoint is ungated Apache 2.0.

## Not a good fit for

- **Clinical advice or real patient variants.** The notebook states this clearly.
- **Full-precision bfloat16 on a free T4.** The Hub shards total about 30 GB and do not fit.
- **vLLM serving, GGUF, thinking mode, or long designed DNA.** Those paths are explicitly out of scope here.

## Running it on a free Colab T4

Pick a T4 GPU. Load 4-bit NF4 with float16 compute via bitsandbytes. Packed 4-bit weights are about 7.4 GB; expected peak is about 10–13 GB of 15 GB (an estimate, not a measured Colab number). Needs Transformers v5+ (5.13+). Keep `enable_thinking=False`. Attention is SDPA. Do not install `flash-attn`. First run downloads about 30 GB of weights plus Qwen3 BPE files for the hybrid tokenizer. `low_cpu_mem_usage=True` streams the 25 GB first shard so host RAM stays under Colab's limit.

## What the notebook demos

- Install Transformers v5+ and bitsandbytes.
- GPU check and T4 caveats.
- Load Carbon-X in 4-bit NF4.
- Chat helper with thinking off.
- Organism group.
- Variant: benign vs pathogenic.
- Next-base prediction.
- Caveats and cleanup.

## Links

- Model card: [HuggingFaceBio/Carbon-X](https://huggingface.co/HuggingFaceBio/Carbon-X)
- Source post: [x.com/_lewtun](https://x.com/_lewtun/status/2113214040516268126)
- Blog: [Carbon-X-Release](https://huggingface.co/spaces/HuggingFaceBio/Carbon-X-Release)
- Code: [huggingface/carbon](https://github.com/huggingface/carbon)
- Notebook: [`colab/Carbon-X-inference-dna.ipynb`](../colab/Carbon-X-inference-dna.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Carbon-X-inference-dna.ipynb)
