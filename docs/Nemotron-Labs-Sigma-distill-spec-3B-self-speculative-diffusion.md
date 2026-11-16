# Nemotron-Labs-Sigma-distill-spec-3B self-speculative diffusion

[`nvidia/Nemotron-Labs-Sigma-distill-spec-3B`](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-distill-spec-3B) is NVIDIA’s ~3B continuous diffusion language model trained for **self-speculative decoding** (NVIDIA Nemotron Open Model License, public, `custom_code`). The diffusion head drafts a 32-token block in one call; the same checkpoint’s autoregressive head verifies it. The text is the greedy AR output — acceptance only changes how many tokens each round commits. Both heads ship in the repo (~4.24B stored parameters). Paper: [arXiv:2610.02665](https://arxiv.org/abs/2610.02665).

## Use cases

- **Try self-speculative decoding without a second draft model.** One backbone drafts and verifies, so you can print mean accepted length, rounds, and tokens/sec on a free T4.
- **Check that speculative output matches greedy AR** on a short math prompt, using the same `ar_head` the verifier uses (remote code has no `generate_ar()`).
- **Read the draft/verify contract** (`draft_width=32`, one diffusion call per round, self-conditioning off) as shipped in `generation_config.json`.

## Not a good fit for

- **A chat-with-tools assistant.** This notebook is a decoding demo. For a Nemotron tool loop see [Nemotron-3.5 Bolt E4B](nemotron-3.5-bolt-e4b-tool-agent.md).
- **Using `model.generate()` as the AR baseline.** That path is Phased-DMD block diffusion (`native_nfe=2`), not greedy AR, so it will not match `generate_speculative`.
- **CPU, native bfloat16, or FlashAttention-2.** The card does not support CPU. A free T4 is `sm_75`.
- **Full float32 weights.** ~17 GB does not fit a ~15 GB T4. Stay on 16-bit weights.
- **Reproducing the card’s 3.28× GSM8K speedup on Colab.** That figure is SGLang BF16 on one NVIDIA GB200. This notebook reports whatever the T4 run measures.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). Stored shards are about **8.5 GB** of BF16 (`total_size` 8,477,603,976 bytes). Load **float16**. The Hub sampler already runs the Bernstein noise schedule in float64 and the 16-dim embedding path, logits, RMSNorm, and RoPE in float32; encoder MLP / attention stay in the weight dtype. If AR logits are non-finite, the notebook tries emulated bfloat16 (same 8.5 GB) and then float32 compute on the SwiGLU MLPs. `transformers>=4.55`, `jinja2>=3.1.6` (for `apply_chat_template`), `trust_remote_code=True`, `AutoModel` (not `AutoModelForCausalLM`). Restart after the install cell. Do not reinstall torch or flash-attn. First run downloads those shards. Keep `max_new_tokens` at 64 for Run All; the card snippet uses 256.

## What the notebook demos

- Install the card floor, check GPU / free VRAM.
- Load in float16 with a finite-logits probe and T4 fallbacks.
- Speculative generation on a short math prompt and a general prompt, printing AL, rounds, and tokens/sec.
- Greedy AR vs `generate_speculative` on the same ids (match check + measured T4 speedup).
- A short draft/verify walkthrough.

## Links

- Model card: [nvidia/Nemotron-Labs-Sigma-distill-spec-3B](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-distill-spec-3B)
- Paper: [arXiv:2610.02665](https://arxiv.org/abs/2610.02665)
- Source post: [x.com/HuggingPapers](https://x.com/HuggingPapers/status/2122382659796021592)
- Notebook: [`colab/Nemotron-Labs-Sigma-distill-spec-3B-self-speculative-diffusion.ipynb`](../colab/Nemotron-Labs-Sigma-distill-spec-3B-self-speculative-diffusion.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Nemotron-Labs-Sigma-distill-spec-3B-self-speculative-diffusion.ipynb)
- Related: [Nemotron-3.5 Bolt E4B](nemotron-3.5-bolt-e4b-tool-agent.md) (tool agent, not speculative decoding), [ARLoop-1.2B](ARLoop-1.2B-looped-transformer.md) (`custom_code` T4 LM), [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md) (chat-tuned assistant)
