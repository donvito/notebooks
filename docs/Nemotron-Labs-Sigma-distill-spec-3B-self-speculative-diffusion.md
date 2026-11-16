# Nemotron-Labs-Sigma-distill-spec-3B self-speculative diffusion

[`nvidia/Nemotron-Labs-Sigma-distill-spec-3B`](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-distill-spec-3B) is NVIDIA’s ~3B continuous diffusion language model trained for **self-speculative decoding** (NVIDIA Nemotron Open Model License, public, `custom_code`). The diffusion head drafts a 32-token block in one call; the same checkpoint’s autoregressive head verifies it. The text is the greedy AR output — acceptance only changes how many tokens each round commits. Both heads ship in the repo (~4.24B stored parameters). Distilled from an MS-SUSReg **base** 3B teacher, not the instruct checkpoint. Tokenize with `tokenizer(prompt)` (BOS included); do not `apply_chat_template`. Paper: [arXiv:2610.02665](https://arxiv.org/abs/2610.02665).

## Use cases

- **Try self-speculative decoding without a second draft model.** One backbone drafts and verifies, so you can print mean accepted length, rounds, and tokens/sec on a free T4.
- **Check that speculative output matches greedy AR** on a short math prompt, using the same `ar_head` the verifier uses (remote code has no `generate_ar()`).
- **Read the draft/verify contract** (`draft_width=32`, one diffusion call per round, self-conditioning off) as shipped in `generation_config.json`.

## Not a good fit for

- **A chat-with-tools assistant.** This notebook is a decoding demo. For a Nemotron tool loop see [Nemotron-3.5 Bolt E4B](nemotron-3.5-bolt-e4b-tool-agent.md). For chat, use [`Nemotron-Labs-Sigma-3B-instruct`](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-3B-instruct), not this distill-spec student.
- **`apply_chat_template` on this checkpoint.** The bundled template is leftover tokenizer metadata. Sibling distill cards use plain `tokenizer(prompt)`. Dropping BOS that way produced garbage on a T4.
- **Using `model.generate()` as the AR baseline.** That path is Phased-DMD block diffusion (`native_nfe=2`), not greedy AR, so it will not match `generate_speculative`.
- **CPU, native bfloat16, or FlashAttention-2.** The card does not support CPU. A free T4 is `sm_75`.
- **transformers 5.x.** Colab’s current 5.23.0 deprecates `rope_scaling` and warns that `llama_4_scaling_beta` is unrecognized in yarn. A T4 run on that stack got Paris as token 1, then `</` / `<|` junk, with AR and speculative ids identical. Pin 4.55–4.57.
- **Full float32 weights.** ~17 GB does not fit a ~15 GB T4. Stay on 16-bit weights.
- **Reproducing the card’s 3.28× GSM8K speedup on Colab.** That figure is SGLang BF16 on one NVIDIA GB200. This notebook reports whatever the T4 run measures.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). Stored shards are about **8.5 GB** of BF16 (`total_size` 8,477,603,976 bytes). Load **float16**. Encode with `tokenizer(prompt)` so BOS is present. The Hub sampler already runs the Bernstein noise schedule in float64 and the 16-dim embedding path, logits, RMSNorm, and RoPE in float32; encoder MLP / attention stay in the weight dtype. After load, the notebook **asserts** ~16 greedy tokens on both the KV-cache AR path and a no-cache full recompute: `The capital of France is` must contain `Paris` with no `</` / `<|` junk, and `1, 2, 3, 4,` must continue the count. A one-token Paris check is not enough. If that fails, it tries emulated bfloat16 (same 8.5 GB) and then **stops**. Pin `transformers>=4.55,<5` (card floor `>=4.55`; Colab 5.x is too new for this remote YaRN/cache code) and `huggingface_hub>=0.34,<1` (transformers 4.57.x), `trust_remote_code=True`, `AutoModel` (not `AutoModelForCausalLM`). Restart after the install cell. Do not reinstall torch or flash-attn. First run downloads those shards. Keep `max_new_tokens` at 64 for Run All; the card snippet uses 256.

## What the notebook demos

- Pin transformers 4.55–4.57, check GPU / free VRAM.
- Load in float16, print missing/unexpected keys and yarn params, and assert the multi-token cache vs no-cache greedy probe (emulated bf16 fallback, then stop).
- Speculative generation on a short math completion and a general completion, printing AL, rounds, and tokens/sec.
- Greedy AR vs `generate_speculative` on the same ids (match check + measured T4 speedup).
- A short draft/verify walkthrough.

## Links

- Model card: [nvidia/Nemotron-Labs-Sigma-distill-spec-3B](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-distill-spec-3B)
- Paper: [arXiv:2610.02665](https://arxiv.org/abs/2610.02665)
- Source post: [x.com/HuggingPapers](https://x.com/HuggingPapers/status/2122382659796021592)
- Notebook: [`colab/Nemotron-Labs-Sigma-distill-spec-3B-self-speculative-diffusion.ipynb`](../colab/Nemotron-Labs-Sigma-distill-spec-3B-self-speculative-diffusion.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Nemotron-Labs-Sigma-distill-spec-3B-self-speculative-diffusion.ipynb)
- Related: [Nemotron-Labs-Sigma-3B-instruct](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-3B-instruct) (chat template), [Nemotron-3.5 Bolt E4B](nemotron-3.5-bolt-e4b-tool-agent.md) (tool agent, not speculative decoding), [ARLoop-1.2B](ARLoop-1.2B-looped-transformer.md) (`custom_code` T4 LM), [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md) (chat-tuned assistant)
