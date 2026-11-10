# ARLoop-1.2B looped transformer

[`osunlp/ARLoop-1.2B`](https://huggingface.co/osunlp/ARLoop-1.2B) is OSU NLP Group’s fully-open 1.2B pretrained looped transformer with attention residuals (Apache-2.0, public, `custom_code`). 24 physical layers are shared across 3 recurrent passes (effective depth 72, 4096 context, SmolLM2 tokenizer). It is a **base** model for text completion, not chat-tuned; the card says it can be used for mid-training and post-training. The 1.2B run was trained on 500B tokens. A smaller sibling [`osunlp/ARLoop-0.6B`](https://huggingface.co/osunlp/ARLoop-0.6B) (18×3 layers, 180B tokens) and intermediate Hub revisions such as `step-010000` also exist.

## Use cases

- **Base English text completion** when you want a fully-open looped transformer rather than a chat-tuned assistant. The card’s own example is continuing `The capital of France is`.
- **A starting checkpoint for mid-training or post-training.** The card describes this release as a base pretrained model intended for those next steps, not as an instruction model.
- **Compare training progress on the same architecture.** Hub revisions such as `step-010000` (~21B tokens) through `main` / `step-238417` (~500B tokens) are independently loadable with the same `revision=` on the model and tokenizer.
- **Inspect attention-residual looping without extra kernels.** The Hub inference file is portable PyTorch SDPA; it says no training repo, FLA, Triton, or flash-attn install is required.

## Not a good fit for

- **Chat, tools, or instruction following.** There is no chat template. This is a base completion model.
- **Treating T4 float16 as the card’s bfloat16 recipe.** The card loads `dtype=torch.bfloat16`. A free T4 has no native bf16. The custom forward still autocasts activations to bf16 (emulated on Turing) and keeps routing / normalization stats in FP32.
- **Loading [`ARLoop-0.6B`](https://huggingface.co/osunlp/ARLoop-0.6B) in this same default runtime.** That card pins `transformers==5.22.0`; this notebook pins the 1.2B card’s `transformers==5.12.1`. The notebook compares an early 1.2B revision instead.
- **Long 4096-token generations on a free T4.** The KV cache is indexed by effective layer (72), so keep completions short.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). Stored shards are about **4.9 GB** of FP32 weights (the card keeps original FP32 in the checkpoint). Load **float16** (~2.4 GB of parameters). If that path yields NaNs or garbage, reload in **float32** (~5 GB of parameters), which still fits a ~15 GB T4. First run downloads those shards. Pin `transformers==5.12.1` as the 1.2B card does, then **restart the session** so Colab’s already-imported transformers is replaced. Do not reinstall torch, flash-attn, FLA, or Triton. `trust_remote_code=True` is required.

## What the notebook demos

- Install the 1.2B card pin and check GPU / free VRAM.
- Load ARLoop-1.2B in float16, with a float32 fallback if logits are non-finite.
- Factual completion (the card’s France prompt).
- Few-shot pattern completion.
- The same prompt on the `step-010000` revision after releasing the final checkpoint.

## Links

- Model card: [osunlp/ARLoop-1.2B](https://huggingface.co/osunlp/ARLoop-1.2B)
- Sibling: [osunlp/ARLoop-0.6B](https://huggingface.co/osunlp/ARLoop-0.6B)
- Paper: [arXiv:2611.05234](https://arxiv.org/abs/2611.05234)
- Code: [OSU-NLP-Group/ARLoop](https://github.com/OSU-NLP-Group/ARLoop)
- Source post: [x.com/yuekun_yao](https://x.com/yuekun_yao/status/2119982422259622086)
- Notebook: [`colab/ARLoop-1.2B-looped-transformer.ipynb`](../colab/ARLoop-1.2B-looped-transformer.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/ARLoop-1.2B-looped-transformer.ipynb)
- Related: [oruk-loop-153M](oruk-loop-153M-looped-asr.md) (looped transformer, English ASR), [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md) (chat-tuned LM, not a base completer)
