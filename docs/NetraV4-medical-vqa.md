# NetraV4 medical VQA

[`dipankar500/NetraV4`](https://huggingface.co/dipankar500/NetraV4) is a ~4.4B Qwen3-VL-4B-Instruct fine-tune (Apache 2.0) for medical image VQA and exam-style reasoning. It reasons in `<think>` then answers in `<answer>`. **Research only, not for diagnosis.**

## Use cases

- **Research VQA on medical images** with an explicit think/answer split.
- **Exam-style reasoning demos**, not clinical decisions.
- **A 4B VL fine-tune** that still fits a T4 in float16 or 4-bit.

## Not a good fit for

- **Diagnosis, treatment, or patient counseling.** The notebook forbids it.
- **The card’s bf16 + 4,096 new tokens on 16 GB.** Too much KV. This notebook uses 512 new tokens and 768 px images.

## Running it on a free Colab T4

Pick a T4 GPU. Float16 + SDPA (~9 GB); 4-bit NF4 fallback. Images capped at 768 px / `max_new_tokens=512` so peak stays under ~14 GB (estimate: ~11–14 GB float16 or ~6–10 GB 4-bit). Do not install `flash-attn`.

## What the notebook demos

- Medical image VQA.
- Exam-style `<think>` / `<answer>` reasoning.
- Research-only disclaimer.

## Links

- Model card: [dipankar500/NetraV4](https://huggingface.co/dipankar500/NetraV4)
- Source post: [x.com/VaidenceAI](https://x.com/VaidenceAI/status/2117133765973840126)
- Notebook: [`colab/NetraV4-medical-vqa.ipynb`](../colab/NetraV4-medical-vqa.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/NetraV4-medical-vqa.ipynb)
