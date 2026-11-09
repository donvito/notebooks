# Ternary Bonsai 2 4B

This notebook chats, describes an image, and calls tools with PrismML llama.cpp and the `PQ2_0` pack of [`prism-ml/Ternary-Bonsai-2-4B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf) (Apache 2.0), a ternary Qwen3.5-4B.

## Use cases

- **On-device ternary Qwen3.5-4B chat** from a ~1.5 GB GGUF.
- **Short image description** through the vision pack.
- **Tool calling** on PrismML llama.cpp.

## Not a good fit for

- **Stock llama.cpp without PrismML’s ternary kernels.**
- **A full-precision 4B on a T4.** This path is the PQ2_0 pack.

## Running it on a free Colab T4

Pick a T4 GPU. About 1.5 GB of GGUF, roughly 2–4 GB with a short context.

## What the notebook demos

- Load PQ2_0 GGUF with PrismML llama.cpp.
- Chat.
- Image description.
- Tool calling.

## Links

- GGUF: [prism-ml/Ternary-Bonsai-2-4B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf)
- Notebook: [`colab/Ternary-Bonsai-2-4B-llamacpp.ipynb`](../colab/Ternary-Bonsai-2-4B-llamacpp.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Ternary-Bonsai-2-4B-llamacpp.ipynb)
