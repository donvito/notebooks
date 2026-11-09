# Granite 4.2 3B chat and tools

[`ibm-granite/granite-4.2-3b`](https://huggingface.co/ibm-granite/granite-4.2-3b) is IBM’s ~3.66B text model (Apache 2.0) for chat, thinking, and tool calling. This notebook is text only. For ASR and speech translation use the Granite Speech notebook.

## Use cases

- **Short on-box chat** with optional `enable_thinking`.
- **Mock tool-calling loops** to see Granite 4.2 emit and consume tool calls before you wire real APIs.
- **An Apache-2.0 3B alternative** to LFM3-1.7B chat when you want IBM Granite rather than LFM Open License.

## Not a good fit for

- **Speech.** Use [Granite Speech 5.0 2B](granite-speech-5.0-2b-asr-translation.md).
- **Long 128K/512K contexts on a T4.** Native context is 128K (extendable to 512K); a T4 KV cache cannot hold that, so generations stay short.
- **Vision.** This checkpoint is text generation.

## Running it on a free Colab T4

Pick a T4 GPU. The card example loads bfloat16; this notebook uses float16 (~7–8 GB of weights) with SDPA. 4-bit bitsandbytes is the fallback if float16 OOMs. No FlashAttention-2.

## What the notebook demos

- Load Granite 4.2 3B in float16 (4-bit fallback).
- Text chat.
- Thinking (`enable_thinking`).
- A mock tool-calling loop.

## Links

- Model card: [ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b)
- Notebook: [`colab/granite-4.2-3b-chat-tools.ipynb`](../colab/granite-4.2-3b-chat-tools.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/granite-4.2-3b-chat-tools.ipynb)
- Related: [Granite Speech](granite-speech-5.0-2b-asr-translation.md), [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md)
