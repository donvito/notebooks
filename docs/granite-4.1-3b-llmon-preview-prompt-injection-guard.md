# Granite 4.1 3B LLMON-preview prompt-injection guard

[`ibm-granite/granite-4.1-3b-llmon-preview`](https://huggingface.co/ibm-granite/granite-4.1-3b-llmon-preview) is IBM’s experimental 3B instruct model (~3.66B, Apache 2.0) that consumes prompts in [LLMON](https://github.com/generative-computing/llmon) instead of a conventional chat template. The bundled `chat_template.jinja` still lets you call `apply_chat_template`. The [blog](https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite) calls it an experimental preview, ready for exploration and evaluation but not for production.

## Use cases

- **Trusted instructions mixed with untrusted material.** The [card](https://huggingface.co/ibm-granite/granite-4.1-3b-llmon-preview) positions the model for prompts that combine system policy / user instructions with retrieved documents, pasted content, or tool results, where that boundary must stay explicit.
- **Indirect prompt-injection resistance at the prompt format.** LLMON wraps data, documents, and tool results as addressable input blocks. The card’s privilege rule: `exec` pointers to the instruction and system blocks are authoritative; everything reached as `exec` input is material to read, not a command to follow. The blog reports that on the OPI benchmark, LLMON preview models largely ignore static indirect prompt injections that hijack conventional instruct models (similar trends on ASB and StruQ). Those are published figures, not numbers this notebook reruns.
- **RAG, summarization, extraction, and tool-calling assistants.** The card lists summarization, classification, extraction, question-answering, RAG, code-related tasks, function calling, and multilingual dialog. Tools use the OpenAI function schema; results come back as untrusted `data` blocks.

## Not a good fit for

- **Production.** The blog: experimental preview, not for production. The checkpoint sits in IBM’s [Granite Experiments](https://huggingface.co/collections/ibm-granite/granite-experiments) collection.
- **A sole agent firewall.** The card: the privilege rule is a learned behavior, not an enforced guarantee. It substantially reduces, but does not eliminate, the chance that an instruction embedded in data, a document, or a tool result is followed. Do not treat it as a sole defense for an agent with access to sensitive systems.
- **Conventional chat templates.** The card: prompts in a conventional chat format are out of distribution and will degrade quality. Use the bundled template.
- **Vision or speech.** This checkpoint is text generation. For Granite ASR / translation use [Granite Speech 5.0 2B](granite-speech-5.0-2b-asr-translation.md). For conventional Granite chat / thinking / tools (a different checkpoint) use [Granite 4.2 3B](granite-4.2-3b-chat-tools.md).
- **The 8B preview on a free T4.** [`ibm-granite/granite-4.1-8b-llmon-preview`](https://huggingface.co/ibm-granite/granite-4.1-8b-llmon-preview) is the sibling size; this notebook loads the 3B weights.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). Weights are one 7.3 GB `model.safetensors` shard (~3.66B parameters). A T4 has no native bfloat16 and no FlashAttention 2, so this notebook loads **float16** with **SDPA** (eager if SDPA refuses) and pins `transformers==5.20.1`. 4-bit is not required. Native context is 131,072 tokens; a T4 KV cache cannot hold that, so prompts stay short. The card snippets call `generate` with `max_new_tokens=256` and no sampler; the notebook is greedy and uses up to 512 new tokens so answers finish.

## What the notebook demos

- Load Granite-4.1-3B-LLMON-preview in float16 on a T4.
- **Demo A:** summarize a trusted customer email using `instruction` + `data` + `requirement`, and print the rendered LLMON.
- **Demo B:** summarize an untrusted vendor email that hides a prompt injection, with the email in a `documents` block, then contrast with pasting the same email into `content` (treated as an instruction).
- **Demo C:** the card’s `reverse_words` tool-calling example, with one mock result fed back.

## Links

- Model card: [ibm-granite/granite-4.1-3b-llmon-preview](https://huggingface.co/ibm-granite/granite-4.1-3b-llmon-preview)
- Blog: [LLMON: An LLM-Native Markup Language, now in Granite](https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite)
- GitHub: [generative-computing/llmon](https://github.com/generative-computing/llmon)
- Source post: [x.com/aitrackerbot](https://x.com/aitrackerbot/status/2120941082871468145)
- Notebook: [`colab/granite-4.1-3b-llmon-preview-prompt-injection-guard.ipynb`](../colab/granite-4.1-3b-llmon-preview-prompt-injection-guard.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/granite-4.1-3b-llmon-preview-prompt-injection-guard.ipynb)
- Related: [Granite 4.2 3B chat and tools](granite-4.2-3b-chat-tools.md), [HaloGuard2-Stream 0.8B streaming output guard](HaloGuard2-Stream-0.8B-streaming-output-guard.md)
