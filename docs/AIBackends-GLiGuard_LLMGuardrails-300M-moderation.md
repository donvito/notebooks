# AIBackends + GliGuard LLMGuardrails-300M

[GliGuard](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) is Fastino's encoder-based LLM safety moderator (Apache 2.0). The Hub checkpoint is named 300M; it screens prompts and responses as structured classification rather than generating a verdict. This notebook shows the `aibackends[guardrails]` backend (v0.5.0) wrapping that model so moderation stays local and independent of whichever generative runtime you serve.

## Use cases

- **Prompt screening before generation.** `moderate_prompt` returns safety, toxicity categories, jailbreak strategies, and an `is_safe` flag, so a chat or agent API can refuse or escalate unsafe requests without calling a remote moderation API.
- **Response screening after generation.** `moderate_response` scores the reply for safety and toxicity, and reports refusal vs compliance, so you can log, redact, or block unsafe completions.
- **Exchange-aware logging.** Passing the original prompt with the response scores the *exchange*, which the notebook uses to distinguish a harmless refusal from an unsafe conversation.
- **Guarded chat turns.** Screen the prompt, generate only if it clears, then screen what came back — the shape of an in-process guardrail around any LLM.
- **Batch inbound queues.** `moderate_prompts` uses native batch inference for mixed benign and attack traffic, useful for support inboxes or eval harnesses.

## Not a good fit for

- **Generative answers.** GliGuard classifies; it does not write, rewrite, or refuse in natural language.
- **Streaming token-level blocks.** This backend scores whole prompts and responses. For per-token output guards, see [`HaloGuard2-Stream-0.8B-streaming-output-guard.ipynb`](../colab/HaloGuard2-Stream-0.8B-streaming-output-guard.ipynb).
- **Replacing a full safety policy.** Thresholds (`threshold` default 0.5, `category_threshold` default 0.4) still need product-specific tuning and human review for high-stakes use.

## Running it on a free Colab T4

Runs on a **free CPU runtime**; a T4 is optional and is picked up automatically. The first call downloads ~1 GB of weights. Warm calls are described as well under 200 ms on CPU. Extra: `aibackends[guardrails]` pulls `gliner2[local]` and `protobuf`.

## What the notebook demos

- Load once, reuse everywhere (cold vs warm cost).
- Screening a prompt for safety, toxicity, and jailbreak.
- Scoring a batch of prompts as a table.
- Screening a response, with and without prompt context.
- Tuning `threshold` and `category_threshold`.
- Async `_async` variants.
- A stub guarded chat turn.
- The same tasks from the `aibackends` CLI.

## Links

- Model card: [fastino/gliguard-LLMGuardrails-300M](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M)
- Notebook: [`colab/AIBackends-GLiGuard_LLMGuardrails-300M-moderation.ipynb`](../colab/AIBackends-GLiGuard_LLMGuardrails-300M-moderation.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-GLiGuard_LLMGuardrails-300M-moderation.ipynb)
- Related: [GLiNER PII redaction](AIBackends-GLiNER-PII-redaction.md), [LFM2.5 prompt routing](AIBackends-LFM2_5-prompt-routing.md) (route then dispatch to GliGuard)
