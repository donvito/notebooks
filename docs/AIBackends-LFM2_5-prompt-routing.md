# AIBackends + LFM2.5 prompt routing

Zero-shot prompt routing with Liquid AI's [LFM2.5-Encoder-350M-Prompt-Router](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router) (~355M, LFM Open License v1.0) through `aibackends[routing]`. Lanes are plain strings you invent at call time; the encoder scores the whole prompt against every lane in one pass. No taxonomy training required.

## Use cases

- **Device-assistant or product lanes.** Route a user utterance to wifi, calendar, media, and similar skills without a trained classifier.
- **Add a category on the fly.** New lanes are just more strings — useful when the product taxonomy is still moving.
- **Support intents with an escalate path.** Softmax scores plus a `threshold` drop weak lanes; `best_route` is `None` when nothing clears, which the notebook uses as a human-escalate switch. English lane names still catch Spanish or Japanese tickets (encoder is multilingual).
- **Model-tier routing.** A cheap encoder pass picks cloud vs local / small vs large model before you spend on generation.
- **Front door for heavier backends.** Route then dispatch: GliGuard for risky prompts, GLiNER PII for personal data, chat model for the rest.

## Not a good fit for

- **Generating the reply.** This is a router, not a chat model. Pair with [LFM2.5 tool calling](AIBackends-LFM2_5-tool-calling.md) or another generator.
- **Calibrated typed decisions over images.** Use d1 / IOTA / Kanana notebooks for `noul`/`choice`/`score` APIs.
- **Production without reading LFM Open License v1.0.** Hub tag is `license: other`, not Apache/MIT.

## Running it on a free Colab T4

Works on a **free CPU runtime**; a T4 is faster and is detected automatically. First load downloads ~1.4 GB. `routing` pulls `torch` + `transformers`; `guardrails` and `pii` extras are only needed for the dispatch section.

## What the notebook demos

- Load the router once.
- Device-assistant routing (`route_device_assistant.py`).
- Add a category on the fly (`route_custom_category.py`).
- Code-language routing as a batch (`route_code_language.py`).
- Support intents, thresholds, and multilingual tickets (`route_support_intent.py`).
- Route by complexity to model tiers (`route_by_complexity.py`).
- Route then dispatch to GliGuard / GLiNER (`route_and_dispatch.py`).
- Async and CLI.

## Links

- Model card: [LiquidAI/LFM2.5-Encoder-350M-Prompt-Router](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router)
- Notebook: [`colab/AIBackends-LFM2_5-prompt-routing.ipynb`](../colab/AIBackends-LFM2_5-prompt-routing.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-LFM2_5-prompt-routing.ipynb)
- Related: [LFM2.5 tool calling](AIBackends-LFM2_5-tool-calling.md), [GliGuard moderation](AIBackends-GLiGuard_LLMGuardrails-300M-moderation.md), [GLiNER PII](AIBackends-GLiNER-PII-redaction.md)
