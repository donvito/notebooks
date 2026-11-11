# BotPsy-0.8B tool-calling assistant

[`qvac/BotPsy-0.8B`](https://huggingface.co/qvac/BotPsy-0.8B) is Tether AI Research’s 0.8B (~853M) tool-calling / agentic model (Apache 2.0, English recommended). It is a fine-tune of [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B): Qwen3.5 architecture (hybrid Gated DeltaNet + gated attention), used text-only. This notebook loads the Hugging Face Transformers checkpoint on a free Colab T4 and runs a single tool call, a multi-step tool loop, and a request with no matching tool.

## Use cases

- **A tiny on-device personal assistant.** The [card](https://huggingface.co/qvac/BotPsy-0.8B) describes BotPsy as a helper that uses tools for simple tasks in calendar, email & messaging, weather / web search, files, device control, and similar domains. A 1.7 GB float16 shard fits a laptop GPU or a free T4 without 4-bit.
- **Mock tool-calling loops before you wire APIs.** The chat template takes an OpenAI-style `tools` list and emits Qwen3.5 XML (`<tool_call><function=…><parameter=…>`). The notebook parses that, runs fake Python functions, and feeds role `tool` results back — the same loop you would use with a real calendar or weather API.
- **Checking that the model declines instead of inventing a tool.** The card reports **86.1** on BFCL v4’s hallucination tests (self-reported; highest of the models they tested): when no suitable tool exists, it should answer in text rather than fabricate a call. Demo C is that check with weather/calendar tools and a payments + messaging request.

## Not a good fit for

- **Hard or high-risk work.** The card says BotPsy is best suited to tasks of **low to medium difficulty**, still makes mistakes, and is **not** intended for safety-critical or high-stakes automated decisions, or for applications involving money, personal information, health, and other sensitive topics.
- **Languages other than English as the default.** English is the recommended language. The card also evaluated Multi-IF (Chinese, French, Hindi, Italian, Portuguese, Russian, Spanish): it performs well in most of them, but scores much lower in Hindi and Russian.
- **Vision, audio, or OCR.** The card says this checkpoint is used text-only. Use a VL or ASR notebook in this repo for screenshots, speech, or page images.
- **A knowledge model.** The card: knowledge still scales mainly with parameter count, so BotPsy is designed to call tools for facts it does not know. Open-ended search is a weaker spot (BrowseComp-Plus 4.5, self-reported).
- **Compiling `flash-linear-attention` / `causal-conv1d` on a free T4.** Those kernels are optional. Building them on Colab has killed sessions in this repo; the notebook uses Transformers’ PyTorch fallback.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). The card example loads bfloat16 and was tested with `transformers` 5.19.0. A T4 has no native bfloat16 and no FlashAttention 2, so this notebook loads **float16** (~1.7 GB of weights) with **SDPA** (eager if SDPA refuses the hybrid stack) and pins `transformers==5.20.1`. Do not install `flash-attn`, `flash-linear-attention`, or `causal-conv1d`. First run downloads the 1.7 GB safetensors shard. After load you should see a few GB allocated; short tool loops stay well under 15 GB. Those VRAM figures are from the weight file plus a 4096-token generate cap, not a measured Colab run.

Sampling follows the card / `generation_config.json`: `temperature=0.3`, `top_p=0.95`, `top_k=20`, `do_sample=True`. Always pass `enable_thinking=True` (the template defaults to non-thinking). The card snippet uses `max_new_tokens=4096`; evaluations allowed up to 32,768, which does not fit a T4 KV cache here.

## What the notebook demos

- Load BotPsy-0.8B in float16 on a T4.
- **Demo A:** one `create_event` call (card-style calendar schema), parse the XML, run a fake function, feed the result back.
- **Demo B:** multi-step loop with `get_weather`, `list_events`, and `create_event` over obviously sample Lisbon / calendar data.
- **Demo C:** a rent-transfer + text request with only weather/calendar tools, so the model should decline rather than hallucinate a call.

## Links

- Model card: [qvac/BotPsy-0.8B](https://huggingface.co/qvac/BotPsy-0.8B)
- Source post: [x.com/BitcoinNews](https://x.com/BitcoinNews/status/2120542632380190952)
- Blog: [BotPsy on Hugging Face](https://huggingface.co/blog/qvac/botpsy)
- GGUFs: [qvac/BotPsy-0.8B-GGUFs](https://huggingface.co/qvac/BotPsy-0.8B-GGUFs)
- Notebook: [`colab/BotPsy-0.8B-tool-calling-assistant.ipynb`](../colab/BotPsy-0.8B-tool-calling-assistant.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/melvin/botpsy-0.8b-tool-calling-511e/colab/BotPsy-0.8B-tool-calling-assistant.ipynb)
- Related: [Spark-X2.6-1.7B on-device agent](Spark-X2.6-1.7B-on-device-agent.md), [Spark-X2.6-4B](Spark-X2.6-4B-on-device-agent.md), [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md), [Granite 4.2 3B chat and tools](granite-4.2-3b-chat-tools.md)
