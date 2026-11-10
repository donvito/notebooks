# Spark-X2.6-1.7B on a free T4

Spark-X2.6-1.7B is SparkLLM's compact on-device agent: about 1.71 billion parameters, Apache 2.0, English and Chinese. This notebook loads the Hugging Face Transformers checkpoint on a free Colab T4 and walks through think-then-answer chat, a small coding task, and a tool-use loop.

## Use cases

- **On-device coding helper.** SparkLLM's [announcement](https://x.com/SparkLLM/status/2115100695775056057) reports SWE Verified 59.8 at high thinking, close to 62.8 for the 4B sibling, at less than half the size. That is a reasonable local helper for small parsers, refactors, and “write this function” prompts when you do not want to send code to an API.
- **Lightweight agent / tool-use.** The same post reports τ²-bench 85.3 versus 86.9 for the 4B, both at high thinking. The chat template accepts an OpenAI-style `tools` list and emits `<tool_call>` blocks, so a short local loop (weather, time, filesystem stubs) fits the model without a larger runtime.
- **Bilingual English / Chinese assistant.** The [model card](https://huggingface.co/XHToken/Spark-X2.6-1.7B) lists `en` and `zh`. A laptop or Colab T4 can keep a 3.5 GB float16 checkpoint resident for mixed-language chat.
- **Tight-memory local chat.** Float16 weights are about 3.5 GB, versus about 8.2 GB for the [4B notebook](../colab/Spark-X2.6-4B-on-device-agent.ipynb). If a 16 GB GPU is already busy, or you want headroom for a longer KV cache, the 1.7B is the sibling that still fits in float16 without 4-bit.
- **Latency-tunable reasoning.** The card exposes thinking **none / low / high**. On the card, switching the 1.7B from high to low drops the team's average score by 4.0 points and shortens average output by 37.3 percent, so you can spend tokens only when the task needs them.

## Not a good fit for

- **Vision, audio, or OCR.** This checkpoint is text-only. Use a VL or ASR notebook in this repo for screenshots, speech, or page images.
- **Hard software-engineering work where you already have VRAM for the 4B.** The announcement's gaps are small (3.0 SWE Verified, 1.6 τ²-bench) but they favor the 4B. If float16 or 4-bit 4B fits, prefer [Spark-X2.6-4B](../colab/Spark-X2.6-4B-on-device-agent.ipynb).
- **The card's 1,048,576-token context on a free T4.** Native context is one million tokens; a 16 GB T4 cannot hold that KV cache. This notebook caps new tokens around 1k.
- **Runtimes that cannot execute Hub custom code.** Loading runs `modeling_spark.py` / `configuration_spark.py` with `trust_remote_code=True`. That is required; there is no stock Transformers class for `spark2_5` on the pinned 4.57.1 wheel.
- **A drop-in replacement for a 7B+ dedicated coder.** Team numbers are strong for the size. They are not a claim that 1.7B matches a much larger coding model on long, multi-file jobs.

## Specs

| | |
| --- | --- |
| Who | SparkLLM / [`XHToken`](https://huggingface.co/XHToken) |
| Checkpoint | [`XHToken/Spark-X2.6-1.7B`](https://huggingface.co/XHToken/Spark-X2.6-1.7B) |
| Size | 1707.7M parameters on the Hub (about 1.71B) |
| Architecture | `spark2_5` / `Spark2_5ForCausalLM` (same as Spark-X2.5-1.7B; this release is post-training) |
| Licence | Apache 2.0 |
| Languages | English, Chinese |
| Custom code | Yes (`trust_remote_code`) |
| Card pin | `transformers==4.57.1` |

The card describes gains in coding, agents, long context, multilingual use, and reasoning, with a controllable thinking budget. It does not publish a numeric SWE / τ² table in the README; those two comparisons below are from the [source post](https://x.com/SparkLLM/status/2115100695775056057).

## Why it is interesting

SparkLLM posted that the 1.7B sits close to the 4B at high thinking:

| Benchmark | Spark-X2.6-1.7B | Spark-X2.6-4B | Source |
| --- | --- | --- | --- |
| SWE Verified | 59.8 | 62.8 | [SparkLLM post](https://x.com/SparkLLM/status/2115100695775056057) |
| τ²-bench | 85.3 | 86.9 | same post, both high thinking |

On the [model card](https://huggingface.co/XHToken/Spark-X2.6-1.7B), switching from high to low thinking drops the 1.7B team's average score by 4.0 points (37.3 percent shorter outputs) versus 1.7 points (22.0 percent shorter) for the 4B. That is a larger quality/length trade on the small model, which is worth knowing if you turn thinking down to save T4 time.

These are vendor-reported evaluations. This notebook does not rerun SWE or τ².

## Expected T4 footprint

A free Colab T4 has about 15 GB of VRAM, no native bfloat16, and no FlashAttention 2. The saved dtype is bfloat16; this notebook loads **float16**.

Weights are about 1.71B × 2 bytes ≈ **3.5 GB**. After load you should see a few GB allocated, with short demos well under 15 GB. Unlike the 4B notebook, there is no bitsandbytes 4-bit fallback: float16 fits.

Sampling follows the card's Transformers snippet and `generation_config.json`: `do_sample=True`, `temperature=1.0`, `top_p=0.95`, `top_k=0`, `repetition_penalty=1.0`. The saved config has `top_k: -1`, which Transformers rejects, so the helper always passes `top_k=0`. The card's `max_new_tokens=131072` is not used.

Install is `transformers==4.57.1` and `accelerate`. The notebook tries to drop a pre-imported Colab `transformers` so **Runtime > Run all** can pick up the pin without Restart session. If the version assert still fails, restart once and continue from the GPU cell.

## Notebook walkthrough

1. **GPU check.** Prints `torch.cuda.is_available()`, the device name, and free VRAM. It imports torch only, so the later pin can take effect. On a T4 it selects float16.
2. **Install.** Pins `transformers==4.57.1`. Does not install bitsandbytes. Does not reinstall torch.
3. **Load.** `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, torch_dtype=float16, device_map="auto")`. Look for about 1.71B parameters and roughly 3.5–5 GB allocated.
4. **Think-then-answer.** Asks how big 1.7B is in float16 versus the 4B, with `enable_thinking=True` and `reasoning_effort="low"`. The card / SGLang default is **high** (the setting used for the posted SWE / τ² numbers); this T4 demo uses **low** because the card reports high→low as 37.3% shorter on the 1.7B, which leaves room to close `</think>` and still write the response channel. Sampling matches the card (`temperature=1.0`, `top_p=0.95`, `top_k=0`, `repetition_penalty=1.0`). The cell splits the think channel from the answer the way the Jinja template and SGLang's `--reasoning-parser qwen3` do: new tokens are think until `</think>`, then the answer. If the think channel never closes, the helper appends `</think>` and generates the answer rather than printing truncated reasoning as the answer. Expect a short arithmetic argument (1.71B × 2 bytes ≈ 3.4 GB, fits 16 GB; 4B is about 8.2 GB). The generate cap is 1024 new tokens, not the card's 131072.
5. **Coding task.** Asks for a `parse_kv` helper with thinking off and temperature 0.4 (a demo choice; the card sample uses 1.0). The following cell writes the source to a temp file, rejects a short list of dangerous imports, and runs four assertions. Look for `all tests passed`.
6. **Tool-use loop.** Offers `get_weather` and `get_local_time` stubs. `get_local_time` returns ISO, a 24-hour `local_24h`, and a `time_of_day` label (morning / afternoon / evening / night) so the final answer can match the tool instead of guessing from the prompt. The user prompt asks the model to state those returned fields. The first generate should contain `<tool_call>` blocks; the notebook executes the stubs and asks for a two-sentence recommendation. A text-only answer with no tool block is a miss, not a crash.
7. **Cleanup.** Deletes the model and empties the CUDA cache so you can load something else on the same T4.

## Prompting, sampling, and thinking

Use `tokenizer.apply_chat_template` with `add_generation_prompt=True`. Pass `enable_thinking` and, when thinking is on, `reasoning_effort` of `"high"` or `"low"`. The Jinja default is thinking **on** and effort **high**. This notebook's think demo uses `"low"` on a T4. Pass `enable_thinking=False` when you want the model to spend tokens on a function or a tool call instead of a trace.

When thinking is on, the template prepends `<think>` to the generation prompt. Decode the new tokens and split on `</think>`: before is the think channel, after is the response channel (same split as SGLang `--reasoning-parser qwen3`). `<think>` and `</think>` are added tokens with `special=false`, so `skip_special_tokens=True` keeps them. If `</think>` never appears, the whole generation is still think, not the answer.

The template renders tools as `<tools>…</tools>` and expects calls of the form `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`. Assistant turns can carry `reasoning_content` and `tool_calls` with **dict** arguments (not JSON strings). Tool results use role `"tool"`.

Card sampling: temperature 1.0, top_p 0.95, top_k 0, repetition_penalty 1.0, `do_sample=True`. Keep `max_new_tokens` around 1k or below on a T4.

## Limitations and gotchas

- Custom code from the Hub runs at load time. Read `modeling_spark.py` before you ship this path.
- Colab's default transformers is newer than 4.57.1. Do not bump the pin unless the card changes.
- High thinking on this 1.7B can fill a small token cap before `</think>`, which makes a naive split print “(no think channel)” and treat truncated reasoning as the answer. The think demo uses `reasoning_effort="low"` and 1024 new tokens; if the channel still does not close, the helper appends `</think>` and generates the response. Raise `max_new_tokens` further only if free VRAM still looks comfortable.
- The modeling file implements eager attention. FlashAttention 2 is not on this path.
- Optional GGUF lives at [`XHToken/Spark-X2.6-1.7B-GGUF`](https://huggingface.co/XHToken/Spark-X2.6-1.7B-GGUF) (`Q4_K_M` about 1.8 GB). llama.cpp needs `b10828` or later for `spark2_5`. Unload the Transformers model before you download a GGUF on a 16 GB card.
- `HF_TOKEN` is optional. The weights are public.

## Compared with the 4B sibling

| | 1.7B (this notebook) | 4B |
| --- | --- | --- |
| Checkpoint | [`XHToken/Spark-X2.6-1.7B`](https://huggingface.co/XHToken/Spark-X2.6-1.7B) | [`XHToken/Spark-X2.6-4B`](https://huggingface.co/XHToken/Spark-X2.6-4B) |
| Parameters | ~1.71B | ~4.11B |
| Float16 weights | ~3.5 GB | ~8.2 GB |
| T4 load | float16, no 4-bit | float16 first, 4-bit NF4 if it overflows |
| SWE Verified (high thinking) | 59.8 | 62.8 |
| τ²-bench (high thinking) | 85.3 | 86.9 |
| High → low (card) | −4.0 avg score, −37.3% length | −1.7 avg score, −22.0% length |

Demos are the same shape: think-then-answer, `parse_kv`, weather/time tools. Choose 1.7B when you want float16 headroom; choose 4B when you can spend the extra VRAM for the posted score gaps.

## Links

- [Model card](https://huggingface.co/XHToken/Spark-X2.6-1.7B)
- [Source post](https://x.com/SparkLLM/status/2115100695775056057)
- [Open in Colab](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Spark-X2.6-1.7B-on-device-agent.ipynb)
- [This notebook](../colab/Spark-X2.6-1.7B-on-device-agent.ipynb)
- [Spark-X2.6-4B notebook](../colab/Spark-X2.6-4B-on-device-agent.ipynb)
- [GGUF](https://huggingface.co/XHToken/Spark-X2.6-1.7B-GGUF)
- Nearby chat/tool notebooks: [LFM3-1.7B](../colab/LFM3-1.7B-chat-tools.ipynb), [Granite 4.2 3B](../colab/granite-4.2-3b-chat-tools.ipynb)
