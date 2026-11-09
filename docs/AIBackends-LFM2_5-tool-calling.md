# AIBackends + LFM2.5-2.6B tool calling

Native tool calling with Liquid AI's [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) (~2.6B, LFM Open License v1.0) on the aibackends llama.cpp runtime. Tools are JSON in the system prompt; the model replies with a Pythonic call such as `[get_weather(city="Paris")]` between tool-call tokens. aibackends parses those with `extract_tool_calls` / `clean_answer`. Mirrors `examples/tasks/tool_calling_lfm.py`.

## Use cases

- **Small on-device agents.** LFM2.5 is positioned for on-device / edge deployment; the notebook's loop (model picks a tool → you execute → `tool` role → answer) is the local-agent shape.
- **Offline stubs you can swap for real APIs.** Weather/time-style tools run without network, then you replace the stubs.
- **Route-then-act.** Pair with [prompt routing](AIBackends-LFM2_5-prompt-routing.md): the 350M encoder chooses a lane, then 2.6B calls tools.

## Not a good fit for

- **Long autonomous coding sessions.** For mini SWE-style agents see FrogNano / Mellum notebooks (later batches).
- **Vision or audio tools.** This notebook is text tool calling on llama.cpp.
- **Apache/MIT-only deployments.** License is LFM Open License v1.0 (`license: other`).

## Running it on a free Colab T4

Works on a **free CPU runtime**; a T4 is faster and is detected automatically. `LFM25_2_6B` resolves to the **Q4_K_M** GGUF from `LiquidAI/LFM2.5-2.6B-GGUF` (~1.6 GB). Optional `quantization="Q8_0"` for higher fidelity. Set `skip_special_tokens=False` so tool-call markers stay in the output.

## What the notebook demos

- Define stub tools.
- Load LFM2.5-2.6B Q4_K_M.
- The tool loop: call → execute → feed result → final answer.

## Links

- Model card: [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B)
- GGUF: [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)
- Notebook: [`colab/AIBackends-LFM2_5-tool-calling.ipynb`](../colab/AIBackends-LFM2_5-tool-calling.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-LFM2_5-tool-calling.ipynb)
- Related: [LFM2.5 prompt routing](AIBackends-LFM2_5-prompt-routing.md), [`LFM-2_5-2_6b-llamacpp-python.ipynb`](../colab/LFM-2_5-2_6b-llamacpp-python.ipynb)
