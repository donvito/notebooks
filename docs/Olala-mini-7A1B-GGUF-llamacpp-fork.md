# Olala mini 7A1B GGUF (llama.cpp fork)

[`OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF) is OVHai LLM’s (OVHcloud) Apache-2.0 GGUF of [`Olala-v1.0-mini-7A1B`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B): a text-only hybrid Mamba-3 / attention MoE (`olala` architecture, 6.8B total, about 1B active per token, 32K context). Upstream llama.cpp, Ollama, LM Studio, and `llama-cpp-python` cannot load it. This notebook builds the [OVHai-LLM llama.cpp fork](https://github.com/OVHai-LLM/llama.cpp) with CUDA for a free Colab T4 and runs the published `q8_0-safe2` file (6.9 GB).

## Use cases

- **A laptop-sized Apache-2.0 chat model that is not a dense 7B.** The [base card](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B) is a 36-layer hybrid: 29 Mamba-3 layers and 7 attention layers (one in five), each followed by a mixture-of-experts block with 256 routed experts (6 active) and 1 shared expert. About 1B parameters run per token, so the interesting case is local chat and instruction following when you want the Olala stack rather than a dense 7–8B.
- **Tool calling with JSON arguments, as the cards show.** The GGUF and the fork’s [setup guide](https://github.com/OVHai-LLM/llama.cpp/blob/main/docs/olala-setup.md) use an OpenAI-style `tools` list. The model writes calls as XML with JSON arguments in a `tools` channel (`get_weather` / `{ "city": "Paris" }` on the cards). The notebook runs that stub, feeds the tool result back, and prints `tool_calls` from `llama-server`.
- **Longer text in one shot, within the 32K window.** The cards cap context at 32,768 tokens (input + output). Recurrent layers cannot shift a prompt that overruns `-c`. The T4 demo summarises a few-thousand-token brief at `-c 8192` so the window is visible without allocating the full 32K KV/state on 15 GB.
- **When stock llama.cpp is not an option.** The GGUF card is explicit: you must build [OVHai-LLM/llama.cpp](https://github.com/OVHai-LLM/llama.cpp). The Transformers path wants `transformers==5.8.*` plus their Mamba fork and scattermoe. This notebook stays on the published GGUF + fork path.

## Not a good fit for

- **Upstream llama.cpp, Ollama, LM Studio, or a PyPI `llama-cpp-python` wheel.** Those do not support the `olala` architecture. The GGUF will not load.
- **The 13.6 GB BF16 GGUF on a free T4.** That file does not fit 15 GB with context. The notebook downloads `Olala-v1.0-mini-7A1B-q8_0-safe2.gguf` (6.9 GB), with `q6_k` (6.3 GB) then `q5_k_m` (6.0 GB) if the T4 OOMs or hits a CUDA kernel issue.
- **Vision, audio, or other non-text input.** The base card is a text-only language model.
- **Answering from memory where accuracy matters** (facts, figures, legal, medical, or financial information). The base card says to give the model the source, or a tool to read it.
- **High-stakes automated decisions.** The base card puts credit, employment, insurance, education, health, justice, and similar uses out of scope without review by a qualified person.
- **Competition mathematics as the main job.** Of the scores on the base card, that is the model’s weakest area.

## Running it on a free Colab T4

Pick a T4 GPU (*Runtime > Change runtime type > T4 GPU*). The first code cell prints `torch.cuda.is_available()`, the GPU name, and free VRAM and stops if you are on CPU — do not start the fork compile on a CPU runtime.

There is no pre-built Colab binary for this architecture. The notebook clones [`OVHai-LLM/llama.cpp`](https://github.com/OVHai-LLM/llama.cpp) and follows the GGUF card’s CUDA build, with two T4-only additions: `-DCMAKE_CUDA_ARCHITECTURES=75` (sm_75 only, so nvcc is not compiling every GPU) and `cmake --build … -j$(nproc) --target llama-cli llama-server`. It also passes `-DLLAMA_USE_PREBUILT_UI=OFF` as the card requires. **The cell prints the wall-clock build time.** Expect several minutes on Colab: this is a CUDA compile, not a wheel install. The same cmake flags without CUDA configure and produce `llama-cli` / `llama-server` on a CPU-only VM.

Default weights are **`Olala-v1.0-mini-7A1B-q8_0-safe2.gguf` (6.9 GB)** from the GGUF card’s files table, plus the card’s `Olala-v1.0-mini-7A1B-chat-template.jinja`. All layers go on the GPU (`-ngl 99`). Context for the demos is **8,192** tokens (the card maximum is 32,768). Sampling matches the cards: `temperature=0.8`, `top_p=0.9`, `top_k=40`. Reasoning effort is **`low`**, the same flag as the GGUF card’s NVIDIA `llama-cli` example (the template default is `medium`). Flash attention is **on** (`-fa 1`), as the GGUF card states.

The fork’s older setup notes warn that quantized files on CUDA can NaN because of fp16 activation scales, and they recommend BF16 on NVIDIA. The **published GGUF card** says all four files run on NVIDIA GPUs. If this T4 (sm_75) hits a kernel error, NaNs, or OOM, the notebook documents the failure and falls back: `-fa 0`, then `q6_k` / `q5_k_m`, then fewer GPU layers. Do not set `OLALA_BF16_STATE` (the cards keep the Mamba state in f32).

This walkthrough is inference only. It does not train or fine-tune.

## What the notebook demos

- Check CUDA, GPU name, and free VRAM.
- Build the OVHai-LLM llama.cpp fork for sm_75 and print the compile time.
- Download `q8_0-safe2` (6.9 GB) and the GGUF chat-template file.
- **Demo 1:** a short `llama-cli` chat and the binary’s tokens/s lines.
- **Demo 2:** `llama-server` with the cards’ `get_weather` JSON tool call, then a mocked tool result.
- **Demo 3:** summarise a few-thousand-token prompt.
- Print `nvidia-smi` used memory as peak VRAM (llama.cpp allocations do not show up in `torch.cuda`).

## Links

- GGUF: [OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF)
- Base card: [OVHaiLLM/Olala-v1.0-mini-7A1B](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B)
- llama.cpp fork: [OVHai-LLM/llama.cpp](https://github.com/OVHai-LLM/llama.cpp)
- Source post: [x.com/OVHcloud](https://x.com/OVHcloud/status/2123370331406483592)
- Notebook: [`colab/Olala-mini-7A1B-GGUF-llamacpp-fork.ipynb`](../colab/Olala-mini-7A1B-GGUF-llamacpp-fork.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/melvin/olala-mini-7a1b-gguf-llamacpp-62dd/colab/Olala-mini-7A1B-GGUF-llamacpp-fork.ipynb)
- Related: [LFM3-1.7B chat and tools](LFM3-1.7B-chat-tools.md), [Granite 4.2 3B chat and tools](granite-4.2-3b-chat-tools.md), [Ternary Bonsai 2 4B](Ternary-Bonsai-2-4B-llamacpp.md), [LFM2.5-2.6B llama-cpp-python](LFM-2_5-2_6b-llamacpp-python.md)
