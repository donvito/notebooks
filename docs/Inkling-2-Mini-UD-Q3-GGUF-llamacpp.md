# Inkling 2 Mini UD-Q3 GGUF (llama.cpp)

[`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) is Unsloth’s UD imatrix GGUF of Thinking Machines Lab [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): Apache-2.0, released 20 November 2026, **image + text → text**. It is a **looped MoE** (26.2B total, 3.4B active, 128 experts, top-8; the middle 12 of 24 layers run three times). This notebook loads the **root** `UD-Q3_K_XL` file (13.63 GB, mean KLD 0.0174 vs UD-Q8_K_XL) plus `mmproj-F16.gguf` (0.64 GB) through Unsloth’s **prebuilt** CUDA 12 `llama-server`. It does **not** compile llama.cpp.

## Use cases

- **Free-Colab T4 look at Inkling 2 Mini** when you want the 3-bit Unsloth GGUF rather than a 24 GB card’s 4-bit/8-bit path.
- **Executed coding checks** (merge intervals, a bug fix, a small SQL query) so you can see whether the model’s code actually runs, not only whether it looks plausible.
- **Short image questions** through the OpenAI-compatible server (`--mmproj` + a `data:` URL).
- **MoE expert offload** (`-ngl 99` + `--n-cpu-moe`) when the GGUF is larger than T4 VRAM.

## Not a good fit for

- **Compiling llama.cpp on Colab.** CUDA source builds on a free T4 repeatedly disconnect the runtime (see [PR #82](https://github.com/donvito/notebooks/pull/82)). This notebook only unpacks Unsloth’s Linux CUDA 12 tarball.
- **`Mainline_Support/*.gguf_file`.** Those copies need mainline llama.cpp plus unmerged [PR #32704](https://github.com/ggml-org/llama.cpp/pull/32704). The root files use Unsloth’s looped layout and match Unsloth’s prebuilts.
- **Full GPU offload of UD-Q3_K_XL on 15 GB.** 13.63 GB weights + 0.64 GB projector + KV cache will not fit. Keep most experts on the GPU with a small `--n-cpu-moe`, or use the **UD-IQ3_XXS** (11.66 GB, nearly fully GPU) option.
- **The card’s 1M / 32K context.** The notebook caps context at **4,096** tokens so KV plus GPU-resident experts fit and demos finish in reasonable time.
- **A published tokens/sec number.** Expert layers that stay on Colab’s ~2 CPU cores are slow, and too many of them also thrash ~12.7 GB of system RAM. The notebook’s demo table and the **⏱ Benchmark** section (`llama-bench` pp512/tg128 plus three server `timings` prompts) print a measurement from the session you run, not a leaderboard figure. The benchmark is capped so a T4 run stays around five minutes.

## Running it on a free Colab T4

Pick a T4 GPU. Default weights are **UD-Q3_K_XL** (13.63 GB) + **mmproj-F16** (0.64 GB). The server is Unsloth `llama-server` (`-ngl 99`, auto-picked `--n-cpu-moe`, `-c 4096`, `-fa auto`, `--jinja`, mmap, no `--mlock`).

**`--n-cpu-moe` (the T4 fit knob):** last N of 24 MoE layers keep expert weights in system RAM (mmap). Colab RAM is only ~12.7 GB, so **most experts should sit in VRAM**. N that is too large (for example 16) leaves several GB of T4 VRAM free and is slow because CPU experts page-cache-thrash. The download cell reads GGUF expert tensor sizes, estimates per-layer expert VRAM, and **auto-picks the smallest N** that leaves ~**1.5 GB** VRAM headroom (target ~12.5–13 GB used on a 15 GB T4). If the GGUF cannot be read, it falls back to **4**. The chosen value is printed. If `llama-server` fails to start or OOMs, the server cell retries with **`N_CPU_MOE+2`**. Set `N_CPU_MOE_OVERRIDE` to an int to force a value.

**If Q3_K_XL still cannot work on T4:** set `QUANT = "UD-IQ3_XXS"` in the download cell (11.66 GB, nearly fully GPU) and rerun from there. The notebook prints that option when load still fails.

Chat completions cap at **384** tokens; coding tests cap at **768** with a per-request timeout. Prompts ask the model to answer concisely. Coding cells catch timeouts so one slow test does not kill the run.

The binary is resolved at runtime from the GitHub API (`unslothai/llama.cpp/releases/latest`) with pinned fallback tag **`b12588-mix-c0c3de8`**. `llama-prebuilt-manifest.json` picks the Linux x64 CUDA 12 asset that includes **sm_75** (portable if it lists 75, otherwise `cuda12-older`).

Sampler: temperature 1.0, top-p 1.0, top-k 0, min-p 0. Public Hub repo — cancel any `HF_TOKEN` prompt.

## What the notebook demos

- GPU / VRAM check.
- Unsloth CUDA 12 prebuilt `llama-server` (no compile).
- `hf download` of only the GGUF + projector, then auto-pick `--n-cpu-moe`.
- Text chat / reasoning question.
- Three coding tests with **executed** asserts: merge overlapping intervals, fix `longest_run`, SQL department averages via `sqlite3` (timeout + try/except per test).
- Image question on a public sample (with fallbacks).
- Tokens/sec and peak `nvidia-smi` VRAM.
- **⏱ Benchmark** (after the demos, before kill): three server prompts (short, ~1k-token prompt, 512-token generation) reporting prompt tok/s and generation tok/s from `timings`; then stop `llama-server` and run prebuilt `llama-bench -p 512 -n 128 -r 2` with the **same chosen** `-ngl` / `--n-cpu-moe` / `-fa` if the binary is in the tarball (skip with a message if it is not — no compile).
- Kill the server.

## Links

- GGUF: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
- Base: [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- Run guide: [unsloth.ai/docs/models/inkling-2-mini](https://unsloth.ai/docs/models/inkling-2-mini)
- Prebuilts: [unslothai/llama.cpp releases](https://github.com/unslothai/llama.cpp/releases)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/Inkling-2-Mini-UD-Q3-GGUF-llamacpp.ipynb`](../colab/Inkling-2-Mini-UD-Q3-GGUF-llamacpp.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/melvin/inkling-2-mini-ud-q3-gguf-cea2/colab/Inkling-2-Mini-UD-Q3-GGUF-llamacpp.ipynb)
