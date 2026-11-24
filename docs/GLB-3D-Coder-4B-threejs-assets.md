# GLB-3D-Coder 4B Three.js assets

[`ccore/Qwen3.5-4B-GLB-3D-Coder`](https://huggingface.co/ccore/Qwen3.5-4B-GLB-3D-Coder) is an Apache-2.0 4B coding finetune of [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B). Describe a low-poly character, prop, building, or environment in English, and it writes one complete standalone Three.js HTML page with a preview and an **Export GLB** button. Open the page, inspect the asset, and import the `.glb` into Godot or another editor that supports GLB. The Hub ships Q4_K_M GGUF (2.7 GB), Q8_0 GGUF (4.5 GB — the quant used for the recorded evaluation), and the original safetensors (five shards, about 17.49 GB, FP32 text + BF16 vision).

## Use cases

- **Turn a short English brief into a previewable low-poly game asset.** The card's job: a complete HTML file that shows the asset and downloads a working GLB. Start with a short idea, or name parts, materials, colors, layout, and animation.
- **Props and small architecture for a Godot (or other GLB) pipeline.** The card's domains include items, furniture, modular architecture, and homestead scenes. This notebook demos a treasure chest and a small house.
- **A T4-sized smoke of the GGUF path** with the card system prompt, sampler (temperature 0.8, top_k 40, top_p 0.95, min_p 0.0, repeat penalty 1.0), and thinking left on.

## Not a good fit for

- **General-purpose coding or chat.** The model is specialized for Three.js / GLB game assets, not a SWE agent.
- **Photoreal meshes or the 17.49 GB original checkpoint on a free T4.** Native FP32 text + BF16 vision safetensors do not fit ~15 GB of VRAM. Use Q4_K_M (default) or Q8_0.
- **An 8K output cap, or thinking turned off.** The card: responses average around 20K tokens and some are longer; 8K can stop during thinking or before the HTML is complete. Thinking should stay enabled with no separate thinking-token budget; the author has not validated quality with thinking disabled.
- **Compiling llama.cpp from source in Colab.** Source builds disconnect the free T4 runtime. Use the pinned prebuilt CUDA wheel.
- **Guaranteed-correct GLB on the first try.** The card: some requests need a retry or a small code correction; materials, lighting, and animations can differ from the browser preview.

## Running it on a free Colab T4

Pick a T4 GPU. Install **`llama-cpp-python` 0.3.37** from the CUDA 12.1 wheel URL (plus `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12`). Do not compile. `qwen35` is in llama.cpp from **b7990**; 0.3.17 is the first Python release with Qwen 3.5 support. Load **Q4_K_M (2.7 GB)** with `n_gpu_layers=-1`, `n_ctx=32768`, `flash_attn=False` (T4 has no FlashAttention-2). The native 262,144-token window from `config.json` is not allocated. System prompt and sampler match the card. Peak VRAM is printed from `nvidia-smi` because llama.cpp does not go through `torch.cuda.memory_allocated`. A complete page can take several minutes.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Pinned CUDA `llama-cpp-python` 0.3.37 wheel (no source compile).
- Q4_K_M GGUF load; print `general.architecture` and require `qwen35`.
- Card system prompt + sampler, thinking on, `max_tokens=-1`.
- Demo 1: low-poly treasure chest → `treasure-chest.html`, inline iframe preview.
- Demo 2: small house → `small-house.html`, inline iframe preview.
- `nvidia-smi` used/free/total after load, after generate, and peak VRAM at cleanup.

## Links

- Model card: [ccore/Qwen3.5-4B-GLB-3D-Coder](https://huggingface.co/ccore/Qwen3.5-4B-GLB-3D-Coder)
- Training dataset: [ccore/ThreeJS-3D-GLB-Assets](https://huggingface.co/datasets/ccore/ThreeJS-3D-GLB-Assets) (3,918 tested and rendered examples across 40 domains)
- Source post: [x.com/alexdelrosso](https://x.com/alexdelrosso/status/2123780570526142700)
- Base model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/GLB-3D-Coder-4B-threejs-assets.ipynb`](../colab/GLB-3D-Coder-4B-threejs-assets.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/GLB-3D-Coder-4B-threejs-assets.ipynb)
- Related: [FrogNano-4B coding agent](FrogNano-4B-coding-agent.md), [Mellum2.1 12B-A2.5B Thinking coding agent](Mellum2.1-12B-A2.5B-Thinking-coding-agent.md), [Shot Reader 8B video shot to 3D](shot-reader-8b-video-shot-to-3d-scene.md)
