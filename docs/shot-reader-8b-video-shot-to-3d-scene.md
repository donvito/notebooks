# Shot Reader 8B — video shot to a JSON 3D scene

[`voyager-so/shot-reader-8b`](https://huggingface.co/voyager-so/shot-reader-8b) is Voyager’s Shot Reader 8B ([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)). It is a fine-tune of [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (`Qwen3VLForConditionalGeneration`, Hub ~8.8B). Hub weights are a single BF16 safetensors file, **17.5 GB**. It reads one continuous video shot and writes a JSON shot sheet: camera height, angle, lens and move, who is in frame, where they stand, how they move, and the set around them, so a 3D tool or a coding agent can rebuild the scene in Blender.

## Use cases

From the [model card](https://huggingface.co/voyager-so/shot-reader-8b):

- **Previs from reference.** Drop in a clip and get an editable 3D blockout to relight, re-lens and reshoot, instead of rebuilding shots by hand.
- **Controllable AI video.** Camera, layout and motion become explicit JSON you can edit, not something a text prompt has to capture.
- **Searchable footage.** Tag libraries with camera moves, framing and blocking (the card quotes about $2 per thousand shots on their stack).
- **Agents that build scenes.** A scene-building agent gets a precise brief instead of a vague description. The card’s motivation is [VibePrevisBench](https://arxiv.org/abs/2611.15789): agents asked for a low angle still deliver eye level in up to two thirds of shots, and an orbit stays an orbit 4% of the time; handing the agent a plain description first jumps the score 15.6 points.

Name the subjects you care about (`P01: person (…)`). Shot Reader keeps those ids so the sheet lines up with your brief.

## Not a good fit for

- **The card’s BF16 load on a free T4.** 17.5 GB of BF16 will not fit ~15 GB. `shot_reader.py --load-in-4bit` is the small-GPU path (~6 GB of VRAM on the card, tested on a 16 GB RTX 5080). This notebook uses that 4-bit NF4 path with **float16 compute** because a T4 (`sm_75`) has no native bfloat16 (`shot_reader.py` still passes `bnb_4bit_compute_dtype=torch.bfloat16`).
- **The card’s 16 frames / 8192 new tokens on 15 GB.** The published recipe samples 16 frames at 448×256. This notebook uses **8 frames** and **2048** new tokens so a free T4 stays inside 15 GB.
- **Long cuts / several shots in one file.** The card: one input is one continuous shot; cut longer videos at the edits first.
- **Events and routes as the thing you trust.** The card lists those as its weakest fields on real footage; Opus 5.5 still writes richer action events. Positions are estimates from a single camera, typically within a few metres.
- **vLLM / Voyager Cloud H100 FP8 numbers as a T4 result.** Time per shot 5.9 s and ~$0.002 on the card are for one H100 with vLLM in FP8, not this notebook.
- **Audio, speech, or boxes/cutouts.** This model writes a camera/blocking JSON sheet, not a caption, transcript, or mask. See [WeMM-Video-2B](WeMM-Video-2B-audio-visual-video-understanding.md) or [WeVisOne-VL 2B](WeVisOne-VL-2B-detect-segment.md).

## Running it on a free Colab T4

Pick a T4 GPU. Keep Colab’s CUDA torch. Pin **`transformers==5.23.0`** (`config.json` `transformers_version`; card floor is `transformers>=4.57`). Do not keep a newer Colab 5.24.x if it is already imported — the install cell restarts the session. Install `bitsandbytes`, `accelerate`, `torchvision`, `pillow`, `av`, `huggingface_hub`. No `flash-attn`. No secrets; weights are public.

Load `AutoProcessor` + `AutoModelForImageTextToText` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`. First run downloads **17.5 GB**. Frames are sampled the way `shot_reader.py` does (evenly across the shot, resized to 448×256, passed as images with the card `PROMPT` plus Hub `schema.md`). Clips are capped at 5 s. Some Colab T4 images need cuDNN disabled around `generate` (Qwen3-VL `conv3d`). Peak VRAM is printed at the end; expected **~6–8 GB**.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card Transformers load in **4-bit NF4 / float16 compute** on T4 (SDPA then eager).
- The exact `shot_reader.py` prompt plus `schema.md`.
- Demo 1: a short public pedestrian clip with named `--subject` ids; pretty-printed JSON shot sheet.
- Demo 2: a short public tennis clip with no ids (the prompt’s “none given” line).
- Public URL fallbacks, then a synthesized interior if downloads fail.
- Session `torch.cuda.max_memory_allocated` after cleanup.

## Links

- Model card: [voyager-so/shot-reader-8b](https://huggingface.co/voyager-so/shot-reader-8b)
- Base model: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Script / schema: [`shot_reader.py`](https://huggingface.co/voyager-so/shot-reader-8b/blob/main/shot_reader.py) · [`schema.md`](https://huggingface.co/voyager-so/shot-reader-8b/blob/main/schema.md)
- VibePrevisBench: [arXiv:2611.15789](https://arxiv.org/abs/2611.15789)
- Source post: [x.com/j_h_h_3](https://x.com/j_h_h_3/status/2123532630645977367)
- Voyager: [voyager.so](https://voyager.so)
- Community GGUF (not this notebook): [mradermacher/shot-reader-8b-GGUF](https://huggingface.co/mradermacher/shot-reader-8b-GGUF)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/shot-reader-8b-video-shot-to-3d-scene.ipynb`](../colab/shot-reader-8b-video-shot-to-3d-scene.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/shot-reader-8b-video-shot-to-3d-scene.ipynb)
- Related: [WeMM-Video-2B](WeMM-Video-2B-audio-visual-video-understanding.md), [LFM3-VL-2B](LFM3-VL-2B-vision.md), [MiniCPM-V 4.7 1B](MiniCPM-V-4.7-1B-vision.md), [LFM3-Omni-2B](LFM3-Omni-2B.md)
