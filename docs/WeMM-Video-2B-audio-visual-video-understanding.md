# WeMM-Video-2B audio-visual video understanding

[`tencent/WeMM-Video-2B`](https://huggingface.co/tencent/WeMM-Video-2B) is Tencent’s dense 2B checkpoint in the WeMM-Video family ([Apache-2.0](https://huggingface.co/tencent/WeMM-Video-2B/blob/main/LICENSE)). It is a Chinese–English bilingual omni-modal model: text, images, audio, and video in, text out. For video it can use the original soundtrack with the frames (`video_audio`). Hub weight files are BF16 safetensors, **6.12 GiB**. Load is `AutoProcessor` + `AutoModelForImageTextToText` with `trust_remote_code`. The Hub lists ~3.28B parameters once the vision and audio encoders are counted. Quantized siblings (FP8, AWQ, GGUF, NVFP4) exist; this notebook uses the BF16 card loaded as float16.

## Use cases

- **Short clips where the soundtrack matters.** The card: combine video frames with the original soundtrack to describe events, transcribe speech, and reason about what is seen and heard (`type: "video_audio"`; GitHub `examples/transformers_video.py`).
- **One 2B checkpoint instead of a VLM + ASR stack** when you only need text out, in English or Chinese. Silent video, standalone audio, images, and text are also accepted.
- **A T4-sized smoke of the Transformers path** before you take the [patched vLLM / SGLang](https://github.com/Tencent/WeMM-Video/blob/main/docs/serving.md) serving recipes.

## Not a good fit for

- **Long or high-resolution video on a free T4.** Processor defaults are `max_frames=128` and `total_pixels=25165824`. The card example uses `fps=1.0`, `min_frames=4`, `max_frames=16`, `total_pixels=262144`. This notebook uses **8 frames** so 15 GB VRAM is enough.
- **Speech output.** The family generates text responses, not TTS.
- **The card’s `dtype=torch.bfloat16` on a T4.** A T4 (`sm_75`) has no native bfloat16. This notebook loads **float16**.
- **Unpatched vLLM / SGLang.** GitHub serving needs the [matching WeMM runtime patch](https://github.com/Tencent/WeMM-Video/blob/main/patch/README.md#manual-install-and-patch). This notebook is Transformers only.
- **Treating the [WeMM-Video-35B-A3B evaluation table](https://huggingface.co/tencent/WeMM-Video-35BA3B#evaluation) as a 2B result.** The 2B card points at that table; those numbers are not measured here.
- **Open-vocab boxes / alpha cutouts.** Use [WeVisOne-VL 2B](WeVisOne-VL-2B-detect-segment.md).

## Running it on a free Colab T4

Pick a T4 GPU. GitHub install pins `torch==2.10.0`, `torchcodec==0.10.0`, and `transformers==5.2.0` (`config.json` records `transformers_version: 5.2.0`). **Do not reinstall Colab’s CUDA torch.** The notebook keeps that wheel, installs `transformers>=5.2.0` (current Colab 5.x already satisfies it), `accelerate`, `pillow`, `librosa`, `soundfile`, `scipy`, `requests`, `av`, and `jinja2>=3.1.6`, then restarts the session. `trust_remote_code` is required. Custom `wemm_media.py` decodes with **torchcodec**; if that fails, the notebook patches in PyAV (video) and soundfile/librosa (audio). First run downloads **6.12 GiB**. Video is capped at 8 frames / 1 fps / 262144 total pixels (even counts; `temporal_patch_size=2`). After load, leftover **bfloat16** weights are recast to float16: `h5_modeling_whisper.py` hardcodes `torch_dtype=torch.bfloat16` on the audio Q-Former and `audio_proj`, which otherwise hits `F.layer_norm` with `expected scalar type Half but found BFloat16`. Processor floating tensors are cast to float16 before generate. Peak VRAM is printed at the end of the notebook.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card Transformers load in **float16** on T4 (`AutoProcessor` + `AutoModelForImageTextToText`, `device_map="auto"`, SDPA then eager). Leftover bf16 Q-Former / `audio_proj` weights are recast to float16; processor floats are cast to the same dtype before generate.
- `video_audio`: official Apollo 11 television clip from the GitHub examples (NASA archives), with a synthesized moving-square + speech/tone fallback.
- Image question on the official Apollo 11 Earth still (NASA / COCO fallbacks).
- Audio transcription plus a one-sentence summary on the official “small step” WAV (Hugging Face dummy speech / espeak fallbacks).
- Session `torch.cuda.max_memory_allocated` after cleanup.

## Links

- Model card: [tencent/WeMM-Video-2B](https://huggingface.co/tencent/WeMM-Video-2B)
- GitHub: [Tencent/WeMM-Video](https://github.com/Tencent/WeMM-Video) (`examples/transformers_video.py`)
- Technical report: [arXiv:2611.26409](https://arxiv.org/abs/2611.26409)
- Source post: [x.com/HuggingPapers](https://x.com/HuggingPapers/status/2123750396002767170)
- Collection: [tencent/wemm-video](https://huggingface.co/collections/tencent/wemm-video)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/WeMM-Video-2B-audio-visual-video-understanding.ipynb`](../colab/WeMM-Video-2B-audio-visual-video-understanding.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/WeMM-Video-2B-audio-visual-video-understanding.ipynb)
- Related: [LFM3-Omni-2B](LFM3-Omni-2B.md), [LFM3-VL-2B](LFM3-VL-2B-vision.md), [LFM3-Audio-2B](LFM3-Audio-2B-voice.md), [WeVisOne-VL 2B](WeVisOne-VL-2B-detect-segment.md), [MiniCPM-V 4.7 1B](MiniCPM-V-4.7-1B-vision.md)
