# Rumble-2-Flash noisy speech-to-text

[`Reson8/Rumble-2-Flash`](https://huggingface.co/Reson8/Rumble-2-Flash) is Reson8’s Apache-2.0 Qwen3-ASR speech recognizer (~0.51B parameters, 508.6M in the Hub safetensors index; `model.safetensors` is 1.0 GB). The card documents native Hugging Face Transformers and vLLM, lists **45 languages**, and is public (not gated). Architecture: `Qwen3ASRForConditionalGeneration` (`model_type: qwen3_asr`). The repo also ships an optional CTC head (`ctc/ctc_head.safetensors`, `ctc_head.py`); this notebook uses the card’s Transformers decoder path, not CTC.

## Use cases

- **Small multilingual ASR** in the 45 languages on the card when you want an Apache-2.0 checkpoint that loads with the card’s `AutoProcessor` + `AutoModelForMultimodalLM` call (`processor.apply_transcription_request`, local path / URL / NumPy array).
- **Forced language vs auto-detect.** The card: pass `language="English"` (English name) to force a language; omit it to detect (`output starts with language <Name><asr_text>`; use `return_format="parsed"` when detecting).
- **Transcript formatting knobs** from the card: `digits` (default true: numbers as digits), `filler` (default false: drop “um” / “uh”), `verbatim` (default true: keep repeated words). Those go through vLLM `chat_template_kwargs` or `processor.apply_chat_template`.
- **Reading the card’s English Open ASR Leaderboard table** (release 18-11-2026, v1.16.1, H200 + vLLM): average WER **6.60** (RTFx 6338). Splits on the card include AMI **9.24**, Earnings22 **6.45**, GigaSpeech **7.56**, SPGISpeech **3.08**, URGENT24 **10.25**, Voice Arena Monsoon **3.56**, VoxPopuli **2.43**, Private (scripted) **3.71**, Private (conversational) **13.14**. Those are the card’s numbers, not a T4 measurement.

## Not a good fit for

- **Treating the card’s RTFx 6338 as a free Colab T4 figure.** That number is from the card’s H200 + vLLM leaderboard run.
- **vLLM serving.** The card’s `vllm serve Reson8/Rumble-2-Flash --max-model-len 4096` path is not this notebook.
- **The optional CTC head as the main transcript.** `ctc_head.py` in the repo: greedy CTC from the audio encoder alone is fast and **much less accurate** than the model’s normal decoder output. Extra download: `ctc/ctc_head.safetensors` (~357.5 MB).
- **bfloat16 on a free T4.** Saved weights are BF16; a T4 has no native bfloat16. This notebook loads **float16**.

## Running it on a free Colab T4

Pick a T4 GPU. The card install is `uv pip install torch transformers accelerate`. `config.json` records `transformers_version: 5.21.0.dev0`; Qwen3-ASR is native in Transformers from 5.13. The notebook keeps Colab’s CUDA torch, installs **`transformers>=5.21.0`**, `accelerate`, and **`jinja2>=3.1.6`** (`apply_chat_template`), and restarts the session so that transformers version is imported. Colab already ships Transformers 5.x, so this does **not** pin 4.x (unlike the sibling [cbx-healthcare](cbx-healthcare-speech-to-text-0.6B-medical-asr.md) notebook, which needed `qwen-asr==0.0.6` / `transformers==4.57.6`). First run downloads **1.0 GB** (`model.safetensors`). Peak VRAM is printed at the end of the notebook.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card Transformers load in **float16** on T4 (`AutoProcessor` + `AutoModelForMultimodalLM`, `device_map="auto"`).
- Clean English public clip (`librispeech_mr_quilter.wav` from [`hf-internal-testing/dummy-audio-samples`](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples)), with a download fallback.
- The same clip with added noise and an 8 kHz resample (synthetic phone-quality; not a card benchmark).
- A non-English clip with language omitted (auto-detect; `return_format="parsed"`).
- A longer public clip (chunked at the feature extractor’s 30 s `chunk_length` if needed).
- Wall time and `torch.cuda.max_memory_allocated` peak.

## Links

- Model card: [Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash)
- Source post: [x.com/raoulritterr](https://x.com/raoulritterr/status/2123053360773271563)
- Open ASR Leaderboard (card): [hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard?version=1.16.1)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/Rumble-2-Flash-noisy-speech-to-text.ipynb`](../colab/Rumble-2-Flash-noisy-speech-to-text.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Rumble-2-Flash-noisy-speech-to-text.ipynb)
- Related: [cbx-healthcare (Qwen3-ASR LoRA)](cbx-healthcare-speech-to-text-0.6B-medical-asr.md), [Fluid Speech Pro](Fluid-Speech-Pro-multilingual-asr.md), [Nemotron-Labs Unified ASR](Nemotron-Labs-Unified-ASR-0.6B-multilingual.md), [Granite Speech](granite-speech-5.0-2b-asr-translation.md), [oruk-loop-153M](oruk-loop-153M-looped-asr.md), [Whisper](AIBackends-Whisper-audio-transcription.md)
