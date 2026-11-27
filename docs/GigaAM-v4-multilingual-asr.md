# GigaAM-v4 + Multilingual-v2 ASR

[`ai-sage/GigaAM-v4`](https://huggingface.co/ai-sage/GigaAM-v4) is Sber’s 238M Zapformer RNN-T (MIT) for English and Russian ASR. [`ai-sage/GigaAM-Multilingual-v2`](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2) is 776M RNN-T for **21 languages**. Both are public, load with Transformers custom code (`trust_remote_code=True`), and return unformatted `raw` text plus formatted `e2e` text. v4 was pretrained on 7 million hours of unlabelled audio and trained on 58,000 hours of supervised speech; Multilingual-v2 on 10 million unlabelled hours and 155,000 supervised hours.

## Use cases

- **English or Russian transcripts** with the smaller v4 checkpoint when you want the card’s dual-head `raw` / `e2e` output (unformatted vs punctuation and capitalization) from `model.transcribe("example.wav")`.
- **The 21 languages on the Multilingual-v2 card** (Russian, English, German, Spanish, French, Italian, Dutch, Polish, Portuguese, Romanian, Azerbaijani, Bashkir, Belarusian, Armenian, Georgian, Kyrgyz, Kazakh, Tajik, Turkmen, Tatar, Uzbek) with an optional language tag (`language="fr"`) or `language="auto"`.
- **A small MIT stack** (v4 `model.safetensors` is 486.3 MB; Multilingual-v2 is 1.6 GB) that the cards load with `device="cuda"` and `fp16_encoder=True`.

## Not a good fit for

- **Languages outside those 21 tags** on Multilingual-v2, or anything other than English/Russian on v4.
- **Files longer than 30 seconds on `transcribe`.** The custom code raises and tells you to use `transcribe_longform`.
- **Streaming, ONNX, CTC, or SSL** in this notebook. The cards document `create_stream` (160 / 320 / 640 ms), ONNX export, v4 `revision="ctc"`, and `revision="ssl"`.
- **Speaker diarization.** Pair with [`Nemotron-3-Diarization.ipynb`](../colab/Nemotron-3-Diarization.ipynb).

## Running it on a free Colab T4

Pick a T4 GPU. The cards install `numpy "transformers<5" onnx onnxruntime hydra-core omegaconf sentencepiece tqdm`. `config.json` records `transformers_version: 4.57.1`. This notebook pins **`transformers>=4.57.1,<5`**, **`huggingface_hub>=0.34,<1`** (Transformers 4.57.x’s range), and **`soundfile`** (imported by `modeling_gigaam.py`). It does **not** reinstall torch. The cards’ load already uses **`fp16_encoder=True`** (fp16, not bf16; a T4 has no native bfloat16). Load v4, run the English/Russian demos, **free the model**, then load Multilingual-v2. Conservative download size is **~2 GB** (the 1.6 GB multilingual checkpoint). Peak VRAM is printed in the notebook; that is not a measured Colab number in this doc.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- The v4 card’s `AutoModel.from_pretrained(..., revision="rnnt", trust_remote_code=True, device="cuda", fp16_encoder=True)` and `model.transcribe(path)` on a public English clip.
- The same v4 call on a public Russian [`PolyAI/minds14`](https://huggingface.co/datasets/PolyAI/minds14) clip.
- Freeing v4, then the Multilingual-v2 card’s load (`language="auto"`) and `transcribe(..., language="fr")` on a French minds14 clip.
- Wall time vs audio duration for each clip.

## Links

- Model cards: [ai-sage/GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) · [ai-sage/GigaAM-Multilingual-v2](https://huggingface.co/ai-sage/GigaAM-Multilingual-v2)
- Source post: [x.com/aitrackerbot](https://x.com/aitrackerbot/status/2126228109972156712)
- Native package: [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM)
- License: [MIT](https://huggingface.co/ai-sage/GigaAM-v4/blob/main/LICENSE)
- Notebook: [`colab/GigaAM-v4-multilingual-asr.ipynb`](../colab/GigaAM-v4-multilingual-asr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/GigaAM-v4-multilingual-asr.ipynb)
- Related: [Rumble-2-Flash](Rumble-2-Flash-noisy-speech-to-text.md), [Fluid Speech Pro](Fluid-Speech-Pro-multilingual-asr.md), [Nemotron-Labs Unified ASR](Nemotron-Labs-Unified-ASR-0.6B-multilingual.md), [Granite Speech](granite-speech-5.0-2b-asr-translation.md), [oruk-loop-153M](oruk-loop-153M-looped-asr.md), [Whisper](AIBackends-Whisper-audio-transcription.md)
