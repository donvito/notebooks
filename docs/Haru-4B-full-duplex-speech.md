# Haru-4B full-duplex speech

[`higurashi-ai/Haru-4B`](https://huggingface.co/higurashi-ai/Haru-4B) is higurashi's pretrained **base** full-duplex spoken dialogue model (Apache-2.0, not gated). The Hub lists **5,270M** stored parameters including the audio codec (`model.safetensors` **10.5 GB**, bfloat16). The backbone is Qwen3.5-4B; audio is a causal VibeVoice tokenizer at 24 kHz, 7.5 Hz frames of 133 ms (3200 samples), with a one-step MeanFlow audio head. Nine languages on the card: en, es, pt, de, fr, zh, ja, ru, ko. It is released for research, not as an instruction-following assistant.

## Use cases

- **Try native full-duplex speech-to-speech without a live mic.** Every 133 ms the model takes one user-audio frame and emits one model-audio frame plus text. The notebook feeds a recording in those chunks, then silence, so you can hear a reply on Colab.
- **Character + voice control.** A fixed-layout text card sets the persona; a 3 s–3 min single-speaker clip sets the voice. Haru ships without voice clips — you bring a clip you have the right to use.
- **Spoken-dialogue research in the nine card languages.** The checkpoint is a multilingual base model for further fine-tuning, not a product assistant.

## Not a good fit for

- **An instruction-following voice assistant, or factual Q&A.** The card: it can hold a natural conversation, but it is not a reliable source of facts and does not follow instructions the way an assistant does.
- **Live microphone duplex on Colab.** The official web demo needs a local GPU server (or the MLX macOS app). This notebook is offline files only.
- **ASR-only transcripts or TTS-from-text.** Use [Fluid Speech Pro](Fluid-Speech-Pro-multilingual-asr.md) or [ZONOS3-Mini](ZONOS3-Mini-voice-clone-tts.md).
- **Cloning a real person without consent.** The card: only use voices you have the right to use. The notebook's references are public-domain LibriVox readings.

## Running it on a free Colab T4

Pick a T4 GPU. The card says an NVIDIA GPU needs about **12 GB**. Saved weights are bfloat16 (`model.safetensors` ~10.5 GB); 5.27B parameters in float16 are ~9.8 GB. A free T4 has no native BF16 and ~15 GB total. The card helper `HaruModel.from_pretrained(..., device="cuda", dtype=torch.float16)` loads the whole bf16 shard onto the GPU and then casts, which OOMs a T4 (~14.4 GB allocated). This notebook builds the module on the meta device in float16 and copies each tensor to CUDA already as float16 (one GPU copy; expect ~10–11.5 GB allocated after load). Restart the session after install so `transformers>=5.16` is what gets imported. Install is the card's `pip install -e ./haru` from [higurashi-ai/Haru](https://github.com/higurashi-ai/Haru), pinned to commit `7fe6b7665f8bf035998f401f8cb779fbe53f2751`, without replacing Colab's torch. The Triton codec extra is skipped (it needs bfloat16 on CUDA). First run downloads the 10.5 GB shard; the last cell prints measured peak VRAM.

## What the notebook demos

- Early GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card install, keeping Colab's torch, plus a restart note.
- Runtime check and T4 dtype (float16).
- T4-safe load: meta-device construct in float16, then one CUDA copy (not `from_pretrained(..., device="cuda")`).
- Public LibriSpeech dummy clips as the voice reference and user utterance (source stated).
- Demo 1: README character card, feed a prerecorded utterance in 3200-sample frames, pad with silence, save/play model audio, print the text stream.
- Demo 2: same user clip with the bundled English `chat` card and a different speaker as the voice.
- Demo 3: per-frame `step` time on this T4 vs the 133 ms realtime budget (a measurement, not a published benchmark).
- Caveats and cleanup, with peak VRAM.

## Links

- Model card: [higurashi-ai/Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B)
- Source post: [x.com/HigurashiAI](https://x.com/HigurashiAI/status/2121052314773717386)
- Technical blog: [blog.higurashi.ai/melte/haru4b](https://blog.higurashi.ai/melte/haru4b/)
- Code: [higurashi-ai/Haru](https://github.com/higurashi-ai/Haru)
- Notebook: [`colab/Haru-4B-full-duplex-speech.ipynb`](../colab/Haru-4B-full-duplex-speech.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Haru-4B-full-duplex-speech.ipynb)
- Related: [LFM3-Audio-2B](LFM3-Audio-2B-voice.md), [ZONOS3-Mini](ZONOS3-Mini-voice-clone-tts.md), [Fluid Speech Pro](Fluid-Speech-Pro-multilingual-asr.md)
