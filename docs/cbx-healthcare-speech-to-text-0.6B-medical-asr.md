# cbx-healthcare-speech-to-text-0.6B medical ASR

[`CobuildX-AI/cbx-healthcare-speech-to-text-0.6B`](https://huggingface.co/CobuildX-AI/cbx-healthcare-speech-to-text-0.6B) is a PEFT LoRA adapter on [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) for English doctor–patient consultation transcription (Apache-2.0, public). Pin revision **v0.1**. The card’s `SpeechToText` API is `stt.transcribe(path | (samples, sr) | 16 kHz array, context="expected words")` and `stt.transcribe_batch([...])`. On PriMock57 the card reports WER 6.01% vs 6.00% for the base, condition names wrong 5.4% vs 16.1%, medicine names wrong 33.9% vs 38.8%.

## Use cases

- **English doctor–patient consultation transcripts** when you want the card’s consultation-tuned LoRA rather than generic ASR. The card says it recognises medical conditions correctly 94.6% of the time (base: 83.9%) and medicine names 66.1% (base: 61.2%) in consultations, while matching the base model on everyday words.
- **A clinic medicine-list hint** via `context=`, for example the card’s `stt.transcribe("visit.wav", context="Clenil, salbutamol")`. The card says this may help with rare names.
- **A small open Apache-2.0 stack** (0.6B base + ~40 MB adapter) that the card says runs on one GPU or on a CPU. On an NVIDIA T4 the card reports 1 minute of speech in about 2.4 s (fp16, batches of 16 clips).

## Not a good fit for

- **Clinical records without a human in the loop.** The card: have transcripts reviewed before they are used.
- **Non-English audio.** English only.
- **Assuming measured accuracy on non-US accents.** The card: training speech is US-based; accuracy on other accents has not been fully measured.
- **Lectures or single short sentences.** The card: for other speech, such as lectures or single short sentences, the base model is currently more accurate.
- **Expecting medicine names to be reliable.** The card: still wrong about one time in three on PriMock57; common names do better (paracetamol 1 wrong in 29, ibuprofen 4 in 25); most remaining mistakes are rare or UK brand names that never appear in the training data (Clenil, Implanon, Dioralyte).

## Running it on a free Colab T4

Pick a T4 GPU. The card install is `hf download ... --revision v0.1` then `pip install -r requirements.txt`. That file pins `qwen-asr==0.0.6`, `transformers==4.57.6`, `peft==0.21.2`, `librosa`, and `torch>=2.4`. The notebook **does not reinstall torch** (Colab already has a CUDA wheel) and skips `qwen-asr` CLI extras (`gradio`, `flask`, `sox`). It also **uninstalls Colab’s `torchao` 0.10.0** (`peft==0.21.2` raises if that version is present; this notebook does not use torchao) and pins **`huggingface_hub>=0.34,<1`** (transformers 4.57.6’s range; hub 2.x is incompatible and pops Colab `HF_TOKEN` secrets). The install cell restarts the session so `transformers==4.57.6` is what gets imported. `SpeechToText` merges the adapter in fp32, then uses float16 on CUDA. First run downloads the ~1.9 GB `Qwen/Qwen3-ASR-0.6B` checkpoint plus the ~40 MB adapter. The card’s T4 speed figure (about 2.4 s per minute of speech, fp16, batch 16) is from the card, not a measurement from this notebook. Peak VRAM is printed at the end of the notebook.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card install at revision v0.1, keeping Colab’s torch, uninstalling torchao, pinning `huggingface_hub<1`, then an automatic session restart.
- `SpeechToText` load (`Qwen/Qwen3-ASR-0.6B` + this adapter).
- PriMock57 mock GP clips from the card’s `example.py` (`day1_consultation01_doctor`), reference vs output, with a public-audio fallback if that download fails.
- Base `Qwen3-ASR-0.6B` vs the adapter on the same clip, skipped on CUDA OOM.
- `context=` with the card’s medicine-list example, plus Dioralyte/paracetamol when those words appear in the PriMock57 reference.
- Optional upload-your-own-audio cell (off by default so Run all does not block).
- Peak VRAM.

## Links

- Model card: [CobuildX-AI/cbx-healthcare-speech-to-text-0.6B](https://huggingface.co/CobuildX-AI/cbx-healthcare-speech-to-text-0.6B)
- Base model: [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- Source post: [x.com/sairaamv](https://x.com/sairaamv/status/2122704454306922853)
- PriMock57: [github.com/babylonhealth/primock57](https://github.com/babylonhealth/primock57)
- License: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Notebook: [`colab/cbx-healthcare-speech-to-text-0.6B-medical-asr.ipynb`](../colab/cbx-healthcare-speech-to-text-0.6B-medical-asr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/cbx-healthcare-speech-to-text-0.6B-medical-asr.ipynb)
- Related: [Fluid Speech Pro](Fluid-Speech-Pro-multilingual-asr.md), [Nemotron-Labs Unified ASR](Nemotron-Labs-Unified-ASR-0.6B-multilingual.md), [Granite Speech](granite-speech-5.0-2b-asr-translation.md), [Whisper](AIBackends-Whisper-audio-transcription.md)
