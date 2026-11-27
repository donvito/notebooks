# Parakeet Singlish v2 ASR

[`Scrya/parakeet-tdt-0.6b-singlish-v2`](https://huggingface.co/Scrya/parakeet-tdt-0.6b-singlish-v2) is Scrya’s 0.6B Parakeet-TDT fine-tune of [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0, public) for Singapore English / Singlish ASR. The card: 1,386 hours of conversational Singapore English from IMDA National Speech Corpus Parts 3–6, trained for meeting rooms, punctuated English, numbers spelled out as words.

## Use cases

- **Singapore English / Singlish transcripts** from meeting-style speech (the card chose the data and augmentations for people talking in one room, with office noise, background talkers, and room echo mixed in during training).
- **Punctuated captions** when you want casing and punctuation in the transcript. The card: the model writes punctuated English and spells numbers as words (e.g. "twenty five" instead of "25").
- **A small open CC-BY-4.0 checkpoint** (~627M parameters; transformers `model.safetensors` 2.4 GB fp32 on the card) instead of a 3B Singapore ASR model. The card reports lower normalised WER than MERaLiON-3-3B-ASR on the same 66,279 NSC test clips (9.16% vs 10.06% over all five parts).

## Not a good fit for

- **Long unsegmented recordings.** The card: trained and tested only on NSC speech, and most clips are under 20 seconds. Split long audio first (e.g. with a voice activity detector).
- **Non-English or code-switched speech as written in those languages.** The card: finetuned on English only; NSC speakers often switch to Mandarin, Malay or other languages, and the model writes English words for that speech or leaves it out. The card also expects the finetuning made the 24 other European languages of the base model worse, but did not test them.
- **Heavily overlapping talk, or treating the noise-test numbers as real-meeting numbers.** The card: scores skip clips where the other speaker covers more than 30% of the clip; the noise test mixes noise into clean recordings, and real meeting recordings may behave differently. Pair speaker labels with [Nemotron-3 Diarization](Nemotron-3-Diarization.md).
- **NeMo-Speech.cpp GGUF in this notebook.** GGUF files are in separate repos ([f16](https://huggingface.co/Scrya/parakeet-tdt-0.6b-singlish-v2-nemo-speech-cpp-f16), [q8_0](https://huggingface.co/Scrya/parakeet-tdt-0.6b-singlish-v2-nemo-speech-cpp-q8_0)), not in this model repo.
- **bfloat16 on a free T4.** A T4 has no native bfloat16. The card decoded in fp32 because greedy decoding in bf16 gave slightly different text for different batch sizes.

## Running it on a free Colab T4

Pick a T4 GPU. This notebook uses the card’s **🤗 Transformers** path (`AutoProcessor` + `ParakeetForTDT`), which is lighter than installing NeMo. `config.json` records `transformers_version: 5.22.1`; the notebook pins **`transformers>=5.22.0`**. Saved weights are float32 (`model.safetensors` 2.4 GB on the card). A free T4 has no native bfloat16, so the notebook loads **float16**. Set `DTYPE = torch.float32` if a transcript looks degraded — that is the dtype the card scored with. Short clips should need a few GB of VRAM; that is an estimate from the checkpoint size, not a measured Colab number.

The card checked the `.nemo` file with **nemo_toolkit 2.7.3**. Transformers 5.22.1 greedy TDT decoding ignores `max_symbols_per_step` (config value 3). On rare clips the model can loop; the card says to use the NeMo file or a GGUF file for exact results.

## What the notebook demos

- GPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Card Transformers load (`ParakeetForTDT.from_pretrained`, `dtype` float16 on T4).
- A public English LibriSpeech-style clip (`librispeech_mr_quilter.wav` from [`hf-internal-testing/dummy-audio-samples`](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples)).
- A public Singapore-English / Singlish conversational clip from [`mjwong/SASRBench-v1`](https://huggingface.co/datasets/mjwong/SASRBench-v1) (Hub license MIT; derived from IMDA NSC Part 3 Same Room close-talk), with a fallback English clip if that download fails.
- Token timestamps from the Transformers Parakeet TDT `processor.decode(..., durations=...)` API, plus wall time. `generate` sets `max_new_tokens=1024` so a ~20 s clip is not cut by Transformers’ default `max_length=222`, and a looping decode cannot run on forever.

## Links

- Model card: [Scrya/parakeet-tdt-0.6b-singlish-v2](https://huggingface.co/Scrya/parakeet-tdt-0.6b-singlish-v2)
- Base model: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Source post: [x.com/testdanrun](https://x.com/testdanrun/status/2126149023539122644)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — credit **Scrya** for the finetuned model and **NVIDIA Parakeet-TDT-0.6B-v3** for the base model
- Notebook: [`colab/Parakeet-Singlish-v2-asr.ipynb`](../colab/Parakeet-Singlish-v2-asr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Parakeet-Singlish-v2-asr.ipynb)
- Related: [Fluid Speech Pro](Fluid-Speech-Pro-multilingual-asr.md) (same Parakeet-TDT-0.6B-v3 lineage, 25 European languages), [Nemotron-Labs Unified ASR](Nemotron-Labs-Unified-ASR-0.6B-multilingual.md), [Rumble-2-Flash](Rumble-2-Flash-noisy-speech-to-text.md), [Granite Speech](granite-speech-5.0-2b-asr-translation.md), [oruk-loop-153M](oruk-loop-153M-looped-asr.md), [Whisper](AIBackends-Whisper-audio-transcription.md)
