# Tweet + Hub sources

Reference table so we can jump back to the X/Twitter post that surfaced a model and the Hugging Face card the notebook actually loads. Newest ships first.

| Column | Meaning |
| --- | --- |
| Notebook | Path under `colab/` |
| Model | Short name as shipped |
| Hugging Face | Model card URL (must match the notebook) |
| Tweet source | Official / original announcement when known |
| PR | Squash-merged ship PR |
| Shipped (SGT) | Merge time of the squash commit, Singapore time |

New notebook PRs **must append a row here**. If a tweet or Hub link is unknown, write `—` plus a short reason, e.g. `— (unknown)`.

| Notebook | Model | Hugging Face | Tweet source | PR | Shipped (SGT) |
| --- | --- | --- | --- | --- | --- |
| [Qwenx86-4B-compact-in-weight-cpu.ipynb](colab/Qwenx86-4B-compact-in-weight-cpu.ipynb) | Qwenx86-4B compact | https://huggingface.co/Quazim0t0/Qwenx86-4B-compact | https://x.com/TeksEdge/status/2118857211992854892 | [#58](https://github.com/donvito/notebooks/pull/58) | 2026-11-09 |
| [Kanana-2-v-decide-typed-decisions.ipynb](colab/Kanana-2-v-decide-typed-decisions.ipynb) | Kanana-2-v-decide 3B (2611) | https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611 | https://x.com/aiupdatetoday/status/2118589305820258788 | [#57](https://github.com/donvito/notebooks/pull/57) | 2026-11-09 |
| [IOTA-Decide-4B-typed-decisions.ipynb](colab/IOTA-Decide-4B-typed-decisions.ipynb) | IOTA Decide 4B v0.4 | https://huggingface.co/macrocosm-os/iota-decide-4b-v0.4 | https://x.com/MacrocosmosAI/status/2118028289813971098 | [#56](https://github.com/donvito/notebooks/pull/56) | 2026-11-06 |
| [HaloGuard2-Stream-0.8B-streaming-output-guard.ipynb](colab/HaloGuard2-Stream-0.8B-streaming-output-guard.ipynb) | HaloGuard 2.0-Stream-0.8B | https://huggingface.co/astroware/HaloGuard2-Stream-0.8B | https://x.com/xnavkum/status/2118010382992457822 | [#55](https://github.com/donvito/notebooks/pull/55) | 2026-11-06 |
| [pplx-embed-v2-context-0.6b-contextual-rag.ipynb](colab/pplx-embed-v2-context-0.6b-contextual-rag.ipynb) | pplx-embed-v2-context-0.6b | https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b | https://x.com/tomaarsen/status/2117618555210641590 | [#52](https://github.com/donvito/notebooks/pull/52) | 2026-11-05 |
| [Audio8-TTS-Realistic-Air-voice-clone.ipynb](colab/Audio8-TTS-Realistic-Air-voice-clone.ipynb) | Audio8-TTS-Realistic-Air | https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air | https://x.com/SamuelZengML/status/2115881504102551856 | [#53](https://github.com/donvito/notebooks/pull/53) | 2026-11-05 |
| [NorthCUA-Nano-computer-use-agent.ipynb](colab/NorthCUA-Nano-computer-use-agent.ipynb) | NorthCUA-Nano | https://huggingface.co/beipo-ai/NorthCUA-Nano | https://x.com/PtVHVQ2FPQ51994/status/2117229952231702676 | [#51](https://github.com/donvito/notebooks/pull/51) | 2026-11-04 03:05 |
| [NetraV4-medical-vqa.ipynb](colab/NetraV4-medical-vqa.ipynb) | NetraV4 | https://huggingface.co/dipankar500/NetraV4 | https://x.com/VaidenceAI/status/2117133765973840126 | [#50](https://github.com/donvito/notebooks/pull/50) | 2026-11-03 03:03 |
| [Silencio-2-tts-voice-clone.ipynb](colab/Silencio-2-tts-voice-clone.ipynb) | Silencio-2 | https://huggingface.co/BlueRoseLabs/Silencio-2 | https://x.com/julian_inada/status/2116987775467835840 | [#49](https://github.com/donvito/notebooks/pull/49) | 2026-11-03 02:59 |
| [Mage-VL-Cowork-computer-use-agent.ipynb](colab/Mage-VL-Cowork-computer-use-agent.ipynb) | Mage-VL-Cowork | https://huggingface.co/microsoft/Mage-VL-Cowork | https://x.com/aitrackerbot/status/2116143353213428048 | [#47](https://github.com/donvito/notebooks/pull/47) | 2026-11-02 03:36 |
| [Xiaomi-OCR-0-document-ocr.ipynb](colab/Xiaomi-OCR-0-document-ocr.ipynb) | Xiaomi-OCR-0 | https://huggingface.co/xiaomi-research/Xiaomi-OCR-0 | https://x.com/QwenAIFans/status/2116705964217016382 | [#46](https://github.com/donvito/notebooks/pull/46) | 2026-11-02 03:09 |
| [d1-2B-inference-decisions.ipynb](colab/d1-2B-inference-decisions.ipynb) | d1-2B (invoice demo) | https://huggingface.co/LiquidAI/d1-2B | — (manual / follow-up demo) | [#45](https://github.com/donvito/notebooks/pull/45) | 2026-10-31 14:48 |
| [LFM3-1.7B-Uncensored-GGUF-llamacpp.ipynb](colab/LFM3-1.7B-Uncensored-GGUF-llamacpp.ipynb) | LFM3-1.7B Uncensored GGUF (demo rewrite) | https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF | https://x.com/atomic_chat_hq/status/2115183360138223808 | [#44](https://github.com/donvito/notebooks/pull/44) | 2026-10-31 13:46 |
| [LFM3-1.7B-Uncensored-GGUF-llamacpp.ipynb](colab/LFM3-1.7B-Uncensored-GGUF-llamacpp.ipynb) | LFM3-1.7B Uncensored GGUF | https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF | https://x.com/atomic_chat_hq/status/2115183360138223808 | [#43](https://github.com/donvito/notebooks/pull/43) | 2026-10-31 13:27 |
| [Carbon-X-inference-dna.ipynb](colab/Carbon-X-inference-dna.ipynb) | Carbon-X | https://huggingface.co/HuggingFaceBio/Carbon-X | https://x.com/_lewtun/status/2113214040516268126 | [#42](https://github.com/donvito/notebooks/pull/42) | 2026-10-30 03:50 |
| [Swar-voice-clone-tts.ipynb](colab/Swar-voice-clone-tts.ipynb) | Swar | https://huggingface.co/Aarjan/swar | https://x.com/ArjanChaudharyy/status/2115809203218755809 | [#41](https://github.com/donvito/notebooks/pull/41) | 2026-10-30 00:55 |
| [Smaug-Humanizer-llamacpp.ipynb](colab/Smaug-Humanizer-llamacpp.ipynb) | Smaug Humanizer | https://huggingface.co/abacusai/Smaug-Humanizer | https://x.com/bindureddy/status/2114938847704571975 | [#40](https://github.com/donvito/notebooks/pull/40) | 2026-10-30 00:50 |
| [Nemotron-Labs-Unified-ASR-0.6B-multilingual.ipynb](colab/Nemotron-Labs-Unified-ASR-0.6B-multilingual.ipynb) | Nemotron-Labs Unified ASR 0.6B | https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b | https://x.com/aitrackerbot/status/2115106347511423216 | [#39](https://github.com/donvito/notebooks/pull/39) | 2026-10-29 03:34 |
| [MiniCPM-OCR-1.0-document-ocr.ipynb](colab/MiniCPM-OCR-1.0-document-ocr.ipynb) | MiniCPM-OCR 1.0 | https://huggingface.co/openbmb/MiniCPM-OCR-1.0 | https://x.com/OpenBMB/status/2115124474882469955 | [#38](https://github.com/donvito/notebooks/pull/38) | 2026-10-29 03:24 |
| [Maya2-Native-indic-tts.ipynb](colab/Maya2-Native-indic-tts.ipynb) | Maya 2 Native | https://huggingface.co/maya-research/maya2-native | — (unknown) | [#36](https://github.com/donvito/notebooks/pull/36) | 2026-10-28 09:50 |
| [oruk-loop-153M-looped-asr.ipynb](colab/oruk-loop-153M-looped-asr.ipynb) | oruk-loop-153M | https://huggingface.co/oruk/oruk-loop-153M | — (unknown) | [#35](https://github.com/donvito/notebooks/pull/35) | 2026-10-28 09:49 |
| [Ming-3.0-tts-0.5B.ipynb](colab/Ming-3.0-tts-0.5B.ipynb) | Ming-3.0-tts 0.5B | https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B | — (unknown) | [#32](https://github.com/donvito/notebooks/pull/32) | 2026-10-28 09:49 |
| [Unlimited-OCR-2-document-ocr.ipynb](colab/Unlimited-OCR-2-document-ocr.ipynb) | Unlimited-OCR-2 | https://huggingface.co/baidu/Unlimited-OCR-2 | — (unknown) | [#33](https://github.com/donvito/notebooks/pull/33) | 2026-10-28 09:47 |
| [LFM3-1.7B-chat-tools.ipynb](colab/LFM3-1.7B-chat-tools.ipynb) | LFM3-1.7B | https://huggingface.co/LiquidAI/LFM3-1.7B | https://x.com/liquidai/status/2115128329003303107 | [#34](https://github.com/donvito/notebooks/pull/34) | 2026-10-28 02:36 |
| [LFM3-VL-2B-vision.ipynb](colab/LFM3-VL-2B-vision.ipynb) | LFM3-VL-2B | https://huggingface.co/LiquidAI/LFM3-VL-2B | https://x.com/liquidai/status/2115128329003303107 | [#34](https://github.com/donvito/notebooks/pull/34) | 2026-10-28 02:36 |
| [LFM3-Audio-2B-voice.ipynb](colab/LFM3-Audio-2B-voice.ipynb) | LFM3-Audio-2B | https://huggingface.co/LiquidAI/LFM3-Audio-2B | https://x.com/liquidai/status/2115128329003303107 | [#34](https://github.com/donvito/notebooks/pull/34) | 2026-10-28 02:36 |
| [LFM3-Omni-2B.ipynb](colab/LFM3-Omni-2B.ipynb) | LFM3-Omni-2B | https://huggingface.co/LiquidAI/LFM3-Omni-2B | https://x.com/liquidai/status/2115128329003303107 | [#34](https://github.com/donvito/notebooks/pull/34) | 2026-10-28 02:36 |
| [d1-2B-inference-decisions.ipynb](colab/d1-2B-inference-decisions.ipynb) | d1-2B | https://huggingface.co/LiquidAI/d1-2B | https://x.com/liquidai/status/2115128329003303107 | [#34](https://github.com/donvito/notebooks/pull/34) | 2026-10-28 02:36 |
| [d1-Omni-2B-inference-decisions.ipynb](colab/d1-Omni-2B-inference-decisions.ipynb) | d1-Omni-2B | https://huggingface.co/LiquidAI/d1-Omni-2B | https://x.com/liquidai/status/2115128329003303107 | [#34](https://github.com/donvito/notebooks/pull/34) | 2026-10-28 02:36 |
