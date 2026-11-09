# LFM3-Omni-2B

[`LiquidAI/LFM3-Omni-2B`](https://huggingface.co/LiquidAI/LFM3-Omni-2B) is Liquid AI’s ~2.17B omni model (LFM Open License v1.0) for image, ASR, and short video → text. Needs `transformers>=5.20.2` and `trust_remote_code`. This notebook is a multimodal smoke, not a full speech-to-speech demo.

## Use cases

- **One checkpoint that can see a picture, hear a clip, and read a few video frames** when you only need text out.
- **A T4-sized omni smoke test** before you commit to a larger omni stack.

## Not a good fit for

- **Speech output on a T4.** Demos ask for text (`generation_mode="text"`) so the T4 is not also decoding speech.
- **Long video.** Video is capped at 8 frames / 2 fps.
- **Calibrated decisions over omni inputs.** Use [d1-Omni-2B](d1-Omni-2B-inference-decisions.md).

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 with SDPA. About 4.3 GB of weights plus LiquidCodec. Video capped at 8 frames / 2 fps.

## What the notebook demos

- Load LFM3-Omni-2B.
- Image → text.
- ASR → text.
- Short video → text.

## Links

- Model card: [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B)
- Source post: [x.com/liquidai](https://x.com/liquidai/status/2115128329003303107)
- Blog: [LFM3](https://www.liquid.ai/blog/lfm3)
- Notebook: [`colab/LFM3-Omni-2B.ipynb`](../colab/LFM3-Omni-2B.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LFM3-Omni-2B.ipynb)
- Related: [LFM3-VL-2B](LFM3-VL-2B-vision.md), [LFM3-Audio-2B](LFM3-Audio-2B-voice.md), [d1-Omni-2B](d1-Omni-2B-inference-decisions.md)
