# Timestral-209M forecast

[`mistralai/Timestral-209M`](https://huggingface.co/mistralai/Timestral-209M) is Mistral’s 209M zero-shot time-series foundation model. Hub tag is `other` (Modified MIT — not standard MIT). About 837 MB of float32 weights, 21 quantile bands. Optional side-by-side with Timestral-5M.

## Use cases

- **Zero-shot forecasts** without training a per-series model.
- **Quantile bands** (21) for uncertainty, not a single point estimate.
- **A CPU-friendly 209M path** that also runs on a T4.

## Not a good fit for

- **Assuming standard MIT.** Read the Modified MIT LICENSE file.
- **A full Timestral-5M bake-off as the default.** 5M is optional in the notebook.

## Running it on a free Colab T4

T4; CPU also works. About 837 MB of float32 weights.

## What the notebook demos

- Zero-shot forecast with Timestral-209M.
- 21 quantile bands.
- Optional Timestral-5M comparison.

## Links

- Model card: [mistralai/Timestral-209M](https://huggingface.co/mistralai/Timestral-209M)
- GitHub: [mistralai/timestral](https://github.com/mistralai/timestral/)
- License: [LICENSE](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE)
- Notebook: [`colab/Timestral-209M-forecast.ipynb`](../colab/Timestral-209M-forecast.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Timestral-209M-forecast.ipynb)
