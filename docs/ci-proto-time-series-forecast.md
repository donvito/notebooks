# ci-proto time-series forecast

[`ContinuousIntelligence/ci-proto`](https://huggingface.co/ContinuousIntelligence/ci-proto) is Continuous Intelligence's **167M** pretrained time-series foundation model for **zero-shot probabilistic forecasting**. It is public (not gated). Outputs are **nine quantiles** from 0.1 to 0.9. Inputs are one or more target series, with optional past and future covariates, timestamps, and related series. `model.ckpt` (~669 MB) covers histories up to **16,384** steps; `model-long.ckpt` is selected automatically for longer histories.

## Use cases

- **A next-day or next-week demand sketch without training a per-series model.** The Hub card's bundled example is five days of hourly bicycle rentals in Washington, D.C., then a 24-hour forecast with nested quantile bands.
- **Planning with a range, not a single point.** The nine quantiles (0.1, 0.2, …, 0.9) are the model's uncertainty grid. The package walkthrough uses them as intervals, exceedance-style thresholds, and a rough calibration check — they are model quantiles, not a promise that 80% of future points will land in the 10–90% band.
- **Known-future covariates and related series.** The [package README](https://github.com/continuousintelligence/ci#usage-details) takes `past_covariates` / `future_covariates` (optional categorical row indices), timestamps plus a pandas `freq` for calendar features, and jointly forecast rows (`[variables, time]`). Mark gaps as `NaN`; do not impute first.

## Not a good fit for

- **Chat, code, or any language-model task.** ci-proto is not an LLM. There is no tokenizer, chat template, or `generate()`.
- **Treating Apache-2.0 as the whole licence.** The Hub tag is `other`. Weights and inference code use the [ci-proto Code and Model Licence v1.0](https://huggingface.co/ContinuousIntelligence/ci-proto/blob/main/LICENCE) (Apache-2.0 based, plus Section 10). Redistribution and externally available products need public attribution and a one-time email to Continuous Intelligence. Purely internal use is exempt from Section 10.
- **Fused Mamba-3 kernels on a free T4.** The package README: fused inference needs Ampere-or-newer NVIDIA GPUs. A Colab T4 is Turing (`sm_75`), so this notebook stays on the portable `"torch"` backend.
- **A leaderboard run.** This notebook is a smoke test on the bundled bike-rental series. The card reports zero-shot fev-bench results separately; this repo does not rerun them.

## Running it on a free Colab T4

Pick a T4 if you have the choice (*Runtime > Change runtime type > T4 GPU*). CPU also works. First run downloads `model.ckpt` (~669 MB / 0.7 GB). `model-long.ckpt` (another ~669 MB) is fetched only if a history is longer than 16,384 steps; the demos stay under that cap.

`ci.load_model` uses CUDA when `torch.cuda.is_available()`, else CPU. On Linux x86-64 a plain `pip install continuousintelligence` also pulls `cuda-toolkit==13.0.3`, `tilelang`, and `apache-tvm-ffi` for fused kernels. Those extras are **not** used on a T4 and would fight Colab's CUDA torch, so the notebook installs the package with `--no-deps` and only `einops`, `huggingface-hub`, `pandas`, and `matplotlib`. Keep Colab's torch. Do not install `flash-attn`.

The released `tta.json` sets `bfloat16: true`. A T4 has no native bfloat16; PyTorch autocast then runs in float32. The package README says reduced-precision results can vary across runtimes.

## What the notebook demos

- GPU / CPU check (`torch.cuda.is_available()`, device name, free VRAM).
- Install `continuousintelligence` without replacing Colab's torch.
- Load `ContinuousIntelligence/ci-proto` and print `model.info`, quantiles, device, and backend.
- Bundled bike-rental history: 24-step forecast and `ci.plot_forecast` quantile plot.
- Calendar timestamps plus a known-future working-day / hour-of-day covariate tensor (package README shapes).
- Missing observations marked as `NaN`.

## Links

- Model card: [ContinuousIntelligence/ci-proto](https://huggingface.co/ContinuousIntelligence/ci-proto)
- Source post: [x.com/Continu_Intel](https://x.com/Continu_Intel/status/2120852944056709139)
- Package / API: [continuousintelligence/ci](https://github.com/continuousintelligence/ci)
- Walkthrough: [examples/walkthrough.ipynb](https://github.com/continuousintelligence/ci/blob/main/examples/walkthrough.ipynb)
- Licence: [ci-proto Code and Model Licence](https://huggingface.co/ContinuousIntelligence/ci-proto/blob/main/LICENCE)
- Notebook: [`colab/ci-proto-time-series-forecast.ipynb`](../colab/ci-proto-time-series-forecast.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/melvin/ci-proto-time-series-forecast-7e78/colab/ci-proto-time-series-forecast.ipynb)
- Related: [Timestral-209M forecast](Timestral-209M-forecast.md)
