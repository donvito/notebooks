# Smaug Humanizer

[`abacusai/Smaug-Humanizer`](https://huggingface.co/abacusai/Smaug-Humanizer) is Abacus.AI’s ~11.96B rewriter (Apache 2.0), a fine-tune of `google/gemma-4-12B`. It takes stiff chatbot prose and returns the same facts, numbers, names, and layout in a more human voice. Inference only. Split long drafts at about 500 words.

## Use cases

- **Rewrite AI-sounding product updates, explainers, or support emails** without dropping dates and numbers.
- **Chunk-and-join long drafts** (card: about 500 words per call).
- **Official Q4_K_M on a T4** instead of the 24 GB BF16 checkpoint.

## Not a good fit for

- **Third-party re-quants.** Use only `abacusai/Smaug-Humanizer-GGUF`.
- **Training.** This notebook does not fine-tune.
- **BF16 on a free T4.** About 24 GB; does not fit.

## Running it on a free Colab T4

Pick a T4 GPU. Official Q4_K_M GGUF (~7.38 GB) via `llama-cpp-python`. Q5_K_M (~8.55 GB) is optional. `flash_attn=False`, 8,192-token context. Weights plus that cache land around 8–10 GB on 15 GB.

## What the notebook demos

- Load official Q4_K_M.
- Humanize helper.
- Stiff product update, generic explainer, template support email.

## Links

- Model card: [abacusai/Smaug-Humanizer](https://huggingface.co/abacusai/Smaug-Humanizer)
- GGUF: [Smaug-Humanizer-GGUF](https://huggingface.co/abacusai/Smaug-Humanizer-GGUF)
- Blog: [Smaug Humanizer](https://blog.abacus.ai/blog/2026/10/26/smaug-humanizer-an-open-weight-12b-humanizer/)
- Source post: [x.com/bindureddy](https://x.com/bindureddy/status/2114938847704571975)
- Notebook: [`colab/Smaug-Humanizer-llamacpp.ipynb`](../colab/Smaug-Humanizer-llamacpp.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Smaug-Humanizer-llamacpp.ipynb)
