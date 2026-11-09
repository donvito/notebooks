# Gemma 4 E2B tweet style

This notebook QLoRA-tunes gated Gemma 4 E2B Instruct ([`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it), Apache 2.0) with Unsloth on your popular tweets so it can turn a topic or writing brief into a short post in a learned style. Upload a CSV, JSON, or JSONL file with at least a `text` column.

## Use cases

- **Personal or brand voice drafts** from a topic, using posts you actually published as the training set.
- **Compare several generations** after training to see whether the adapter picked up your phrasing.
- **Export LoRA, a merged model, or GGUF** for later serving.

## Not a good fit for

- **Impersonating someone else's account.** Train on posts you have the right to use.
- **Decision routing or email triage.** Those are separate Unsloth notebooks.
- **CPU-only training.** This is 4-bit QLoRA on a GPU.

## Running it on a free Colab T4

Start a GPU runtime (T4). Load Gemma 4 E2B in 4-bit. Weights are gated; accept the license on the Hub card. The notebook does not publish a measured VRAM figure beyond “a free T4 is the intended runtime.”

## What the notebook demos

- GPU runtime and Unsloth install.
- Load and clean tweets.
- Load Gemma 4 E2B in 4-bit.
- Optional writing briefs.
- Convert to Gemma 4 chat format, add LoRA, train.
- Plot loss, optional validation, test generations.
- Save adapter, optional merge, GGUF, Hub push.

## Links

- Model card: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- Notebook: [`colab/gemma4_e2b_tweet_style_colab.ipynb`](../colab/gemma4_e2b_tweet_style_colab.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/gemma4_e2b_tweet_style_colab.ipynb)
- Related: [Gemma 4 E2B model router](gemma4_e2b_model_router.md)
