# LightOnOCR-3 4B document OCR

[`lightonai/LightOnOCR-3-4B`](https://huggingface.co/lightonai/LightOnOCR-3-4B) is LightOn’s Apache 2.0 ~4.5B grounded OCR quality upgrade of the 0.8B notebook, aimed at multi-column pages, dense tables, and forms.

## Use cases

- **Harder layout pages** where 0.8B quality is not enough.
- **Forms and dense tables** at the cost of 4-bit weights on a T4.
- **A same-recipe comparison** with the 0.8B notebook.

## Not a good fit for

- **Speed on simple receipts.** Prefer [0.8B](LightOnOCR-3-0.8B-document-ocr.md).
- **Full float16 on a free T4.** Hub bf16 is 9.1 GB and leaves little room for a 2048 px page.
- **2048 px quality recipe on 16 GB.** This notebook defaults to 1540 px (LightOn’s higher-throughput resolution). Set `USE_4BIT = False` and `LONGEST = 2048` only on a GPU larger than 16 GB.

## Running it on a free Colab T4

Pick a T4 GPU. Default is 4-bit NF4 with SDPA and longest edge 1540 px. No native BF16, no FlashAttention 2.

## What the notebook demos

- 4-bit load of LightOnOCR-3-4B.
- Harder pages than the 0.8B notebook (multi-column, dense tables, forms).
- Notes on switching off 4-bit / 2048 px on larger GPUs.

## Links

- Model card: [lightonai/LightOnOCR-3-4B](https://huggingface.co/lightonai/LightOnOCR-3-4B)
- Notebook: [`colab/LightOnOCR-3-4B-document-ocr.ipynb`](../colab/LightOnOCR-3-4B-document-ocr.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LightOnOCR-3-4B-document-ocr.ipynb)
- Related: [LightOnOCR-3 0.8B](LightOnOCR-3-0.8B-document-ocr.md)
