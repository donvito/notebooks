# Colab Notebooks

A collection of runnable notebooks for experimenting with open models in Google Colab and compatible local Jupyter environments.

## Notebooks

### LFM2.5-2.6B with `llama-cpp-python`

[Open notebook](colab/LFM-2_5-2_6b-llamacpp-python.ipynb) · [Open in Colab](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/LFM-2_5-2_6b-llamacpp-python.ipynb)

Demonstrates Liquid AI's LFM2.5-2.6B model using its Q4_K_M GGUF checkpoint. The notebook includes:

- A pinned CUDA-enabled `llama-cpp-python` wheel
- Full model offload to an NVIDIA GPU
- A 16,384-token context window
- Basic chat generation with finish-reason inspection
- Repeatable latency and tokens-per-second benchmarking
- A two-pass multi-tool workflow using weather and local-time examples

Requirements: an NVIDIA T4 GPU or better. In Colab, select **Runtime > Change runtime type > T4 GPU** before running the notebook. If a CPU build of `llama-cpp-python` was already imported, install the CUDA wheel, restart the session, and then run the notebook from the beginning.

## Getting started

1. Open a notebook directly or launch it in Colab.
2. Select the required accelerator for the notebook.
3. Run the cells from top to bottom.
4. Review each model's license and usage terms before deploying its output.

Local execution may require adapting the accelerator-specific installation cell to your operating system and CUDA or Metal environment.

## Repository structure

```text
.
├── colab/
│   └── LFM-2_5-2_6b-llamacpp-python.ipynb
├── .gitignore
└── README.md
```
