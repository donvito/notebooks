# Qwenx86-4B compact in-weight CPU

[`Quazim0t0/Qwenx86-4B-compact`](https://huggingface.co/Quazim0t0/Qwenx86-4B-compact) is Qwen3.5-4B plus 13 compiled (not learned) chip layers that implement a 64-bit x86 computer in the weights (Apache-2.0). Tools such as `calc` and MicroPython are firmware in `model/boot/*.safetensors`. Load with the repo’s `qwenx86` compact loader, not `AutoModel`. This is a teaching demo of in-weight computation, not a faster CPU — machine-token tool runs take tens of seconds.

## Use cases

- **Show arithmetic happening inside the transformer**, checked against a reference machine.
- **Run the card’s live `calc` and a short `python 2**100`** without a host interpreter.
- **Inspect compiled-in firmware tools** (calc, MicroPython, qcc, notes, science lookup).

## Not a good fit for

- **Replacing a host CPU or calculator.** Card example: calc `(1200-350)/5` takes 56,667 tokens / ~41 s.
- **`AutoModelForCausalLM.from_pretrained`.** Wrong loader.
- **Email-triage training.** That is [Qwen3.5-4B decision email triage](qwen3_5_4b_decision_email_triage.md).
- **v2 host memory, v3 `memory_*` chat, mixed mode, GGUF, or the debugger Space recorder.** Out of scope.

## Running it on a free Colab T4

Pick a T4 GPU. Card: ~9 GB GPU, “fits on a 16 GB GPU.” Snapshot-download `model/compact.safetensors` (~8.75 GB) plus boot firmware. Official compact 16 GB loader (`safe_open` + per-tensor `.to("cuda")`). Do not invent a second quant. Keep Colab’s torch.

## What the notebook demos

- Install card deps, keep Colab’s torch.
- Load with `qwenx86`.
- `calc` (card live-chat).
- Short `python 2**100`.
- Optional extra tools commented out (do not Run all).

## Links

- Model card: [Quazim0t0/Qwenx86-4B-compact](https://huggingface.co/Quazim0t0/Qwenx86-4B-compact)
- Source post: [x.com/TeksEdge](https://x.com/TeksEdge/status/2118857211992854892)
- Qwenputer: [divagr18/qwenputer](https://github.com/divagr18/qwenputer)
- Notebook: [`colab/Qwenx86-4B-compact-in-weight-cpu.ipynb`](../colab/Qwenx86-4B-compact-in-weight-cpu.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Qwenx86-4B-compact-in-weight-cpu.ipynb)
