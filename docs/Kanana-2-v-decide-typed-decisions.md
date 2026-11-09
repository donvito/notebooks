# Kanana-2-v-decide typed decisions

[`kakaocorp/kanana-2-v-decide-3b-2611`](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611) is Kakao’s multimodal decision model (Kanana Open License). Total size is 3.67B (0.49B vision encoder + 3.18B LM). Each question is encoded with the state into one embedding and each candidate into another; a softmax over cosine similarities is the answer. It is Korean-strong and accepts images. It does not write replies.

## Use cases

- **Korean (and English) support triage** with typed `noul` / `choice` / `score` questions.
- **The same noul question with a product photo**, which text-only IOTA/elephant do not take.
- **JSON-state routing** for agent-style payloads.

## Not a good fit for

- **Generating a customer reply.** Typed probabilities only.
- **Apache/MIT-only deployments.** Hub tag is `license: other` (Kanana Open License). Kakao’s 2026-11-06 press notes commercial use is allowed; read the LICENSE file.
- **Audio decisions.** Images and text/JSON, not speech. Use d1-Omni for audio.

## Running it on a free Colab T4

Pick a T4 GPU. Load float16 with SDPA/eager (the card is BF16). About 7.33 GB of weights. `trust_remote_code` as required by the custom `system_one` code.

## What the notebook demos

- Install and GPU check.
- Load Kanana-2-v-decide.
- Support triage (card example).
- Same noul question with a product photo.
- JSON-state routing.
- Caveats and cleanup.

## Links

- Model card: [kakaocorp/kanana-2-v-decide-3b-2611](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611)
- Source post: [x.com/aiupdatetoday](https://x.com/aiupdatetoday/status/2118589305820258788)
- License: [Kanana Open License](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/blob/main/LICENSE)
- Notebook: [`colab/Kanana-2-v-decide-typed-decisions.ipynb`](../colab/Kanana-2-v-decide-typed-decisions.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/Kanana-2-v-decide-typed-decisions.ipynb)
- Related: [IOTA Decide 4B](IOTA-Decide-4B-typed-decisions.md), [d1-2B](d1-2B-inference-decisions.md)
