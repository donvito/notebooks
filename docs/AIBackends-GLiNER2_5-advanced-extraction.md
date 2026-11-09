# AIBackends + GLiNER2.5 advanced extraction

Follow-up to the [GLiNER2.5 extraction notebook](AIBackends-GLiNER2_5-extraction.md). Same Fastino GLiNER2.5 family (Apache 2.0) through aibackends, covering remaining `examples/gliner25/` scripts: spans longer than GLiNER2's ~12-word cap, multilingual NER with `gliner2.5-multi-v1`, clinical span attributes, and multi-head operational decisions with `GLiNER2.5-Decide`.

## Use cases

- **Long quoted spans.** The boundary architecture scores span starts and ends, so a 25-word quote is in budget — useful for contract clauses or citation extraction.
- **Multilingual NER with English label names.** `gliner2.5-multi-v1` plus native batching for mixed-language inboxes.
- **Clinical span attributes.** Negation and medication status decoded in the same forward pass as the spans.
- **Operational decisions.** `model="decide"` loads Fastino's GLiNER2.5-Decide classifier for intent, routing, sentiment, priority, and policy, with several heads in one pass (policy filter, then support routing in the notebook).

## Not a good fit for

- **Medical diagnosis.** Clinical attributes are a demo of span+attribute decoding, not a validated clinical system.
- **Calibrated yes/no/score APIs over images or audio.** Decide here is text classification heads. For Liquid d1-style multimodal decisions, see the d1 notebooks.
- **First look at GLiNER2.5.** Start with the [extraction notebook](AIBackends-GLiNER2_5-extraction.md) for the core entity / classify / KG API.

## Running it on a free Colab T4

Works on a **free CPU runtime**; a T4 is faster and is detected automatically. Weight sizes for the advanced scripts are not separately documented here (the extraction notebook notes ~800 MB for the default base checkpoint).

## What the notebook demos

- Unlimited span width (`unlimited_spans.py`).
- Multilingual NER (`multilingual_ner.py`).
- Clinical span attributes (`span_attributes.py`).
- Operational decisions with GLiNER2.5-Decide (`decide_classification.py`), including policy filter then support routing.

## Links

- Model cards: [gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) · [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
- Notebook: [`colab/AIBackends-GLiNER2_5-advanced-extraction.ipynb`](../colab/AIBackends-GLiNER2_5-advanced-extraction.ipynb)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donvito/notebooks/blob/main/colab/AIBackends-GLiNER2_5-advanced-extraction.ipynb)
- Related: [GLiNER2.5 extraction](AIBackends-GLiNER2_5-extraction.md), [`GLiNER2_5-Decide-classification.ipynb`](../colab/GLiNER2_5-Decide-classification.ipynb)
