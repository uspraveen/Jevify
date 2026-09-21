---
title: Jevify Playground
emoji: 🎲
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Ask an open System One model typed questions, see calibrated probabilities
models:
  - Praveenrajus/jevify-qwen3.5-4b
  - Praveenrajus/jevify-qwen3.5-2b
datasets:
  - Praveenrajus/jev-bench
---

# Jevify playground

A **System One** model doesn't write text. It reads a `state`, answers typed questions, and
returns calibrated probability distributions your code can branch on:

| primitive | question | answer |
|---|---|---|
| `choice` | which of these K options? | probabilities over the options + `confidence` |
| `score` | where on these K ordered levels? | probabilities over levels, expected `score` |
| `noul` | is this true? | a single `P(yes)` |

These are open checkpoints given that interface by [Jevify](https://github.com/uspraveen/Jevify)
and scored against TypeSafe's Jev on [jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench).
The probabilities are read from the model's own logits — nothing is parsed out of generated text.

The same wire format as TypeSafe's API: `jevify-serve --model <repo>` exposes `/v1/systemone`,
and the official `typesafe-sdk` works against it by setting `TYPESAFE_BASE_URL`.
