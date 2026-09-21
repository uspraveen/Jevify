---
library_name: jevify
pipeline_tag: text-classification
tags: [jevify, system-one, decision-model, calibration, jev]
base_model: Qwen/Qwen3.5-2B
license: apache-2.0
datasets: [Praveenrajus/jev-bench]
---

# jevify-qwen3.5-2b

A **System One decision model**: it does not write text. It reads a `state`, answers typed
questions, and returns calibrated probability distributions your code can branch on.

| primitive | question | answer |
|---|---|---|
| `choice` | which of these K options? | probabilities over the options + `confidence` |
| `score` | where on these K ordered levels? | probabilities over levels, expected `score`, `confidence` |
| `noul` | is this true? | a single `P(yes)` |

This repo holds only what Jevify adds to `Qwen/Qwen3.5-2B`: **2,628,107 parameters** of trained
decision heads (10.5 MB) plus the calibration recipe. The backbone is pulled from its
own repo at load time, so nothing is duplicated or relicensed.

## Results on [jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench)

Scored on all 22,773 test records, against TypeSafe's Jev 1.13.0 on the identical records.

| model | held-out sources: acc / ECE | sources seen in training: acc / ECE |
|---|---|---|
| Qwen3.5-2B (Tier 1 residual) — Tier 1 heads | 0.631 / 0.090 | 0.632 / 0.061 |
| Jev 1.13.0 — API (zero-shot) | 0.835 / 0.090 | 0.694 / 0.122 |
| Qwen/Qwen3.5-2B — Tier 0 (recipe refit w/o held-out) | 0.628 / 0.097 | 0.552 / 0.100 |

Across the whole benchmark: **macro accuracy 0.632** against Jev's 0.733, **ECE 0.069** against 0.113, and **0.374** mean distance to human label distributions against Jev's 0.432 — lower is better, and that last number is the one a calibration claim rests on.

## Use it

```bash
pip install git+https://github.com/uspraveen/Jevify
```

```python
from jevify import load_jevified

model = load_jevified("Praveenrajus/jevify-qwen3.5-2b")
answer = model.ask(
    state={"ticket": "I was charged twice for order A-104, please refund the duplicate."},
    questions={
        "dept": {"type": "choice", "instructions": "Which team should handle `ticket`?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems", "other": None}},
        "refund": {"type": "noul", "instructions": "Does `ticket` ask for a refund?"},
        "anger": {"type": "score", "instructions": "How angry is the customer?",
                  "criteria": ["calm", "annoyed", "furious"]},
    },
)
print(answer["dept"]["choice"], answer["dept"]["confidence"])
print(answer["refund"]["noul"])
```

### As a drop-in for the TypeSafe API

```bash
jevify-serve --model Praveenrajus/jevify-qwen3.5-2b --port 8000
```

```bash
TYPESAFE_BASE_URL=http://localhost:8000 python your_existing_typesafe_code.py
```

The official `typesafe-sdk` works against this unchanged — that is a test in the repo.

## How it was built

Decision heads read the backbone's hidden state at each option's own line, so they score what an option *means* rather than how likely its identifier token is. They are applied as a **residual on the model's own log-score** — `score_i = w·lm_i + f(...)` with `f` zero-initialized — so training starts exactly at the untrained baseline and can only add to it. Trained on 8,685 records from 16 sources with at most 16 options each, which is what keeps the head usable at any K.

Training optimizes strictly proper scoring rules directly — log score for Choice and Noul,
ranked probability score for the ordinal Score — so calibration is the objective rather than a
post-hoc repair. No reinforcement learning is involved: with a differentiable head the
calibration objective is just a loss.

Six sources were **held out of training entirely** so generalization to unseen question types is
measured rather than assumed. Full method, findings and limitations:
[github.com/uspraveen/Jevify](https://github.com/uspraveen/Jevify) ·
[FINDINGS.md](https://github.com/uspraveen/Jevify/blob/main/docs/FINDINGS.md)

## Limitations

- English-first, text only, following the benchmark it was tuned on.
- Ordinal (`score`) questions on scales unlike those in training are the weakest case.
- The heads are trained on jev-bench's own train splits, so "held out" means held-out *source*,
  not a wholly different data universe.
- One seed per backbone. Directions replicate across two backbones; magnitudes will move.
