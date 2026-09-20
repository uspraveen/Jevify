# Jevify

**Turn any open LLM into a calibrated Jev-style System One decision model.**

A *System One* model doesn't write text. It reads a `state`, answers typed questions, and
returns probability distributions your code can branch on:

| primitive | question | answer |
|---|---|---|
| `choice` | which of these K options? | probabilities over the options + `confidence` |
| `score`  | where on these K ordered levels? | probabilities over levels, expected `score`, `confidence` |
| `noul`   | is this true? | a single `P(yes)` |

TypeSafe AI's [Jev](https://typesafe.ai) showed this is a real product category: fast
(~100–500 ms), cheap, and — the important part — **calibrated**: an answer given 0.8 should be
right about 80% of the time. Jev is closed: no paper, no weights, no data. Jevify is the open
version — a recipe, a benchmark, and a serving stack, for any base model.

## Status — research phase, benchmark first

- [x] Wire format identical to TypeSafe's API (`jevify/wire.py`), so the official
      `typesafe-sdk` and integrations work against a Jevified model via `TYPESAFE_BASE_URL`.
- [x] **jev-bench**: 21 human-labeled sources reformatted into System One questions, with human
      label *distributions* where available (`jevify/bench`). → `Praveenrajus/jev-bench` on the Hub.
- [x] Calibration metrics (ECE, Brier, NLL, RPS, selective accuracy, AURC, divergence to human
      distributions) and an API runner that scores Jev itself on the same data.
- [x] **Baseline: Jev 1.13.0 on all 22,773 test records** → [`reports/jev-1.13.0`](reports/jev-1.13.0/README.md).
      Headline: excellent and calibrated on crisp tasks (ARC 97.9%, FEVER 97.2%, ECE ≤ 0.06), but far from
      human label distributions where humans disagree (ChaosNLI TVD 0.33, ECE 0.22) — the axis to compete on.
- [ ] Tier 0 runner: any HF checkpoint, zero training — logit readout + debiasing + temperature.
- [ ] Tier 1: frozen backbone + decision heads trained with proper scoring rules.
- [ ] Tier 2: + LoRA on the backbone, only where the benchmark shows it pays.
- [ ] Label-first synthetic data pipeline (the moat, per TypeSafe's own account).
- [ ] Server (`/v1/systemone`), HF Space, model zoo, VLM backbones, vision-tower autoresearch.

![Jev 1.13.0 model vs human probability](reports/jev-1.13.0/figures/human_vs_model.png)

## Why tiers, and why no RL

Calibration comes from optimizing a strictly proper scoring rule against real outcomes.
TypeSafe calls their version RLCD. When the model *emits* a decision you need RL to push that
objective through a sampling step; when the decision is read from a differentiable head on the
backbone's hidden states, the same objective is just a loss. So Jevify trains heads (and
optionally LoRA) by gradient descent on log/Brier/ranked-probability losses — no reward model,
no rollouts — and each tier is justified only by the number it moves on jev-bench.

## Quickstart

```bash
pip install -e ".[bench,dev]"
jevify-bench list                                   # the 21 sources
jevify-bench build --out data/jev-bench             # ~200k rows; --evict-cache on small disks
TYPESAFE_API_KEY=... jevify-run api --records data/jev-bench --out preds/jev.jsonl --limit 200
jevify-run report --records data/jev-bench --preds preds/jev.jsonl --md reports/jev.md
```

Docs: [`docs/JEV_CONTRACT.md`](docs/JEV_CONTRACT.md) (what we verified about Jev's API),
[`docs/DATASETS.md`](docs/DATASETS.md) (source selection and sampling policy).

## License

Apache-2.0 for the code. Benchmark rows carry their upstream licenses (see the manifest).
