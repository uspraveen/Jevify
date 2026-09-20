<div align="center">

# Jevify

**Turn any open LLM into a calibrated, Jev-style System One decision model.**

[jev-bench on the Hub](https://huggingface.co/datasets/Praveenrajus/jev-bench) ·
[What we verified about Jev](docs/JEV_CONTRACT.md) ·
[Dataset rationale](docs/DATASETS.md) ·
[Baseline report + label audit](reports/jev-1.13.0/README.md)

</div>

A *System One* model doesn't write text. It reads a `state`, answers typed questions, and returns
probability distributions your code can branch on:

| primitive | question | answer |
|---|---|---|
| `choice` | which of these K options? | probabilities over the options + `confidence` |
| `score`  | where on these K ordered levels? | probabilities over levels, expected `score`, `confidence` |
| `noul`   | is this true? | a single `P(yes)` |

TypeSafe's [Jev](https://typesafe.ai) made this a product category: fast, cheap, and — the actual
claim — **calibrated**: an answer given 0.8 should be right about 80% of the time. Jev ships no
paper, weights or data. Jevify is the open version: a benchmark that can measure the claim, an
engine that gives any Hugging Face checkpoint the same interface, and a training recipe that
optimizes the same objective.

![Jev 1.13.0 on jev-bench: model probability vs human vote share](reports/jev-1.13.0/figures/human_vs_model.png)

## What's here

**jev-bench** — 22 configs, 166k rows of real human-labeled data reformatted into System One
questions in Jev's exact wire format, with **human label distributions** on four configs
(ChaosNLI, Civil Comments, Measuring Hate Speech, GoEmotions) so calibration is measured against
how humans actually split. Natural base rates everywhere. → `jevify/bench`, `jevify-bench build`.

**Metrics and figures** — ECE, Brier, NLL, RPS, selective accuracy, AURC, AUROC, TVD/KL to human
distributions; reliability diagrams, calibration maps with bootstrap CIs, risk–coverage curves,
model-vs-human plots. → `jevify/bench/metrics.py`, `jevify/bench/figures.py`.

**Behavioral probes** — within-item experiments that isolate one factor: decision-set size (same
items, gold kept, K distractors), option order, opaque option keys, distractor injection, and the
same question asked through a different primitive. → `jevify/bench/probes.py`.

**The engine (Tier 0)** — render `state` + question, score every allowed answer by teacher-forcing
against a shared prefix (KV cache expanded across candidates, so 151-option Choice is one prefix
pass), then calibrate. Raw log-scores are stored on every prediction, so temperature scaling,
permutation averaging and contextual-prior correction are offline CPU experiments; nothing is
tuned on test. → `jevify/engine`.

**A drop-in server** — `jevify-serve --model <hf-id>` exposes `/v1/systemone` and `/v1/models`.
The official `typesafe-sdk` works against it by setting `TYPESAFE_BASE_URL`; that is a test in
this repo. → `jevify/server.py`.

**A Jev runner** — score the real API on the same records, same metrics, same figures.
→ `jevify-run api|report|recipe|compare`.

## What we know about Jev so far

From 22,773 test records plus a manual audit of its errors
([full report](reports/jev-1.13.0/README.md)):

- **Crisp, grounded decisions are excellent and calibrated** — ARC 97.9%, MMLU 92.3%,
  FEVER-with-evidence 97.2%, BoolQ 91.7%, with ECE ≤ 0.06. As a router or guard on
  well-specified questions, the confidence is usable as advertised.
- **Where humans disagree, the probabilities do not track human uncertainty** — ChaosNLI TVD 0.33
  with p=0.94–0.98 answers on items where 100 annotators split 60/40; hate-speech vote shares
  TVD 0.43; ~0.29 "toxic" when zero Jigsaw raters flagged the comment. "Calibrated" fails on
  exactly the inputs where it matters.
- **The System Two gap is measurable** — 78.5% closed-book vs 95.6% grounded on the same
  StrategyQA questions.
- **Decision-set size is a cost, not a cliff** — within-item, with the gold answer always present,
  clinc150 goes 99.5% → 91% from K=2 to K=151 with ECE ≤ 0.05, while GoEmotions is 85% at K=2 and
  30% by K=25: ambiguity does the damage, not cardinality.
- **The primitive is an instrument, not a skin** — the same yes/no question is twice as well
  calibrated as Noul than as a 2-way Choice; Score beats an unordered Choice over the same levels.
  Option order flips 0–13% of answers (scaling with ambiguity); opaque option keys cost nothing if
  descriptions remain; nonsense options attract ≤3% of the mass.
  → [behavioral probes](reports/jev-1.13.0/probes/README.md)

## Roadmap

- [x] jev-bench v0.1.1, Jev 1.13.0 baseline, figures, label audit, behavioral probes
- [x] Tier 0 engine + server + offline recipe search (tests pass on CPU with a 135M model)
- [ ] Tier 0 sweep on Modal: K2-Horizon-0.9B, Qwen3.5 0.8B–9B (base and instruct), Gemma 4 E2B/E4B/12B,
      SmolLM3-3B, Olmo-3-7B, Apertus-4B — scored against Jev on the same records, latency measured
      from the same vantage point
- [ ] Tier 1: frozen backbone + decision heads trained with proper scoring rules on cached features
- [ ] Tier 2: LoRA where Tier 1 leaves a gap the benchmark can see
- [ ] Label-first synthetic data pipeline; HF Space; model zoo; VLM backbones; vision-tower autoresearch

## Why tiers, and why no RL

Calibration comes from optimizing a strictly proper scoring rule against real outcomes. TypeSafe
calls their version RLCD. When the model *emits* a decision you need RL to push that objective
through a sampling step; when the decision is read from a differentiable head on the backbone's
hidden states, the same objective is just a loss. Jevify trains heads (and optionally LoRA) by
gradient descent on log/Brier/ranked-probability losses — no reward model, no rollouts — and each
tier has to earn its place on jev-bench.

## Quickstart

```bash
pip install -e ".[bench,engine,serve,dev]"

# the benchmark
jevify-bench list
jevify-bench build --out data/jev-bench                     # or load Praveenrajus/jev-bench from the Hub

# score Jev (or any /v1/systemone server) on it
TYPESAFE_API_KEY=... jevify-run api --records data/jev-bench --out preds/jev.jsonl
jevify-run report --records data/jev-bench --preds preds/jev.jsonl --md report.md --figures figures/

# Jevify a checkpoint and serve it
jevify-serve --model Qwen/Qwen3.5-0.8B --port 8000
TYPESAFE_BASE_URL=http://localhost:8000 python -c "from typesafe_sdk import *; print(TypeSafeClient().system_one('Refund please', {'r': Noul(instructions='Is this a refund request?')}))"

# behavioral probes
jevify-bench probe --records data/jev-bench --out data/jev-probes --n 200
jevify-run api --records data/jev-probes --out preds/probes.jsonl
jevify-bench probe-report --records data/jev-probes --preds preds/probes.jsonl --out results/probes
```

Tests: `pytest` (engine and server tests download a 135M model and run on CPU).

## License

Apache-2.0 for the code. Benchmark rows carry their upstream licenses (see the manifest and
[docs/DATASETS.md](docs/DATASETS.md)).
