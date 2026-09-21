<div align="center">

# Jevify

**Turn any open LLM into a calibrated, Jev-style System One decision model.**

[**Try it**](https://uspraveenraj--jevify-playground.modal.run) ·
[**Findings**](docs/FINDINGS.md) ·
[jev-bench on the Hub](https://huggingface.co/datasets/Praveenrajus/jev-bench) ·
[Jev baseline + label audit](results/jev-1.13.0/README.md) ·
[Behavioral probes](results/jev-1.13.0/probes/README.md) ·
[Tier 1 write-up](reports/tier1/README.md) ·
[Jev API contract](docs/JEV_CONTRACT.md) ·
[Dataset rationale](docs/DATASETS.md)

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

## Headline findings

Full catalogue with evidence and caveats in **[docs/FINDINGS.md](docs/FINDINGS.md)**.

**1 · Jev is calibrated right up until humans disagree — which is when calibration matters.**
On ChaosNLI, where 100 annotators label every item, Jev's confidence is essentially **flat no
matter how much the annotators agree** — 0.81 / 0.84 / 0.83 / 0.88 as agreement goes from a
split (< 50%, n = 191) through contested (50–70%, n = 828) and clear (70–90%, n = 523) to
consensus (≥ 90%, n = 57), Pearson r = 0.046 — while its accuracy on those same bands runs
**0.47 / 0.55 / 0.73 / 0.96**. It is more confident than the human majority on 80% of items; its
distributions sit **TVD 0.33** from the human ones, and 0.43 on hate-speech vote shares.
Meanwhile it is excellent and well calibrated on crisp, grounded questions (ARC 0.979, FEVER
0.972, ECE ≤ 0.06).

![Jev: confidence vs human agreement on ChaosNLI](results/figures/confidence_vs_agreement.png)

**2 · Evaluating on held-out *records* instead of held-out *sources* would have shipped the
wrong architecture — and evaluating on one seed nearly shipped an overclaim.** A trained head that
replaces the model's own scorer gains +0.077 accuracy on sources it trained on and loses **−0.098**
on sources it never saw. Making it a zero-initialized residual on that scorer — so training provably
starts at Tier 0 — keeps the gain (+0.080). The first version of this README said it also "erased
the regression (−0.000)". **Five seeds say otherwise:** held-out accuracy is **0.579 ± 0.049** —
two seeds match Tier 0 (0.628), three regress by ~0.09 and turn overconfident on unseen sources —
while trained-source accuracy is stable at 0.635 ± 0.007. The residual head halves the damage of
replacement on average; it does not remove it, and the published run was the top of its own
distribution. The learned LM weights (**0.95 / 0.98 / 1.01** at 2B, **0.96 / 0.97 / 1.01** at 4B)
still say the model's prior is worth keeping at full strength.

![Tier 1: trained vs held-out sources](results/figures/tier1_story.png)

**3 · An open 4B now beats Jev on the overall number — and Jev still wins where it counts most.**
Qwen3.5-4B with LoRA and residual heads (Tier 2): macro accuracy **0.747 vs Jev's 0.733**, macro
ECE 0.110 vs 0.113, Noul accuracy 0.903 vs 0.881, Score accuracy 0.529 vs 0.503, TVD to human
label distributions 0.347 vs 0.432. The qualifier is the split: on the six sources it never
trained on, Jev leads **0.835 to 0.769**; the open model's overall edge comes from the sixteen it
did train on. One seed. Without touching the backbone, Qwen3.5-4B with residual heads alone
reaches 0.698 / ECE 0.089 / TVD 0.360; the 2B head's ECE of 0.069 is the best of anything tested
but, per (2), seed-dependent on unseen sources.

**4 · Structure generalizes; knowledge does not.** Heads trained with ≤16 options transfer
unchanged to K=151 (clinc150 moves −0.008). What collapses under replacement is knowledge
(arc_challenge −0.199, mmlu −0.119 *even when trained*) and ordinal scales never seen
(measuring_hate_speech −0.482).

**5 · LoRA buys the accuracy a frozen backbone cannot, and at 4B it buys calibration too.**
Tier 2 (rank-16 LoRA trained jointly with the residual heads, one A40) lifts Qwen3.5-2B from
0.632 to **0.690** macro accuracy and Qwen3.5-4B from 0.698 to **0.747**, and fixes the Tier 1 weak
spot outright: the held-out ordinal scale gains **+0.17** accuracy *and* gets better calibrated.
At 2B the gain came with Jev-like overconfidence on ambiguous questions (ChaosNLI ECE 0.077 →
0.215; Jev: 0.222); a lower LoRA learning rate (3e-5) gave more accuracy *and* held-out ECE 0.086,
and training on human label distributions instead did not help (TVD 0.433, Jev's number). But
those three arms are one seed each and sit within the ±0.05 the seed study found, so which arm is
"best" is unresolved until their seeds finish. What holds regardless: at 4B the LoRA improves
held-out accuracy (0.714 → 0.769) and held-out ECE (0.139 → 0.107) at once, the 4B does not
overfit after one pass the way the 2B does (its best epoch is the second), and the 2B arms all
overfit after one. [· detail](docs/FINDINGS.md#7-tier-2-letting-the-backbone-move)

**6 · Instruction tuning does hurt calibration — 1.3–1.8× worse raw ECE — but it is almost
entirely a temperature problem.** Gemma-4-E2B-it starts at ECE 0.361 and lands at 0.158 after one
scalar per primitive; its accuracy is +0.195 over its base checkpoint for +0.043 ECE. Take the
instruct checkpoint and always fit the temperature.
[· detail](docs/FINDINGS.md#5-does-instruction-tuning-hurt-calibration)

**7 · What hurts Jev is ambiguity, not option count.** Controlled within-item probes: with the
gold answer always present, clinc150 goes 0.995 → 0.910 from K=2 to K=151, while GoEmotions is
0.850 at K=2 and 0.300 by K=25. Option order flips 0–13% of answers, scaling with ambiguity
rather than K; opaque option keys cost nothing as long as descriptions remain; nonsense options
attract ≤3.3% of the mass.

**8 · The primitives are different instruments.** The same yes/no question is ~2× better
calibrated asked as a Noul than as a two-option Choice (ECE 0.028 vs 0.054); Score beats an
unordered Choice over the same levels. This is why Tier 1 gives Noul its own absolute head
instead of a softmax over {yes, no}.

**9 · Two of our own benchmark bugs, found by reading the model's errors.** HelpSteer2 verbosity
levels contradicted NVIDIA's verbatim scale, and GoEmotions' single-label subset hid rater
disagreement (rebuilt from raw votes; plurality agreement is only 0.66, which is the accuracy
ceiling). Low benchmark scores deserve an audit before they become claims.

**10 · It transfers to vision — and so does the ordering of the primitives.** Qwen3-VL-2B, no
training, on POPE / A-OKVQA / AI2D (4,244 records): Noul **ECE 0.083** vs Choice 0.124 and 0.175,
the same ~2× gap found in text with a different model, modality and data. Starving the vision
encoder then costs accuracy *and* calibration together: from 42 to 187 image tokens, accuracy
rises 0.048 and ECE falls 0.037. The dangerous regime — losing accuracy while keeping confidence
— never appears; on the hallucination benchmark, a starved encoder produces doubt, not confident
hallucination. And the budget saturates silently: 512 and 1024 patches both resolve to 280
tokens, which only the *measured* token count reveals.
[· detail](docs/FINDINGS.md#9-vision-does-any-of-this-transfer)

![What the vision encoder's budget buys](results/vision-budget/vision_budget.png)

## Try it

**Playground:** [https://uspraveenraj--jevify-playground.modal.run](https://uspraveenraj--jevify-playground.modal.run) — ask a Jevified open model typed questions and watch the
distributions. Scales to zero, so the first question waits ~30 s for a cold start.

**Hosted API**, same wire format as TypeSafe's:

```bash
curl -X POST https://uspraveenraj--jevify-playground.modal.run/api/v1/systemone   -H 'Content-Type: application/json'   -d '{"state": "I was charged twice, please refund.", "model": "jevify-latest",
       "questions": {"refund": {"type": "noul", "instructions": "Is this a refund request?"}}}'
```

The official `typesafe-sdk` works against it unchanged:

```python
# TYPESAFE_BASE_URL=https://uspraveenraj--jevify-playground.modal.run/api
from typesafe_sdk import Noul, TypeSafeClient
with TypeSafeClient() as client:
    r = client.system_one(state="I was charged twice, please refund.", model="jevify-latest",
                          questions={"refund": Noul(instructions="Is this a refund request?")})
```

**Models:** [jevify-qwen3.5-4b](https://huggingface.co/Praveenrajus/jevify-qwen3.5-4b) ·
[jevify-qwen3.5-2b](https://huggingface.co/Praveenrajus/jevify-qwen3.5-2b) — ~11 MB each; the
backbone is pulled from its own repo, so nothing is duplicated.

```python
from jevify import load_jevified
model = load_jevified("Praveenrajus/jevify-qwen3.5-4b")
model.ask(state, questions)
```

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

**Training that runs anywhere** — Tier 1, Tier 2 and the vision pass with no cloud dependency:
`python -m jevify.train tier1|tier2|vision --model-id … --run-id …`. The Modal app calls the same
functions. → `jevify/train.py`.

**Vision** — images ride on `state`; the question and answer set are unchanged. The vision tower is
addressable, not opaque: `describe()` it, set a pixel budget and *measure* the tokens it actually
produced, freeze it, or scope LoRA to it with a full-path regex (suffix names like `q_proj` exist on
both halves of a VLM, so a suffix list silently adapts everything). Tower swapping is deliberately
not offered — the projector is trained against one encoder's geometry. → `jevify/engine/vision.py`,
`jevify/engine/vision_backbone.py`, `scripts/vision_budget.py`.

## Leaderboard

Every model on the same 22,773 test records. Tier 0 = no training (prompt + logit readout + a
recipe fitted on validation splits only). Tier 1 = trained decision heads, six sources held out
of training. **TVD→human** is the mean distance to human label distributions on the four
calibration-gold configs — lower is better, and it is the number Jev's own claim rests on.

| model | tier | macro acc | macro ECE | macro Brier | sel@90 | choice acc | score acc | noul acc | TVD→human | GPU | test cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Jev 1.13.0 (TypeSafe API)** | API | 0.733 | 0.113 | 0.349 | 0.760 | 0.770 | 0.503 | 0.881 | 0.432 |  |  |
| Qwen/Qwen3.5-4B (Tier 1 residual) | Tier 1 residual | 0.698 | 0.089 | 0.378 | 0.729 | 0.699 | 0.507 | 0.862 | 0.360 | A100-80GB | $1.28 |
| Qwen/Qwen3.5-4B | Tier 0 | 0.662 | 0.093 | 0.402 | 0.689 | 0.687 | 0.468 | 0.796 | 0.438 | A100-80GB | $0.96 |
| google/gemma-4-E4B-it | Tier 0 | 0.658 | 0.148 | 0.426 | 0.678 | 0.699 | 0.432 | 0.798 | 0.432 | A100-80GB | $1.00 |
| Qwen/Qwen3.5-2B (Tier 1 residual) | Tier 1 residual | 0.632 | 0.069 | 0.445 | 0.657 | 0.596 | 0.449 | 0.835 | 0.374 | A100-80GB | $0.67 |
| Qwen/Qwen3.5-2B (Tier 1 replace) | Tier 1 replace | 0.599 | 0.083 | 0.475 | 0.621 | 0.567 | 0.378 | 0.828 | 0.416 | A100-80GB | $0.62 |
| google/gemma-4-E2B-it | Tier 0 | 0.591 | 0.158 | 0.502 | 0.609 | 0.595 | 0.440 | 0.716 | 0.490 | L4 | $0.75 |
| Qwen/Qwen3.5-2B | Tier 0 | 0.577 | 0.089 | 0.485 | 0.599 | 0.545 | 0.424 | 0.750 | 0.458 | A100-80GB | $0.66 |
| HuggingFaceTB/SmolLM3-3B | Tier 0 | 0.552 | 0.111 | 0.519 | 0.570 | 0.504 | 0.401 | 0.744 | 0.458 | A100-80GB | $0.64 |
| Qwen/Qwen3.5-0.8B | Tier 0 | 0.526 | 0.113 | 0.543 | 0.544 | 0.435 | 0.388 | 0.759 | 0.440 | L4 | $0.55 |
| Qwen/Qwen3.5-0.8B-Base | Tier 0 | 0.466 | 0.105 | 0.582 | 0.478 | 0.326 | 0.412 | 0.692 | 0.452 | L4 | $0.50 |
| IFM/K2-Horizon-0.9B | Tier 0 | 0.448 | 0.119 | 0.589 | 0.461 | 0.344 | 0.343 | 0.670 | 0.479 | L4 | $0.35 |
| google/gemma-4-E2B | Tier 0 | 0.396 | 0.115 | 0.609 | 0.404 | 0.247 | 0.382 | 0.600 | 0.496 | L4 | $0.69 |

![models](results/leaderboard/models_map.png)

Per config, for every model at once — Jev's Score row is the dark one:

![ECE per config, every model](results/leaderboard/heatmap_ece.png)

Refresh with `python scripts/leaderboard.py`; every row has `results/<run>/` with its predictions,
per-config metrics, recipe and figures.

## Roadmap

- [x] jev-bench v0.1.1, Jev 1.13.0 baseline, figures, label audit, behavioral probes
- [x] Tier 0 engine + server + offline recipe search
- [x] Tier 0 sweep: 9 open checkpoints scored on all 22,773 test records against Jev
- [x] Tier 1 residual decision heads, with held-out-source generalization measured
- [x] Findings, figures and reports published ([docs/FINDINGS.md](docs/FINDINGS.md))
- [x] Published models + hosted playground and API
- [x] Tier 1 replicated on a second backbone (Qwen3.5-4B)
- [x] Training portable off Modal (`python -m jevify.train`)
- [x] VLM backbones: Tier 0 on POPE / A-OKVQA / AI2D; vision tower inspectable, budgetable, freezable, LoRA-scopable
- [x] Vision-encoder budget sweep: accuracy and calibration degrade together, budget saturation measured
- [x] Tier 2: LoRA jointly with residual heads — +0.058 macro accuracy at 2B; at lr 3e-5 the best held-out ECE and human-agreement of any model; overfits after one pass
- [x] Tier 2 ablation: learning rate vs hard-label loss — the learning rate was the cause of the overconfidence; soft-label training is a negative result on human agreement
- [x] Five-seed error bars on Tier 1: held-out generalization is seed-dependent (0.579 ± 0.049); README corrected
- [x] Tier 2 at 4B: macro 0.747, above Jev's 0.733; Jev still leads on held-out sources
- [ ] Seeds on the Tier 2 arms and the 4B head (running); soft labels at the low learning rate
- [ ] Vision-scoped LoRA on AI2D (the perceptual weak spot); more ordinal scales; Tier 1 at 7B+
- [ ] Label-first synthetic data pipeline; HF Space; model zoo

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
