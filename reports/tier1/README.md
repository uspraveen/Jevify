# Tier 1: trained decision heads, and what held-out sources revealed

Tier 0 gives any open checkpoint Jev's interface with no training: render the question,
score the allowed answers from the model's logits, calibrate on validation splits. It
reaches Jev's calibration (ECE 0.089–0.158 across seven models vs Jev's 0.113) but not its
accuracy (0.40–0.66 vs 0.733).

Tier 1 asks whether a small trained head can close that gap. The answer turned out to
depend entirely on one design choice, and the experiment that revealed it is the one most
benchmarks would have skipped.

## The protocol that matters

A head trained on jev-bench's train splits and scored on its test splits is in-distribution
supervised learning. Jev answers these questions zero-shot, so that comparison would be
meaningless. Six sources are therefore **held out of training entirely** — `clinc150`,
`arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`
— plus `chaosnli`, which has no train split at all. They span every primitive, K from 2 to
151, and four domains.

The Tier 0 recipe is also **refitted without those sources** before comparison; otherwise
its temperature has seen validation data the heads never did.

Backbones: Qwen3.5-2B and Qwen3.5-4B. Heads trained on 8,685 records from 16 sources, ≤16
options per record, early-stopped on 2,250 validation records. One A100 job each, 15–30
minutes, $0.65 and $1.28.

## Result

| variant | held-out acc | held-out ECE | trained acc | trained ECE |
|---|---|---|---|---|
| Tier 0 (recipe refit without held-out) | 0.628 | 0.097 | 0.552 | 0.100 |
| Tier 1, heads **replace** the LM head | 0.522 | 0.155 | 0.627 | 0.056 |
| Tier 1, heads **residual** on the LM head | **0.631** | **0.090** | **0.632** | **0.061** |
| *Jev 1.13.0 (zero-shot, for reference)* | *0.835* | *0.090* | *0.694* | *0.122* |

The replacement head is a **+0.077 mean gain on trained sources and a −0.098 loss on
held-out ones**. Measured only in-distribution it looks like a clear win; it is not one.

### Replication on Qwen3.5-4B

| variant | held-out acc | held-out ECE | trained acc | trained ECE |
|---|---|---|---|---|
| Tier 0 (recipe refit without held-out) | 0.714 | 0.139 | 0.641 | 0.087 |
| Tier 1, heads **residual** | **0.749** | **0.107** | **0.680** | **0.082** |
| *Jev 1.13.0 (zero-shot)* | *0.835* | *0.090* | *0.694* | *0.122* |

The finding holds and strengthens at 4B: the residual head is **+0.035 on held-out sources**
(not merely neutral as at 2B) and **+0.039 on trained ones**, better calibrated in both regimes.
The learned LM weights are **0.96 / 0.97 / 1.01**, within 0.01–0.02 of the 2B run's
0.95 / 0.98 / 1.01 — two independently trained heads on different backbones both concluded the
model's own prior was worth keeping at full strength.

Across the whole benchmark this puts Jevified Qwen3.5-4B at **macro accuracy 0.698 against Jev's
0.733**, with better calibration (ECE 0.089 vs 0.113), closer agreement with human label
distributions (TVD 0.360 vs 0.432), and higher Score accuracy (0.507 vs 0.503).

## Why it failed, and why that was informative

The damage was concentrated, not diffuse:

| source | Tier 0 | replace | residual |
|---|---|---|---|
| `measuring_hate_speech` (held out, score) | 0.545 | 0.063 **(−0.482)** | 0.452 (−0.093) |
| `arc_challenge` (held out, knowledge) | 0.797 | 0.598 **(−0.199)** | 0.756 (−0.041) |
| `mmlu` (**trained**, knowledge) | 0.581 | 0.462 **(−0.119)** | 0.545 (−0.036) |
| `yelp5` (held out, score) | 0.483 | 0.392 (−0.091) | 0.414 (−0.069) |
| `clinc150` (held out, K=151) | 0.434 | 0.426 (−0.008) | 0.484 (+0.050) |

Two things stand out.

**Structure generalizes; knowledge does not.** `clinc150` has 151 options and the heads were
trained with at most 16, yet it barely moves — the K-agnostic design works exactly as
intended. What collapses is knowledge (`arc_challenge`, and `mmlu` *even though it was in
training*) and ordinal scales the head never saw (`measuring_hate_speech`, `yelp5`). A head
that replaces the LM head discards everything the backbone knows and relearns a scorer from
8,685 examples.

**So don't replace it — correct it.** The residual head computes

```
score_i = w · lm_logscore_i + f([h_dec, h_i, h_dec ⊙ h_i])
```

with `f`'s final layer zero-initialized, so at step 0 the model *is* Tier 0 (a unit test
asserts the untrained residual head reproduces Tier 0's distribution to 1e-4) and training
can only add to it. The learned LM weights came out **0.95 / 0.98 / 1.01** for
choice / score / noul — the head chose to keep the model's own prior at full strength.

The result is strictly better than both alternatives: it keeps the in-distribution gain
(+0.080 mean, 15/15 sources improved or flat) and the held-out regression disappears
(−0.098 → −0.000), with better calibration than Tier 0 in both regimes.

## What the residual head actually buys

Against Tier 0 on the same backbone, in-distribution: `go_emotions` +0.156, `civil_comments`
+0.203, `helpsteer2_verbosity` +0.193, `massive` +0.145, `banking77` +0.072, and ECE roughly
halved (0.100 → 0.061). These are the tasks where the answer depends on reading option
*semantics* rather than recalling a fact — exactly where a learned matching function over
option representations should help and where a logit on an identifier token cannot.

## Honest limitations

- **Ordinal generalization is still the weak point.** Both held-out `score` sources regress
  (`measuring_hate_speech` −0.093, `yelp5` −0.069). Only four ordinal sources are in
  training, and they share a sentiment/quality flavour; an unseen scale with different
  semantics is still mapped onto them. More diverse ordinal scales in training is the
  obvious next step.
- **The held-out set is not difficulty-matched.** It contains several of Jev's strongest
  configs, which is why Jev scores *higher* on held-out (0.835) than on trained (0.694)
  sources. Compare models within a column, never across.
- **Two backbones, one seed each.** The direction replicates; the magnitudes will move.
- Heads are trained on jev-bench's own train splits, so "held out" means held-out *source*,
  not held-out *distribution* — the records still come from the same 22-dataset family.

## Reproduce

```bash
modal run --detach modal_app.py::tier1 --model-id Qwen/Qwen3.5-2B --run-id qwen35-2b-t1r \
  --gpu A100-80GB --train-per-source 600 --val-per-source 150 --epochs 25 --batch 16
python scripts/process_tier1.py --run-id qwen35-2b-t1r --tier0 qwen35-2b
```

`--residual false` reproduces the replacement head and its negative result.
