# Retraining the 4B Tier 2 model the Tev1 way: Together's data and shuffled options

Two things Together AI's Tev1 recipe does that Jevify's Tier 2 did not: it trains on 23,340
code-generated decision examples (rule policies whose answer can be "insufficient information", ordered
routing rules, fictional research summaries), and it presents every example's options in a random order.
Both were added to the published Qwen3.5-4B Tier 2 recipe (LoRA lr 3e-5, two epochs, seed 0), alone and
together, and the unchanged recipe was re-run on the same hardware as the reference:

- **Together's data**: [`scripts/tev1_synthetic.py`](../../scripts/tev1_synthetic.py) runs Together's own
  generators (MIT, pinned commit) and adds four training-only sources beside jev-bench -- 1,591 training
  and 387 validation examples, capped per source like every jev-bench source. Tev1's public-data half is
  left out: four of its five datasets are jev-bench sources Jevify already trains on.
- **Shuffled options**: `python -m jevify.train tier2 --shuffle-options` -- every training batch sees
  each Choice record's options in a fresh random order (Score levels keep theirs; validation does not
  shuffle).

Each run was then held to the same checks: jev-bench, with the six held-out sources on their own; the
community benchmarks as the guardrail (the phishing collapse of Section 12); and the three new tests of
[`../b/`](../b/README.md).

| Qwen3.5-4B Tier 2 | all 22 | 6 held-out | 16 trained | TVD→human | phishing recall | stated rule | "none" when the answer is gone |
|---|---|---|---|---|---|---|---|
| published (A40) | 0.734 / 0.096 | 0.765 / 0.098 | 0.722 / 0.095 | 0.337 | 0.118 | 0.653 | 0.658 |
| **re-run (same hardware as the rest)** | 0.736 / 0.096 | 0.763 / 0.102 | 0.725 / 0.094 | 0.335 | 0.040 | 0.667 | 0.540 |
| + shuffled options | 0.726 / 0.094 | **0.748** / 0.107 | 0.718 / 0.090 | 0.334 | 0.033 | 0.697 | 0.446 |
| + Together's data | 0.735 / **0.089** | 0.760 / 0.102 | 0.726 / **0.085** | 0.329 | 0.048 | 0.652 | 0.570 |
| + both | 0.739 / 0.092 | 0.760 / 0.104 | 0.732 / 0.088 | **0.324** | 0.049 | 0.649 | 0.516 |
| *untrained base (Tier 0)* | *0.662 / 0.090* | *0.719 / 0.102* | | *0.438* | *0.436* | *0.618* | *0.482* |

*accuracy / ECE. Phishing recall: the share of the 1,000 phishing emails called phishing (PhishNChips
verdict question).*

- **Neither change helps where it was meant to.** On the six held-out sources, shuffling costs 0.015
  and Together's data 0.003; together, 0.003. Overall accuracy moves by at most 0.010.
- **Together's data buys a little calibration and human agreement** -- ECE 0.096 → 0.089, TVD 0.335 →
  0.324 with both changes -- the one consistent effect.
- **It does not repair the phishing collapse.** Every Tier 2 run flags almost no phishing (recall
  3–5%, against 44% for the untrained base), with or without Together's data -- as Tev1 itself does.
- **Its rule data does not transfer to rules written by lawyers** (stated rule 0.652 against 0.667),
  the same result as for Tev1 (Section 14).
- **Behaviour outside jev-bench is not stable across runs of the same recipe.** The re-run matches the
  published model on jev-bench to 0.002, yet its phishing recall is 0.040 against 0.118 and its "none"
  rate 0.540 against 0.658. One run per arm cannot rank these arms on phishing or on the new tests; only
  the jev-bench columns are reliable here.

Tool-call risk and ticket routing (60 and 27 items) do not separate any of these runs. Everything is in
[`jev-bench/comparison.md`](jev-bench/comparison.md), [`community/tables.md`](community/tables.md),
[`b/tables.md`](b/tables.md), and each run's `run.json`, `test_metrics.json` and `test_report.md` under
[`runs/`](runs/); predictions are on the Hub.
