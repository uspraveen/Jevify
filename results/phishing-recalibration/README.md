# The phishing collapse is a threshold, and a handful of labelled emails repairs it

Every fine-tuned model in this project, and Together's Tev1, calls almost every PhishNChips email
legitimate ([Section 12](../../docs/FINDINGS.md#12-three-benchmarks-we-did-not-write), [`../c/`](../c/README.md)),
while its ranking of the same emails stays about as good as its untrained base's. That points at a
moved threshold rather than lost knowledge, and a deployment would repair a threshold the obvious way:
label a few of its own emails and shift the model's log-odds for "phishing" by one number.

[`scripts/phishing_recalibration.py`](../../scripts/phishing_recalibration.py) measures exactly that:
draw *n* labelled emails (half phishing, half legitimate), fit one additive shift on the model's
phishing log-odds by maximum likelihood, and score every **other** email; 200 random draws per size.
Nothing else about any model changes.

The verdict question (the one Together scored), 2,000 emails:

| model | ranking (AUROC) | as served: accuracy / recall | 16 labelled | 64 labelled | recall with 16 labels (5–95% of draws) |
|---|---|---|---|---|---|
| Jev 1.13.0 | 0.688 | 0.628 / 0.436 | 0.613 | 0.613 | 0.45–0.73 |
| Qwen3.5-4B, Tier 0 | 0.784 | 0.700 / 0.436 | 0.717 | 0.721 | 0.52–0.75 |
| Gemma-4-12B, Tier 0 | 0.906 | **0.825** / 0.717 | **0.825** | **0.832** | 0.66–0.84 |
| Tev1-4B + recipe | 0.848 | 0.511 / 0.022 | 0.761 | 0.765 | 0.61–0.80 |
| Jevify 4B Tier 2 (published) | 0.782 | 0.557 / 0.118 | 0.701 | 0.704 | 0.49–0.76 |
| Jevify 4B Tier 2 (re-run) | 0.746 | 0.520 / 0.040 | 0.665 | 0.665 | 0.47–0.72 |

- **Sixteen labelled emails take Tev1 from 0.511 to 0.761 and the Jevify Tier 2 model from 0.557 to
  0.701** -- from coin-flip to useful -- on emails the shift never saw. Their recall goes from 2% and
  12% to about 70% and 62%. The collapse was a threshold.
- **It is not free.** With 16 labels, which 16 matters: the recall a deployment ends up with spans about
  0.5 to 0.8 across draws. With 64 it narrows by more than half, and more labels than that buy little -- the
  fitted shift is then as good as one fitted on all 2,000 ("best single shift" in the tables).
- **It does not rank models differently.** A shift cannot improve the ranking, so the ceiling is set by
  AUROC: the untrained Gemma-4-12B ranks best and stays best (0.83), and Jev -- the weakest ranker here,
  with probabilities the API rounds to 0.01 -- gains nothing.

The other three phishing wordings (the same question as a Noul, "should the user click?", "classify
this email") are in [`tables.md`](tables.md), with 16, 32, 64 and 128 labels; every number is in
[`report.json`](report.json). The same pattern holds, weaker where the ranking itself is weak (the
Noul wording, AUROC 0.55 for the Tier 2 models). [`models.json`](models.json) lists the prediction
files on the Hub the study reads.
