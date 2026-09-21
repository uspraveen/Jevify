# Qwen3.5-2B (Tier 1 residual, seed 0): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 6.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B (Tier 1 residual, seed 0) | Tier 1 heads | 0.626 | 0.097 | 0.464 | 0.630 | 0.058 | 0.437 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen3.5-2B (Tier 1 residual, seed 0)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.756 | 0.038 | 0.331 | 0.62 | 0.794 | 0.936 | 0.084 |  |  |  |  |
| `banking77` | choice | 1000 | 0.608 | 0.079 | 0.534 | 1.53 | 0.659 | 0.838 | 0.189 |  |  |  |  |
| `boolq` | noul | 1000 | 0.849 | 0.029 | 0.111 | 0.36 | 0.886 | 0.964 | 0.056 |  |  | 0.916 |  |
| `chaosnli` | choice | 1599 | 0.562 | 0.086 | 0.547 | 0.89 | 0.582 | 0.652 | 0.335 |  |  |  | 0.270 |
| `civil_comments` | noul | 2000 | 0.921 | 0.030 | 0.067 | 0.25 | 0.945 | 0.971 | 0.033 |  |  | 0.754 | 0.102 |
| `clinc150` | choice | 1000 | 0.487 | 0.090 | 0.717 | 3.34 | 0.521 | 0.660 | 0.312 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.889 | 0.028 | 0.087 | 0.30 | 0.918 | 0.958 | 0.041 |  |  | 0.962 |  |
| `go_emotions` | choice | 1000 | 0.403 | 0.064 | 0.759 | 2.06 | 0.428 | 0.536 | 0.441 |  |  |  | 0.624 |
| `helpsteer2_helpfulness` | score | 1000 | 0.378 | 0.055 | 0.712 | 1.38 | 0.386 | 0.456 | 0.548 | 0.160 | 0.98 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.629 | 0.074 | 0.542 | 1.04 | 0.656 | 0.680 | 0.311 | 0.088 | 0.52 |  |  |
| `ledgar` | choice | 1000 | 0.684 | 0.075 | 0.445 | 1.19 | 0.722 | 0.896 | 0.142 |  |  |  |  |
| `massive` | choice | 1000 | 0.567 | 0.081 | 0.569 | 1.60 | 0.620 | 0.798 | 0.202 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.437 | 0.136 | 0.736 | 1.38 | 0.446 | 0.488 | 0.509 | 0.304 | 0.92 |  | 0.513 |
| `mmlu` | choice | 1000 | 0.546 | 0.100 | 0.571 | 1.09 | 0.578 | 0.740 | 0.246 |  |  |  |  |
| `mnli` | choice | 1000 | 0.742 | 0.029 | 0.353 | 0.60 | 0.769 | 0.878 | 0.114 |  |  |  |  |
| `paws` | noul | 1000 | 0.831 | 0.029 | 0.123 | 0.39 | 0.856 | 0.950 | 0.063 |  |  | 0.912 |  |
| `sms_spam` | noul | 800 | 0.963 | 0.016 | 0.031 | 0.12 | 0.988 | 0.995 | 0.012 |  |  | 0.971 |  |
| `sst5` | score | 1000 | 0.416 | 0.095 | 0.696 | 1.31 | 0.424 | 0.454 | 0.534 | 0.122 | 0.74 |  |  |
| `strategyqa_closed` | noul | 687 | 0.594 | 0.037 | 0.235 | 0.66 | 0.615 | 0.651 | 0.332 |  |  | 0.639 |  |
| `strategyqa_grounded` | noul | 687 | 0.785 | 0.077 | 0.151 | 0.47 | 0.812 | 0.933 | 0.091 |  |  | 0.879 |  |
| `stsb` | score | 1000 | 0.386 | 0.052 | 0.707 | 1.41 | 0.409 | 0.516 | 0.485 | 0.110 | 0.82 |  |  |
| `yelp5` | score | 1000 | 0.405 | 0.209 | 0.763 | 1.61 | 0.392 | 0.454 | 0.497 | 0.154 | 0.89 |  |  |
