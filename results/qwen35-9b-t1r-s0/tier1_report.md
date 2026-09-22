# Qwen/Qwen3.5-9B (Tier 1): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 5.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B (Tier 1) | Tier 1 heads | 0.744 | 0.084 | 0.328 | 0.703 | 0.068 | 0.375 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-9B | Tier 0 (recipe refit w/o held-out) | 0.729 | 0.135 | 0.335 | 0.670 | 0.092 | 0.401 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-9B (Tier 1)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.884 | 0.028 | 0.169 | 0.33 | 0.931 | 0.996 | 0.022 |  |  |  |  |
| `banking77` | choice | 1000 | 0.723 | 0.130 | 0.421 | 1.27 | 0.764 | 0.908 | 0.106 |  |  |  |  |
| `boolq` | noul | 1000 | 0.900 | 0.024 | 0.072 | 0.24 | 0.938 | 0.986 | 0.022 |  |  | 0.964 |  |
| `chaosnli` | choice | 1599 | 0.606 | 0.177 | 0.539 | 0.88 | 0.621 | 0.721 | 0.257 |  |  |  | 0.303 |
| `civil_comments` | noul | 2000 | 0.926 | 0.017 | 0.061 | 0.22 | 0.948 | 0.983 | 0.022 |  |  | 0.815 | 0.096 |
| `clinc150` | choice | 1000 | 0.765 | 0.083 | 0.360 | 1.06 | 0.801 | 0.912 | 0.100 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.923 | 0.016 | 0.055 | 0.19 | 0.960 | 0.988 | 0.016 |  |  | 0.989 |  |
| `go_emotions` | choice | 1000 | 0.476 | 0.065 | 0.712 | 1.88 | 0.497 | 0.594 | 0.388 |  |  |  | 0.577 |
| `helpsteer2_helpfulness` | score | 1000 | 0.405 | 0.130 | 0.722 | 1.38 | 0.419 | 0.458 | 0.519 | 0.150 | 0.91 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.621 | 0.062 | 0.532 | 0.99 | 0.647 | 0.730 | 0.288 | 0.082 | 0.47 |  |  |
| `ledgar` | choice | 1000 | 0.719 | 0.108 | 0.436 | 1.23 | 0.759 | 0.876 | 0.139 |  |  |  |  |
| `massive` | choice | 1000 | 0.771 | 0.098 | 0.344 | 1.02 | 0.823 | 0.948 | 0.085 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.396 | 0.268 | 0.778 | 1.45 | 0.404 | 0.516 | 0.446 | 0.300 | 0.85 |  | 0.521 |
| `mmlu` | choice | 1000 | 0.707 | 0.069 | 0.412 | 0.85 | 0.743 | 0.888 | 0.127 |  |  |  |  |
| `mnli` | choice | 1000 | 0.865 | 0.023 | 0.199 | 0.35 | 0.894 | 0.980 | 0.036 |  |  |  |  |
| `paws` | noul | 1000 | 0.826 | 0.023 | 0.120 | 0.38 | 0.858 | 0.954 | 0.062 |  |  | 0.910 |  |
| `sms_spam` | noul | 800 | 0.964 | 0.014 | 0.030 | 0.11 | 0.986 | 0.995 | 0.006 |  |  | 0.974 |  |
| `sst5` | score | 1000 | 0.548 | 0.066 | 0.586 | 1.05 | 0.563 | 0.602 | 0.397 | 0.091 | 0.56 |  |  |
| `strategyqa_closed` | noul | 687 | 0.702 | 0.052 | 0.194 | 0.57 | 0.725 | 0.799 | 0.181 |  |  | 0.789 |  |
| `strategyqa_grounded` | noul | 687 | 0.907 | 0.074 | 0.075 | 0.26 | 0.940 | 1.000 | 0.017 |  |  | 0.977 |  |
| `stsb` | score | 1000 | 0.497 | 0.027 | 0.614 | 1.14 | 0.514 | 0.546 | 0.411 | 0.081 | 0.63 |  |  |
| `yelp5` | score | 1000 | 0.591 | 0.035 | 0.530 | 0.97 | 0.572 | 0.650 | 0.336 | 0.078 | 0.53 |  |  |
