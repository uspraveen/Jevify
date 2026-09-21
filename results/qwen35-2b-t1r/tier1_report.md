# Qwen3.5-2B (Tier 1 residual): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 6.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B (Tier 1 residual) | Tier 1 heads | 0.631 | 0.090 | 0.459 | 0.632 | 0.061 | 0.439 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen3.5-2B (Tier 1 residual)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.756 | 0.037 | 0.335 | 0.63 | 0.799 | 0.934 | 0.086 |  |  |  |  |
| `banking77` | choice | 1000 | 0.596 | 0.086 | 0.551 | 1.59 | 0.646 | 0.826 | 0.198 |  |  |  |  |
| `boolq` | noul | 1000 | 0.848 | 0.029 | 0.111 | 0.36 | 0.882 | 0.964 | 0.055 |  |  | 0.917 |  |
| `chaosnli` | choice | 1599 | 0.570 | 0.077 | 0.539 | 0.88 | 0.589 | 0.659 | 0.327 |  |  |  | 0.266 |
| `civil_comments` | noul | 2000 | 0.922 | 0.050 | 0.069 | 0.28 | 0.943 | 0.970 | 0.033 |  |  | 0.753 | 0.099 |
| `clinc150` | choice | 1000 | 0.484 | 0.086 | 0.720 | 3.40 | 0.522 | 0.662 | 0.314 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.886 | 0.033 | 0.090 | 0.32 | 0.918 | 0.950 | 0.045 |  |  | 0.962 |  |
| `go_emotions` | choice | 1000 | 0.397 | 0.069 | 0.768 | 2.09 | 0.411 | 0.532 | 0.452 |  |  |  | 0.632 |
| `helpsteer2_helpfulness` | score | 1000 | 0.382 | 0.070 | 0.717 | 1.39 | 0.393 | 0.468 | 0.538 | 0.161 | 0.98 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.630 | 0.068 | 0.543 | 1.05 | 0.656 | 0.676 | 0.316 | 0.088 | 0.52 |  |  |
| `ledgar` | choice | 1000 | 0.690 | 0.070 | 0.442 | 1.19 | 0.730 | 0.896 | 0.139 |  |  |  |  |
| `massive` | choice | 1000 | 0.576 | 0.078 | 0.568 | 1.61 | 0.630 | 0.796 | 0.199 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.452 | 0.106 | 0.702 | 1.29 | 0.454 | 0.522 | 0.456 | 0.283 | 0.89 |  | 0.498 |
| `mmlu` | choice | 1000 | 0.545 | 0.112 | 0.582 | 1.11 | 0.564 | 0.728 | 0.255 |  |  |  |  |
| `mnli` | choice | 1000 | 0.753 | 0.045 | 0.348 | 0.59 | 0.781 | 0.892 | 0.110 |  |  |  |  |
| `paws` | noul | 1000 | 0.832 | 0.036 | 0.121 | 0.38 | 0.859 | 0.958 | 0.061 |  |  | 0.912 |  |
| `sms_spam` | noul | 800 | 0.963 | 0.012 | 0.031 | 0.12 | 0.986 | 0.995 | 0.012 |  |  | 0.970 |  |
| `sst5` | score | 1000 | 0.418 | 0.097 | 0.695 | 1.31 | 0.434 | 0.466 | 0.523 | 0.121 | 0.74 |  |  |
| `strategyqa_closed` | noul | 687 | 0.601 | 0.027 | 0.236 | 0.66 | 0.607 | 0.640 | 0.338 |  |  | 0.638 |  |
| `strategyqa_grounded` | noul | 687 | 0.793 | 0.082 | 0.149 | 0.46 | 0.824 | 0.924 | 0.090 |  |  | 0.878 |  |
| `stsb` | score | 1000 | 0.395 | 0.050 | 0.710 | 1.42 | 0.413 | 0.496 | 0.493 | 0.111 | 0.83 |  |  |
| `yelp5` | score | 1000 | 0.414 | 0.193 | 0.758 | 1.58 | 0.402 | 0.458 | 0.494 | 0.153 | 0.89 |  |  |
