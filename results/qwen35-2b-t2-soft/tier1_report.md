# Qwen3.5-2B (Tier 2, soft labels): Tier 2 vs Tier 0 vs Jev

Heads trained on 5,885 records from 16 sources (max 16 options per record), early-stopped on 1,500 validation records; epoch 0.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B (Tier 2, soft labels) | Tier 2 heads + LoRA | 0.672 | 0.139 | 0.414 | 0.675 | 0.107 | 0.411 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen3.5-2B (Tier 2, soft labels)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.808 | 0.076 | 0.284 | 0.58 | 0.850 | 0.952 | 0.063 |  |  |  |  |
| `banking77` | choice | 1000 | 0.686 | 0.193 | 0.494 | 1.61 | 0.719 | 0.892 | 0.125 |  |  |  |  |
| `boolq` | noul | 1000 | 0.851 | 0.029 | 0.108 | 0.35 | 0.886 | 0.952 | 0.058 |  |  | 0.919 |  |
| `chaosnli` | choice | 1599 | 0.689 | 0.175 | 0.486 | 0.99 | 0.714 | 0.800 | 0.191 |  |  |  | 0.332 |
| `civil_comments` | noul | 2000 | 0.923 | 0.267 | 0.139 | 0.45 | 0.954 | 0.986 | 0.020 |  |  | 0.835 | 0.280 |
| `clinc150` | choice | 1000 | 0.667 | 0.172 | 0.515 | 1.84 | 0.714 | 0.878 | 0.135 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.931 | 0.145 | 0.078 | 0.29 | 0.966 | 0.988 | 0.015 |  |  | 0.977 |  |
| `go_emotions` | choice | 1000 | 0.444 | 0.084 | 0.733 | 1.94 | 0.470 | 0.514 | 0.409 |  |  |  | 0.594 |
| `helpsteer2_helpfulness` | score | 1000 | 0.407 | 0.090 | 0.727 | 1.51 | 0.433 | 0.492 | 0.520 | 0.168 | 0.97 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.654 | 0.048 | 0.506 | 0.96 | 0.674 | 0.738 | 0.271 | 0.078 | 0.45 |  |  |
| `ledgar` | choice | 1000 | 0.717 | 0.160 | 0.443 | 1.44 | 0.754 | 0.896 | 0.129 |  |  |  |  |
| `massive` | choice | 1000 | 0.741 | 0.121 | 0.387 | 1.12 | 0.796 | 0.916 | 0.106 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.446 | 0.160 | 0.763 | 1.48 | 0.463 | 0.508 | 0.491 | 0.323 | 0.94 |  | 0.525 |
| `mmlu` | choice | 1000 | 0.580 | 0.199 | 0.578 | 1.15 | 0.612 | 0.796 | 0.210 |  |  |  |  |
| `mnli` | choice | 1000 | 0.806 | 0.110 | 0.324 | 0.72 | 0.834 | 0.902 | 0.095 |  |  |  |  |
| `paws` | noul | 1000 | 0.827 | 0.030 | 0.113 | 0.35 | 0.870 | 0.962 | 0.051 |  |  | 0.943 |  |
| `sms_spam` | noul | 800 | 0.958 | 0.017 | 0.031 | 0.11 | 0.989 | 1.000 | 0.004 |  |  | 0.987 |  |
| `sst5` | score | 1000 | 0.506 | 0.047 | 0.610 | 1.11 | 0.531 | 0.560 | 0.392 | 0.100 | 0.61 |  |  |
| `strategyqa_closed` | noul | 687 | 0.581 | 0.070 | 0.227 | 0.64 | 0.592 | 0.701 | 0.280 |  |  | 0.722 |  |
| `strategyqa_grounded` | noul | 687 | 0.786 | 0.044 | 0.140 | 0.43 | 0.819 | 0.939 | 0.078 |  |  | 0.898 |  |
| `stsb` | score | 1000 | 0.424 | 0.076 | 0.676 | 1.29 | 0.433 | 0.442 | 0.580 | 0.099 | 0.76 |  |  |
| `yelp5` | score | 1000 | 0.392 | 0.238 | 0.704 | 1.21 | 0.397 | 0.416 | 0.552 | 0.108 | 0.63 |  |  |
