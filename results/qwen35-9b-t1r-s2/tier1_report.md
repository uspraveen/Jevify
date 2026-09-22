# Qwen/Qwen3.5-9B (Tier 1): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 5.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B (Tier 1) | Tier 1 heads | 0.741 | 0.092 | 0.337 | 0.702 | 0.066 | 0.375 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-9B | Tier 0 (recipe refit w/o held-out) | 0.729 | 0.135 | 0.335 | 0.670 | 0.092 | 0.401 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-9B (Tier 1)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.882 | 0.027 | 0.179 | 0.34 | 0.918 | 0.990 | 0.025 |  |  |  |  |
| `banking77` | choice | 1000 | 0.722 | 0.129 | 0.415 | 1.25 | 0.769 | 0.918 | 0.102 |  |  |  |  |
| `boolq` | noul | 1000 | 0.898 | 0.022 | 0.073 | 0.24 | 0.936 | 0.988 | 0.022 |  |  | 0.963 |  |
| `chaosnli` | choice | 1599 | 0.601 | 0.164 | 0.523 | 0.84 | 0.621 | 0.725 | 0.251 |  |  |  | 0.290 |
| `civil_comments` | noul | 2000 | 0.927 | 0.017 | 0.061 | 0.22 | 0.949 | 0.984 | 0.023 |  |  | 0.814 | 0.095 |
| `clinc150` | choice | 1000 | 0.776 | 0.082 | 0.347 | 1.02 | 0.810 | 0.918 | 0.092 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.918 | 0.018 | 0.057 | 0.20 | 0.959 | 0.982 | 0.017 |  |  | 0.989 |  |
| `go_emotions` | choice | 1000 | 0.472 | 0.086 | 0.717 | 1.90 | 0.489 | 0.574 | 0.393 |  |  |  | 0.578 |
| `helpsteer2_helpfulness` | score | 1000 | 0.394 | 0.140 | 0.728 | 1.40 | 0.412 | 0.458 | 0.542 | 0.149 | 0.91 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.625 | 0.027 | 0.527 | 0.99 | 0.649 | 0.736 | 0.280 | 0.081 | 0.46 |  |  |
| `ledgar` | choice | 1000 | 0.728 | 0.112 | 0.430 | 1.22 | 0.760 | 0.884 | 0.133 |  |  |  |  |
| `massive` | choice | 1000 | 0.773 | 0.111 | 0.359 | 1.08 | 0.819 | 0.922 | 0.092 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.393 | 0.277 | 0.809 | 1.59 | 0.386 | 0.498 | 0.488 | 0.324 | 0.90 |  | 0.534 |
| `mmlu` | choice | 1000 | 0.709 | 0.074 | 0.417 | 0.86 | 0.742 | 0.882 | 0.132 |  |  |  |  |
| `mnli` | choice | 1000 | 0.863 | 0.013 | 0.198 | 0.34 | 0.896 | 0.984 | 0.035 |  |  |  |  |
| `paws` | noul | 1000 | 0.822 | 0.012 | 0.120 | 0.38 | 0.858 | 0.950 | 0.061 |  |  | 0.909 |  |
| `sms_spam` | noul | 800 | 0.964 | 0.011 | 0.030 | 0.12 | 0.986 | 0.995 | 0.007 |  |  | 0.973 |  |
| `sst5` | score | 1000 | 0.541 | 0.051 | 0.592 | 1.06 | 0.550 | 0.606 | 0.409 | 0.092 | 0.57 |  |  |
| `strategyqa_closed` | noul | 687 | 0.697 | 0.052 | 0.193 | 0.57 | 0.723 | 0.802 | 0.181 |  |  | 0.788 |  |
| `strategyqa_grounded` | noul | 687 | 0.905 | 0.085 | 0.079 | 0.27 | 0.934 | 1.000 | 0.018 |  |  | 0.976 |  |
| `stsb` | score | 1000 | 0.494 | 0.042 | 0.621 | 1.14 | 0.516 | 0.546 | 0.414 | 0.081 | 0.63 |  |  |
| `yelp5` | score | 1000 | 0.571 | 0.065 | 0.551 | 0.99 | 0.559 | 0.606 | 0.368 | 0.080 | 0.53 |  |  |
