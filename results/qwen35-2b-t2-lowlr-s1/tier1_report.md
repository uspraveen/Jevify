# Qwen/Qwen3.5-2B (Tier 2): Tier 2 vs Tier 0 vs Jev

Heads trained on 5,885 records from 16 sources (max 16 options per record), early-stopped on 1,500 validation records; epoch 1.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B (Tier 2) | Tier 2 heads + LoRA | 0.720 | 0.085 | 0.347 | 0.689 | 0.111 | 0.404 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-2B (Tier 2)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.817 | 0.087 | 0.283 | 0.61 | 0.853 | 0.960 | 0.056 |  |  |  |  |
| `banking77` | choice | 1000 | 0.728 | 0.127 | 0.416 | 1.24 | 0.774 | 0.920 | 0.107 |  |  |  |  |
| `boolq` | noul | 1000 | 0.861 | 0.082 | 0.111 | 0.44 | 0.898 | 0.952 | 0.070 |  |  | 0.924 |  |
| `chaosnli` | choice | 1599 | 0.575 | 0.328 | 0.719 | 1.65 | 0.587 | 0.689 | 0.295 |  |  |  | 0.412 |
| `civil_comments` | noul | 2000 | 0.923 | 0.049 | 0.067 | 0.26 | 0.948 | 0.981 | 0.026 |  |  | 0.800 | 0.095 |
| `clinc150` | choice | 1000 | 0.733 | 0.084 | 0.395 | 1.13 | 0.781 | 0.910 | 0.106 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.914 | 0.027 | 0.067 | 0.23 | 0.943 | 0.992 | 0.019 |  |  | 0.972 |  |
| `go_emotions` | choice | 1000 | 0.528 | 0.181 | 0.684 | 1.77 | 0.550 | 0.658 | 0.324 |  |  |  | 0.542 |
| `helpsteer2_helpfulness` | score | 1000 | 0.357 | 0.073 | 0.710 | 1.39 | 0.371 | 0.440 | 0.571 | 0.156 | 0.95 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.654 | 0.021 | 0.500 | 0.93 | 0.673 | 0.726 | 0.282 | 0.075 | 0.43 |  |  |
| `ledgar` | choice | 1000 | 0.737 | 0.104 | 0.392 | 1.12 | 0.778 | 0.914 | 0.104 |  |  |  |  |
| `massive` | choice | 1000 | 0.771 | 0.108 | 0.346 | 0.97 | 0.819 | 0.946 | 0.082 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.574 | 0.077 | 0.553 | 1.04 | 0.586 | 0.782 | 0.225 | 0.217 | 0.70 |  | 0.401 |
| `mmlu` | choice | 1000 | 0.600 | 0.195 | 0.573 | 1.20 | 0.631 | 0.814 | 0.194 |  |  |  |  |
| `mnli` | choice | 1000 | 0.827 | 0.118 | 0.287 | 0.64 | 0.860 | 0.960 | 0.057 |  |  |  |  |
| `paws` | noul | 1000 | 0.846 | 0.115 | 0.128 | 0.48 | 0.884 | 0.966 | 0.045 |  |  | 0.935 |  |
| `sms_spam` | noul | 800 | 0.971 | 0.023 | 0.025 | 0.12 | 0.989 | 0.998 | 0.004 |  |  | 0.988 |  |
| `sst5` | score | 1000 | 0.495 | 0.090 | 0.627 | 1.11 | 0.517 | 0.570 | 0.431 | 0.100 | 0.62 |  |  |
| `strategyqa_closed` | noul | 687 | 0.671 | 0.129 | 0.228 | 0.70 | 0.681 | 0.747 | 0.229 |  |  | 0.735 |  |
| `strategyqa_grounded` | noul | 687 | 0.811 | 0.075 | 0.138 | 0.45 | 0.840 | 0.939 | 0.072 |  |  | 0.910 |  |
| `stsb` | score | 1000 | 0.474 | 0.038 | 0.643 | 1.23 | 0.502 | 0.580 | 0.422 | 0.090 | 0.69 |  |  |
| `yelp5` | score | 1000 | 0.472 | 0.158 | 0.648 | 1.13 | 0.481 | 0.514 | 0.447 | 0.099 | 0.60 |  |  |
