# Qwen3.5-2B (Tier 1 residual, seed 4): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 8.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B (Tier 1 residual, seed 4) | Tier 1 heads | 0.555 | 0.157 | 0.531 | 0.635 | 0.079 | 0.440 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen3.5-2B (Tier 1 residual, seed 4)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.704 | 0.070 | 0.417 | 0.81 | 0.743 | 0.860 | 0.133 |  |  |  |  |
| `banking77` | choice | 1000 | 0.643 | 0.132 | 0.522 | 1.41 | 0.686 | 0.820 | 0.182 |  |  |  |  |
| `boolq` | noul | 1000 | 0.841 | 0.020 | 0.112 | 0.36 | 0.873 | 0.958 | 0.057 |  |  | 0.916 |  |
| `chaosnli` | choice | 1599 | 0.600 | 0.088 | 0.521 | 0.87 | 0.620 | 0.696 | 0.293 |  |  |  | 0.269 |
| `civil_comments` | noul | 2000 | 0.922 | 0.052 | 0.070 | 0.28 | 0.945 | 0.972 | 0.032 |  |  | 0.758 | 0.100 |
| `clinc150` | choice | 1000 | 0.394 | 0.189 | 0.828 | 4.31 | 0.420 | 0.574 | 0.400 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.853 | 0.051 | 0.114 | 0.39 | 0.887 | 0.940 | 0.058 |  |  | 0.962 |  |
| `go_emotions` | choice | 1000 | 0.479 | 0.066 | 0.715 | 1.92 | 0.509 | 0.590 | 0.396 |  |  |  | 0.596 |
| `helpsteer2_helpfulness` | score | 1000 | 0.377 | 0.069 | 0.709 | 1.38 | 0.387 | 0.438 | 0.560 | 0.160 | 0.97 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.633 | 0.038 | 0.539 | 1.06 | 0.659 | 0.686 | 0.304 | 0.088 | 0.51 |  |  |
| `ledgar` | choice | 1000 | 0.635 | 0.130 | 0.523 | 1.41 | 0.677 | 0.818 | 0.186 |  |  |  |  |
| `massive` | choice | 1000 | 0.589 | 0.126 | 0.552 | 1.57 | 0.638 | 0.828 | 0.187 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.275 | 0.283 | 0.856 | 1.65 | 0.276 | 0.324 | 0.644 | 0.322 | 0.94 |  | 0.582 |
| `mmlu` | choice | 1000 | 0.501 | 0.195 | 0.658 | 1.33 | 0.531 | 0.678 | 0.312 |  |  |  |  |
| `mnli` | choice | 1000 | 0.769 | 0.025 | 0.325 | 0.56 | 0.802 | 0.912 | 0.097 |  |  |  |  |
| `paws` | noul | 1000 | 0.832 | 0.052 | 0.126 | 0.40 | 0.853 | 0.942 | 0.066 |  |  | 0.913 |  |
| `sms_spam` | noul | 800 | 0.951 | 0.027 | 0.036 | 0.14 | 0.985 | 0.995 | 0.012 |  |  | 0.973 |  |
| `sst5` | score | 1000 | 0.438 | 0.110 | 0.688 | 1.30 | 0.451 | 0.478 | 0.517 | 0.118 | 0.72 |  |  |
| `strategyqa_closed` | noul | 687 | 0.559 | 0.078 | 0.246 | 0.68 | 0.565 | 0.619 | 0.351 |  |  | 0.639 |  |
| `strategyqa_grounded` | noul | 687 | 0.713 | 0.081 | 0.170 | 0.51 | 0.749 | 0.898 | 0.122 |  |  | 0.879 |  |
| `stsb` | score | 1000 | 0.393 | 0.048 | 0.700 | 1.39 | 0.421 | 0.506 | 0.481 | 0.109 | 0.82 |  |  |
| `yelp5` | score | 1000 | 0.389 | 0.267 | 0.803 | 1.77 | 0.382 | 0.450 | 0.491 | 0.164 | 0.91 |  |  |
