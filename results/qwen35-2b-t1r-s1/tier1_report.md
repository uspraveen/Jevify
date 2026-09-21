# Qwen3.5-2B (Tier 1 residual, seed 1): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 5.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B (Tier 1 residual, seed 1) | Tier 1 heads | 0.636 | 0.089 | 0.452 | 0.627 | 0.061 | 0.440 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen3.5-2B (Tier 1 residual, seed 1)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.771 | 0.037 | 0.316 | 0.60 | 0.809 | 0.948 | 0.075 |  |  |  |  |
| `banking77` | choice | 1000 | 0.598 | 0.088 | 0.555 | 1.62 | 0.641 | 0.828 | 0.200 |  |  |  |  |
| `boolq` | noul | 1000 | 0.849 | 0.038 | 0.110 | 0.36 | 0.886 | 0.960 | 0.055 |  |  | 0.916 |  |
| `chaosnli` | choice | 1599 | 0.556 | 0.091 | 0.549 | 0.89 | 0.575 | 0.639 | 0.342 |  |  |  | 0.271 |
| `civil_comments` | noul | 2000 | 0.921 | 0.046 | 0.069 | 0.27 | 0.944 | 0.971 | 0.032 |  |  | 0.754 | 0.100 |
| `clinc150` | choice | 1000 | 0.486 | 0.117 | 0.719 | 3.40 | 0.520 | 0.670 | 0.314 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.885 | 0.041 | 0.092 | 0.32 | 0.918 | 0.944 | 0.047 |  |  | 0.962 |  |
| `go_emotions` | choice | 1000 | 0.410 | 0.049 | 0.753 | 2.05 | 0.437 | 0.542 | 0.436 |  |  |  | 0.622 |
| `helpsteer2_helpfulness` | score | 1000 | 0.352 | 0.067 | 0.716 | 1.39 | 0.359 | 0.438 | 0.559 | 0.161 | 0.99 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.624 | 0.062 | 0.542 | 1.05 | 0.650 | 0.684 | 0.314 | 0.088 | 0.53 |  |  |
| `ledgar` | choice | 1000 | 0.691 | 0.051 | 0.436 | 1.17 | 0.730 | 0.898 | 0.137 |  |  |  |  |
| `massive` | choice | 1000 | 0.563 | 0.081 | 0.574 | 1.64 | 0.616 | 0.788 | 0.208 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.474 | 0.091 | 0.697 | 1.28 | 0.471 | 0.522 | 0.464 | 0.288 | 0.90 |  | 0.495 |
| `mmlu` | choice | 1000 | 0.557 | 0.080 | 0.557 | 1.06 | 0.590 | 0.760 | 0.234 |  |  |  |  |
| `mnli` | choice | 1000 | 0.730 | 0.040 | 0.367 | 0.62 | 0.758 | 0.868 | 0.126 |  |  |  |  |
| `paws` | noul | 1000 | 0.832 | 0.039 | 0.122 | 0.38 | 0.859 | 0.954 | 0.062 |  |  | 0.911 |  |
| `sms_spam` | noul | 800 | 0.965 | 0.022 | 0.032 | 0.13 | 0.986 | 0.995 | 0.012 |  |  | 0.969 |  |
| `sst5` | score | 1000 | 0.410 | 0.113 | 0.706 | 1.33 | 0.429 | 0.452 | 0.536 | 0.123 | 0.75 |  |  |
| `strategyqa_closed` | noul | 687 | 0.587 | 0.039 | 0.234 | 0.66 | 0.608 | 0.660 | 0.324 |  |  | 0.640 |  |
| `strategyqa_grounded` | noul | 687 | 0.785 | 0.065 | 0.150 | 0.46 | 0.811 | 0.927 | 0.094 |  |  | 0.879 |  |
| `stsb` | score | 1000 | 0.379 | 0.066 | 0.716 | 1.43 | 0.400 | 0.494 | 0.496 | 0.112 | 0.84 |  |  |
| `yelp5` | score | 1000 | 0.418 | 0.181 | 0.736 | 1.49 | 0.403 | 0.468 | 0.495 | 0.144 | 0.85 |  |  |
