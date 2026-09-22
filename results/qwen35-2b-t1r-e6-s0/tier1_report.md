# Qwen/Qwen3.5-2B (Tier 1): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 5.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B (Tier 1) | Tier 1 heads | 0.616 | 0.107 | 0.481 | 0.637 | 0.067 | 0.439 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-2B (Tier 1)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.742 | 0.052 | 0.350 | 0.67 | 0.786 | 0.906 | 0.096 |  |  |  |  |
| `banking77` | choice | 1000 | 0.604 | 0.161 | 0.562 | 1.59 | 0.651 | 0.814 | 0.195 |  |  |  |  |
| `boolq` | noul | 1000 | 0.854 | 0.020 | 0.109 | 0.35 | 0.880 | 0.958 | 0.055 |  |  | 0.917 |  |
| `chaosnli` | choice | 1599 | 0.558 | 0.107 | 0.573 | 0.94 | 0.573 | 0.620 | 0.360 |  |  |  | 0.288 |
| `civil_comments` | noul | 2000 | 0.920 | 0.015 | 0.066 | 0.24 | 0.944 | 0.971 | 0.032 |  |  | 0.756 | 0.107 |
| `clinc150` | choice | 1000 | 0.466 | 0.168 | 0.764 | 3.79 | 0.496 | 0.640 | 0.341 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.884 | 0.023 | 0.089 | 0.30 | 0.917 | 0.962 | 0.041 |  |  | 0.962 |  |
| `go_emotions` | choice | 1000 | 0.466 | 0.057 | 0.723 | 1.94 | 0.486 | 0.570 | 0.402 |  |  |  | 0.597 |
| `helpsteer2_helpfulness` | score | 1000 | 0.374 | 0.072 | 0.715 | 1.39 | 0.387 | 0.462 | 0.544 | 0.161 | 0.98 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.627 | 0.041 | 0.538 | 1.05 | 0.646 | 0.688 | 0.312 | 0.088 | 0.49 |  |  |
| `ledgar` | choice | 1000 | 0.675 | 0.096 | 0.470 | 1.25 | 0.713 | 0.880 | 0.147 |  |  |  |  |
| `massive` | choice | 1000 | 0.611 | 0.111 | 0.535 | 1.53 | 0.659 | 0.856 | 0.173 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.455 | 0.126 | 0.727 | 1.37 | 0.458 | 0.496 | 0.500 | 0.304 | 0.92 |  | 0.507 |
| `mmlu` | choice | 1000 | 0.547 | 0.128 | 0.596 | 1.16 | 0.578 | 0.716 | 0.259 |  |  |  |  |
| `mnli` | choice | 1000 | 0.756 | 0.032 | 0.342 | 0.58 | 0.787 | 0.898 | 0.108 |  |  |  |  |
| `paws` | noul | 1000 | 0.828 | 0.029 | 0.121 | 0.38 | 0.862 | 0.942 | 0.065 |  |  | 0.908 |  |
| `sms_spam` | noul | 800 | 0.956 | 0.015 | 0.036 | 0.14 | 0.983 | 0.995 | 0.013 |  |  | 0.964 |  |
| `sst5` | score | 1000 | 0.448 | 0.084 | 0.691 | 1.33 | 0.458 | 0.500 | 0.510 | 0.120 | 0.73 |  |  |
| `strategyqa_closed` | noul | 687 | 0.578 | 0.048 | 0.236 | 0.67 | 0.594 | 0.666 | 0.330 |  |  | 0.640 |  |
| `strategyqa_grounded` | noul | 687 | 0.763 | 0.038 | 0.155 | 0.47 | 0.790 | 0.913 | 0.104 |  |  | 0.879 |  |
| `stsb` | score | 1000 | 0.392 | 0.054 | 0.713 | 1.43 | 0.412 | 0.506 | 0.520 | 0.113 | 0.83 |  |  |
| `yelp5` | score | 1000 | 0.385 | 0.238 | 0.803 | 1.78 | 0.366 | 0.452 | 0.495 | 0.177 | 1.00 |  |  |
