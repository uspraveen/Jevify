# Qwen/Qwen3.5-2B (Tier 1): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 5.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B (Tier 1) | Tier 1 heads | 0.620 | 0.107 | 0.482 | 0.635 | 0.068 | 0.438 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-2B (Tier 1)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.758 | 0.026 | 0.346 | 0.66 | 0.792 | 0.916 | 0.094 |  |  |  |  |
| `banking77` | choice | 1000 | 0.593 | 0.169 | 0.569 | 1.61 | 0.638 | 0.816 | 0.200 |  |  |  |  |
| `boolq` | noul | 1000 | 0.852 | 0.020 | 0.109 | 0.35 | 0.881 | 0.958 | 0.056 |  |  | 0.917 |  |
| `chaosnli` | choice | 1599 | 0.561 | 0.097 | 0.555 | 0.91 | 0.583 | 0.640 | 0.344 |  |  |  | 0.280 |
| `civil_comments` | noul | 2000 | 0.921 | 0.022 | 0.067 | 0.25 | 0.944 | 0.971 | 0.032 |  |  | 0.755 | 0.105 |
| `clinc150` | choice | 1000 | 0.448 | 0.171 | 0.769 | 3.80 | 0.479 | 0.626 | 0.346 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.881 | 0.027 | 0.093 | 0.32 | 0.914 | 0.954 | 0.046 |  |  | 0.962 |  |
| `go_emotions` | choice | 1000 | 0.472 | 0.072 | 0.724 | 1.95 | 0.498 | 0.578 | 0.405 |  |  |  | 0.595 |
| `helpsteer2_helpfulness` | score | 1000 | 0.353 | 0.065 | 0.718 | 1.40 | 0.366 | 0.416 | 0.578 | 0.161 | 0.99 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.624 | 0.037 | 0.537 | 1.04 | 0.643 | 0.694 | 0.308 | 0.087 | 0.50 |  |  |
| `ledgar` | choice | 1000 | 0.676 | 0.096 | 0.461 | 1.22 | 0.714 | 0.892 | 0.142 |  |  |  |  |
| `massive` | choice | 1000 | 0.599 | 0.124 | 0.541 | 1.55 | 0.650 | 0.848 | 0.178 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.488 | 0.136 | 0.728 | 1.37 | 0.481 | 0.480 | 0.515 | 0.316 | 0.94 |  | 0.502 |
| `mmlu` | choice | 1000 | 0.555 | 0.121 | 0.587 | 1.14 | 0.586 | 0.726 | 0.252 |  |  |  |  |
| `mnli` | choice | 1000 | 0.751 | 0.040 | 0.348 | 0.59 | 0.786 | 0.888 | 0.111 |  |  |  |  |
| `paws` | noul | 1000 | 0.833 | 0.027 | 0.119 | 0.38 | 0.872 | 0.952 | 0.061 |  |  | 0.910 |  |
| `sms_spam` | noul | 800 | 0.959 | 0.015 | 0.034 | 0.13 | 0.983 | 0.995 | 0.012 |  |  | 0.967 |  |
| `sst5` | score | 1000 | 0.437 | 0.089 | 0.690 | 1.31 | 0.443 | 0.486 | 0.518 | 0.120 | 0.74 |  |  |
| `strategyqa_closed` | noul | 687 | 0.581 | 0.043 | 0.235 | 0.66 | 0.592 | 0.669 | 0.327 |  |  | 0.643 |  |
| `strategyqa_grounded` | noul | 687 | 0.761 | 0.038 | 0.156 | 0.47 | 0.790 | 0.910 | 0.104 |  |  | 0.880 |  |
| `stsb` | score | 1000 | 0.391 | 0.058 | 0.717 | 1.44 | 0.408 | 0.500 | 0.531 | 0.114 | 0.84 |  |  |
| `yelp5` | score | 1000 | 0.382 | 0.243 | 0.800 | 1.75 | 0.363 | 0.448 | 0.502 | 0.173 | 0.98 |  |  |
