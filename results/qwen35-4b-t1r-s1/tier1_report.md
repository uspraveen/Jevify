# Qwen/Qwen3.5-4B (Tier 1): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 5.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B (Tier 1) | Tier 1 heads | 0.735 | 0.110 | 0.353 | 0.690 | 0.075 | 0.387 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-4B | Tier 0 (recipe refit w/o held-out) | 0.714 | 0.139 | 0.355 | 0.641 | 0.087 | 0.424 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-4B (Tier 1)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.898 | 0.022 | 0.147 | 0.30 | 0.944 | 0.990 | 0.018 |  |  |  |  |
| `banking77` | choice | 1000 | 0.750 | 0.075 | 0.384 | 1.11 | 0.797 | 0.930 | 0.098 |  |  |  |  |
| `boolq` | noul | 1000 | 0.900 | 0.037 | 0.077 | 0.27 | 0.929 | 0.986 | 0.028 |  |  | 0.961 |  |
| `chaosnli` | choice | 1599 | 0.581 | 0.148 | 0.543 | 0.88 | 0.596 | 0.698 | 0.284 |  |  |  | 0.290 |
| `civil_comments` | noul | 2000 | 0.923 | 0.029 | 0.063 | 0.23 | 0.950 | 0.986 | 0.023 |  |  | 0.812 | 0.094 |
| `clinc150` | choice | 1000 | 0.667 | 0.049 | 0.471 | 1.55 | 0.717 | 0.844 | 0.143 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.935 | 0.081 | 0.071 | 0.26 | 0.948 | 0.956 | 0.031 |  |  | 0.980 |  |
| `go_emotions` | choice | 1000 | 0.438 | 0.100 | 0.764 | 2.10 | 0.461 | 0.534 | 0.460 |  |  |  | 0.596 |
| `helpsteer2_helpfulness` | score | 1000 | 0.392 | 0.153 | 0.738 | 1.45 | 0.398 | 0.470 | 0.518 | 0.157 | 0.93 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.644 | 0.055 | 0.513 | 0.97 | 0.661 | 0.732 | 0.288 | 0.076 | 0.44 |  |  |
| `ledgar` | choice | 1000 | 0.736 | 0.089 | 0.389 | 1.15 | 0.784 | 0.916 | 0.105 |  |  |  |  |
| `massive` | choice | 1000 | 0.756 | 0.065 | 0.356 | 1.01 | 0.804 | 0.946 | 0.086 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.492 | 0.177 | 0.653 | 1.17 | 0.504 | 0.626 | 0.353 | 0.274 | 0.81 |  | 0.444 |
| `mmlu` | choice | 1000 | 0.668 | 0.092 | 0.453 | 0.88 | 0.712 | 0.872 | 0.136 |  |  |  |  |
| `mnli` | choice | 1000 | 0.832 | 0.025 | 0.242 | 0.42 | 0.859 | 0.970 | 0.053 |  |  |  |  |
| `paws` | noul | 1000 | 0.825 | 0.047 | 0.125 | 0.39 | 0.856 | 0.940 | 0.067 |  |  | 0.906 |  |
| `sms_spam` | noul | 800 | 0.954 | 0.017 | 0.035 | 0.13 | 0.983 | 0.995 | 0.007 |  |  | 0.977 |  |
| `sst5` | score | 1000 | 0.490 | 0.158 | 0.655 | 1.19 | 0.507 | 0.566 | 0.431 | 0.102 | 0.59 |  |  |
| `strategyqa_closed` | noul | 687 | 0.681 | 0.051 | 0.207 | 0.60 | 0.689 | 0.762 | 0.214 |  |  | 0.742 |  |
| `strategyqa_grounded` | noul | 687 | 0.885 | 0.128 | 0.104 | 0.35 | 0.913 | 0.988 | 0.030 |  |  | 0.953 |  |
| `stsb` | score | 1000 | 0.475 | 0.056 | 0.640 | 1.20 | 0.497 | 0.542 | 0.420 | 0.082 | 0.64 |  |  |
| `yelp5` | score | 1000 | 0.532 | 0.202 | 0.671 | 1.27 | 0.529 | 0.538 | 0.465 | 0.105 | 0.61 |  |  |
