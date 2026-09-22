# Qwen/Qwen3.5-2B (Tier 1): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 5.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B (Tier 1) | Tier 1 heads | 0.622 | 0.115 | 0.483 | 0.633 | 0.069 | 0.440 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-2B (Tier 1)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.755 | 0.046 | 0.346 | 0.66 | 0.793 | 0.914 | 0.092 |  |  |  |  |
| `banking77` | choice | 1000 | 0.601 | 0.156 | 0.568 | 1.61 | 0.646 | 0.826 | 0.199 |  |  |  |  |
| `boolq` | noul | 1000 | 0.852 | 0.022 | 0.108 | 0.35 | 0.883 | 0.960 | 0.055 |  |  | 0.917 |  |
| `chaosnli` | choice | 1599 | 0.535 | 0.127 | 0.591 | 0.97 | 0.552 | 0.595 | 0.383 |  |  |  | 0.294 |
| `civil_comments` | noul | 2000 | 0.920 | 0.025 | 0.067 | 0.25 | 0.944 | 0.971 | 0.032 |  |  | 0.757 | 0.104 |
| `clinc150` | choice | 1000 | 0.478 | 0.182 | 0.753 | 3.72 | 0.506 | 0.640 | 0.328 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.887 | 0.022 | 0.087 | 0.30 | 0.917 | 0.962 | 0.040 |  |  | 0.962 |  |
| `go_emotions` | choice | 1000 | 0.463 | 0.065 | 0.721 | 1.93 | 0.492 | 0.580 | 0.402 |  |  |  | 0.595 |
| `helpsteer2_helpfulness` | score | 1000 | 0.345 | 0.075 | 0.715 | 1.39 | 0.359 | 0.420 | 0.574 | 0.161 | 0.99 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.623 | 0.030 | 0.538 | 1.04 | 0.646 | 0.690 | 0.307 | 0.088 | 0.50 |  |  |
| `ledgar` | choice | 1000 | 0.682 | 0.086 | 0.455 | 1.21 | 0.724 | 0.898 | 0.137 |  |  |  |  |
| `massive` | choice | 1000 | 0.604 | 0.113 | 0.535 | 1.54 | 0.652 | 0.852 | 0.175 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.455 | 0.168 | 0.770 | 1.48 | 0.456 | 0.436 | 0.560 | 0.337 | 0.97 |  | 0.521 |
| `mmlu` | choice | 1000 | 0.560 | 0.119 | 0.587 | 1.14 | 0.588 | 0.718 | 0.249 |  |  |  |  |
| `mnli` | choice | 1000 | 0.749 | 0.036 | 0.353 | 0.60 | 0.777 | 0.882 | 0.117 |  |  |  |  |
| `paws` | noul | 1000 | 0.831 | 0.031 | 0.121 | 0.38 | 0.866 | 0.948 | 0.064 |  |  | 0.908 |  |
| `sms_spam` | noul | 800 | 0.961 | 0.018 | 0.034 | 0.13 | 0.983 | 0.995 | 0.013 |  |  | 0.965 |  |
| `sst5` | score | 1000 | 0.435 | 0.101 | 0.698 | 1.32 | 0.447 | 0.476 | 0.529 | 0.121 | 0.74 |  |  |
| `strategyqa_closed` | noul | 687 | 0.582 | 0.037 | 0.236 | 0.66 | 0.600 | 0.654 | 0.326 |  |  | 0.643 |  |
| `strategyqa_grounded` | noul | 687 | 0.764 | 0.038 | 0.153 | 0.46 | 0.796 | 0.916 | 0.100 |  |  | 0.880 |  |
| `stsb` | score | 1000 | 0.380 | 0.071 | 0.718 | 1.44 | 0.398 | 0.510 | 0.528 | 0.113 | 0.83 |  |  |
| `yelp5` | score | 1000 | 0.390 | 0.232 | 0.791 | 1.72 | 0.374 | 0.450 | 0.501 | 0.169 | 0.97 |  |  |
