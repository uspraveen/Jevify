# Qwen3.5-4B (Tier 2): Tier 2 vs Tier 0 vs Jev

Heads trained on 5,885 records from 16 sources (max 16 options per record), early-stopped on 1,500 validation records; epoch 1.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B (Tier 2) | Tier 2 heads + LoRA | 0.769 | 0.107 | 0.303 | 0.739 | 0.111 | 0.357 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-4B | Tier 0 (recipe refit w/o held-out) | 0.714 | 0.139 | 0.355 | 0.641 | 0.087 | 0.424 |

## Per-config (Tier 1)

**model:** `Qwen3.5-4B (Tier 2)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.932 | 0.049 | 0.120 | 0.30 | 0.968 | 0.996 | 0.011 |  |  |  |  |
| `banking77` | choice | 1000 | 0.785 | 0.134 | 0.349 | 1.12 | 0.822 | 0.946 | 0.072 |  |  |  |  |
| `boolq` | noul | 1000 | 0.911 | 0.074 | 0.081 | 0.39 | 0.936 | 0.994 | 0.021 |  |  | 0.966 |  |
| `chaosnli` | choice | 1599 | 0.645 | 0.290 | 0.626 | 1.53 | 0.667 | 0.746 | 0.225 |  |  |  | 0.399 |
| `civil_comments` | noul | 2000 | 0.931 | 0.048 | 0.060 | 0.24 | 0.959 | 0.993 | 0.013 |  |  | 0.892 | 0.088 |
| `clinc150` | choice | 1000 | 0.829 | 0.112 | 0.288 | 0.91 | 0.861 | 0.964 | 0.058 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.954 | 0.037 | 0.040 | 0.18 | 0.976 | 0.996 | 0.009 |  |  | 0.986 |  |
| `go_emotions` | choice | 1000 | 0.544 | 0.264 | 0.702 | 1.90 | 0.573 | 0.700 | 0.301 |  |  |  | 0.529 |
| `helpsteer2_helpfulness` | score | 1000 | 0.407 | 0.033 | 0.677 | 1.30 | 0.422 | 0.484 | 0.510 | 0.138 | 0.87 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.672 | 0.033 | 0.458 | 0.83 | 0.690 | 0.784 | 0.245 | 0.066 | 0.40 |  |  |
| `ledgar` | choice | 1000 | 0.757 | 0.150 | 0.390 | 1.24 | 0.802 | 0.898 | 0.104 |  |  |  |  |
| `massive` | choice | 1000 | 0.822 | 0.112 | 0.294 | 0.93 | 0.867 | 0.966 | 0.054 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.625 | 0.078 | 0.512 | 1.02 | 0.637 | 0.810 | 0.213 | 0.201 | 0.63 |  | 0.369 |
| `mmlu` | choice | 1000 | 0.730 | 0.188 | 0.429 | 1.01 | 0.774 | 0.936 | 0.092 |  |  |  |  |
| `mnli` | choice | 1000 | 0.890 | 0.085 | 0.193 | 0.48 | 0.922 | 0.980 | 0.032 |  |  |  |  |
| `paws` | noul | 1000 | 0.898 | 0.085 | 0.092 | 0.42 | 0.930 | 0.968 | 0.044 |  |  | 0.959 |  |
| `sms_spam` | noul | 800 | 0.980 | 0.019 | 0.020 | 0.12 | 0.994 | 1.000 | 0.002 |  |  | 0.994 |  |
| `sst5` | score | 1000 | 0.547 | 0.033 | 0.582 | 1.05 | 0.569 | 0.612 | 0.385 | 0.092 | 0.59 |  |  |
| `strategyqa_closed` | noul | 687 | 0.745 | 0.180 | 0.214 | 0.84 | 0.764 | 0.852 | 0.147 |  |  | 0.822 |  |
| `strategyqa_grounded` | noul | 687 | 0.901 | 0.071 | 0.079 | 0.29 | 0.942 | 1.000 | 0.016 |  |  | 0.977 |  |
| `stsb` | score | 1000 | 0.553 | 0.053 | 0.550 | 0.98 | 0.574 | 0.620 | 0.338 | 0.067 | 0.54 |  |  |
| `yelp5` | score | 1000 | 0.372 | 0.297 | 0.777 | 1.43 | 0.379 | 0.458 | 0.514 | 0.128 | 0.72 |  |  |
