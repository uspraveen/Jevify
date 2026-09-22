# Qwen/Qwen3.5-4B (Tier 1): Tier 1 vs Tier 0 vs Jev

Heads trained on 8,685 records from 16 sources (max 16 options per record), early-stopped on 2,250 validation records; epoch 7.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B (Tier 1) | Tier 1 heads | 0.700 | 0.142 | 0.411 | 0.690 | 0.073 | 0.387 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-4B | Tier 0 (recipe refit w/o held-out) | 0.714 | 0.139 | 0.355 | 0.641 | 0.087 | 0.424 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-4B (Tier 1)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.864 | 0.036 | 0.196 | 0.39 | 0.909 | 0.986 | 0.030 |  |  |  |  |
| `banking77` | choice | 1000 | 0.753 | 0.089 | 0.376 | 1.08 | 0.797 | 0.928 | 0.092 |  |  |  |  |
| `boolq` | noul | 1000 | 0.903 | 0.025 | 0.073 | 0.24 | 0.933 | 0.990 | 0.023 |  |  | 0.962 |  |
| `chaosnli` | choice | 1599 | 0.651 | 0.073 | 0.454 | 0.77 | 0.679 | 0.777 | 0.206 |  |  |  | 0.253 |
| `civil_comments` | noul | 2000 | 0.913 | 0.027 | 0.070 | 0.24 | 0.942 | 0.986 | 0.024 |  |  | 0.810 | 0.113 |
| `clinc150` | choice | 1000 | 0.478 | 0.273 | 0.747 | 2.34 | 0.513 | 0.682 | 0.282 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.943 | 0.027 | 0.050 | 0.19 | 0.964 | 0.982 | 0.021 |  |  | 0.981 |  |
| `go_emotions` | choice | 1000 | 0.431 | 0.084 | 0.758 | 2.05 | 0.452 | 0.524 | 0.462 |  |  |  | 0.597 |
| `helpsteer2_helpfulness` | score | 1000 | 0.377 | 0.144 | 0.747 | 1.46 | 0.382 | 0.440 | 0.545 | 0.159 | 0.96 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.647 | 0.051 | 0.519 | 0.97 | 0.669 | 0.710 | 0.293 | 0.078 | 0.45 |  |  |
| `ledgar` | choice | 1000 | 0.714 | 0.118 | 0.425 | 1.23 | 0.763 | 0.898 | 0.116 |  |  |  |  |
| `massive` | choice | 1000 | 0.749 | 0.070 | 0.350 | 1.00 | 0.801 | 0.954 | 0.083 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.510 | 0.205 | 0.681 | 1.26 | 0.520 | 0.616 | 0.366 | 0.299 | 0.84 |  | 0.448 |
| `mmlu` | choice | 1000 | 0.625 | 0.170 | 0.535 | 1.13 | 0.663 | 0.802 | 0.181 |  |  |  |  |
| `mnli` | choice | 1000 | 0.834 | 0.023 | 0.245 | 0.45 | 0.870 | 0.966 | 0.051 |  |  |  |  |
| `paws` | noul | 1000 | 0.831 | 0.023 | 0.123 | 0.38 | 0.853 | 0.942 | 0.063 |  |  | 0.908 |  |
| `sms_spam` | noul | 800 | 0.966 | 0.028 | 0.029 | 0.11 | 0.981 | 0.995 | 0.005 |  |  | 0.978 |  |
| `sst5` | score | 1000 | 0.479 | 0.128 | 0.639 | 1.17 | 0.501 | 0.572 | 0.423 | 0.100 | 0.59 |  |  |
| `strategyqa_closed` | noul | 687 | 0.691 | 0.033 | 0.203 | 0.59 | 0.709 | 0.773 | 0.206 |  |  | 0.747 |  |
| `strategyqa_grounded` | noul | 687 | 0.881 | 0.084 | 0.095 | 0.31 | 0.917 | 0.991 | 0.029 |  |  | 0.955 |  |
| `stsb` | score | 1000 | 0.467 | 0.084 | 0.645 | 1.21 | 0.496 | 0.538 | 0.421 | 0.082 | 0.62 |  |  |
| `yelp5` | score | 1000 | 0.522 | 0.226 | 0.699 | 1.35 | 0.520 | 0.522 | 0.480 | 0.111 | 0.62 |  |  |
