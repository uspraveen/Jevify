# Qwen/Qwen3.5-2B (Tier 2): Tier 2 vs Tier 0 vs Jev

Heads trained on 5,885 records from 16 sources (max 16 options per record), early-stopped on 1,500 validation records; epoch 0.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B (Tier 2) | Tier 2 heads + LoRA | 0.715 | 0.079 | 0.353 | 0.669 | 0.100 | 0.406 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-2B (Tier 2)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.816 | 0.026 | 0.265 | 0.51 | 0.856 | 0.968 | 0.054 |  |  |  |  |
| `banking77` | choice | 1000 | 0.719 | 0.092 | 0.439 | 1.38 | 0.763 | 0.884 | 0.122 |  |  |  |  |
| `boolq` | noul | 1000 | 0.858 | 0.070 | 0.111 | 0.40 | 0.893 | 0.952 | 0.066 |  |  | 0.925 |  |
| `chaosnli` | choice | 1599 | 0.626 | 0.159 | 0.541 | 0.98 | 0.651 | 0.739 | 0.259 |  |  |  | 0.310 |
| `civil_comments` | noul | 2000 | 0.923 | 0.029 | 0.063 | 0.23 | 0.949 | 0.985 | 0.023 |  |  | 0.815 | 0.094 |
| `clinc150` | choice | 1000 | 0.687 | 0.108 | 0.466 | 1.48 | 0.733 | 0.886 | 0.134 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.909 | 0.031 | 0.065 | 0.23 | 0.951 | 0.994 | 0.018 |  |  | 0.973 |  |
| `go_emotions` | choice | 1000 | 0.510 | 0.167 | 0.690 | 1.81 | 0.543 | 0.658 | 0.325 |  |  |  | 0.543 |
| `helpsteer2_helpfulness` | score | 1000 | 0.310 | 0.132 | 0.743 | 1.45 | 0.321 | 0.370 | 0.618 | 0.161 | 0.97 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.622 | 0.117 | 0.553 | 1.03 | 0.659 | 0.698 | 0.302 | 0.087 | 0.48 |  |  |
| `ledgar` | choice | 1000 | 0.711 | 0.071 | 0.409 | 1.10 | 0.758 | 0.890 | 0.115 |  |  |  |  |
| `massive` | choice | 1000 | 0.741 | 0.115 | 0.395 | 1.09 | 0.787 | 0.924 | 0.104 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.646 | 0.074 | 0.491 | 0.91 | 0.654 | 0.788 | 0.192 | 0.208 | 0.66 |  | 0.347 |
| `mmlu` | choice | 1000 | 0.596 | 0.073 | 0.502 | 0.95 | 0.623 | 0.810 | 0.187 |  |  |  |  |
| `mnli` | choice | 1000 | 0.829 | 0.050 | 0.265 | 0.49 | 0.858 | 0.930 | 0.068 |  |  |  |  |
| `paws` | noul | 1000 | 0.860 | 0.064 | 0.111 | 0.38 | 0.887 | 0.964 | 0.048 |  |  | 0.939 |  |
| `sms_spam` | noul | 800 | 0.959 | 0.037 | 0.037 | 0.18 | 0.971 | 0.998 | 0.007 |  |  | 0.983 |  |
| `sst5` | score | 1000 | 0.450 | 0.122 | 0.672 | 1.20 | 0.478 | 0.524 | 0.480 | 0.109 | 0.67 |  |  |
| `strategyqa_closed` | noul | 687 | 0.575 | 0.215 | 0.278 | 0.80 | 0.571 | 0.683 | 0.295 |  |  | 0.705 |  |
| `strategyqa_grounded` | noul | 687 | 0.757 | 0.118 | 0.177 | 0.55 | 0.778 | 0.919 | 0.106 |  |  | 0.911 |  |
| `stsb` | score | 1000 | 0.412 | 0.084 | 0.680 | 1.31 | 0.430 | 0.502 | 0.531 | 0.096 | 0.73 |  |  |
| `yelp5` | score | 1000 | 0.474 | 0.118 | 0.656 | 1.19 | 0.493 | 0.536 | 0.437 | 0.106 | 0.64 |  |  |
