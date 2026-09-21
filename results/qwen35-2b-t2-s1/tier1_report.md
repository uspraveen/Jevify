# Qwen/Qwen3.5-2B (Tier 2): Tier 2 vs Tier 0 vs Jev

Heads trained on 5,885 records from 16 sources (max 16 options per record), early-stopped on 1,500 validation records; epoch 1.

**Held-out sources** (never seen in training): `clinc150`, `arc_challenge`, `yelp5`, `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`. `chaosnli` has no train split, so it is held out by construction.

| model | variant | held-out acc | held-out ECE | held-out Brier | trained acc | trained ECE | trained Brier |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B (Tier 2) | Tier 2 heads + LoRA | 0.701 | 0.116 | 0.367 | 0.695 | 0.122 | 0.408 |
| Jev 1.13.0 | API (zero-shot) | 0.835 | 0.090 | 0.235 | 0.694 | 0.122 | 0.391 |
| Qwen/Qwen3.5-2B | Tier 0 (recipe refit w/o held-out) | 0.628 | 0.097 | 0.436 | 0.552 | 0.100 | 0.506 |

## Per-config (Tier 1)

**model:** `Qwen/Qwen3.5-2B (Tier 2)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arc_challenge` | choice | 1000 | 0.827 | 0.099 | 0.277 | 0.59 | 0.867 | 0.960 | 0.054 |  |  |  |  |
| `banking77` | choice | 1000 | 0.760 | 0.147 | 0.399 | 1.28 | 0.801 | 0.904 | 0.113 |  |  |  |  |
| `boolq` | noul | 1000 | 0.863 | 0.098 | 0.113 | 0.43 | 0.902 | 0.970 | 0.044 |  |  | 0.934 |  |
| `chaosnli` | choice | 1599 | 0.533 | 0.376 | 0.798 | 1.76 | 0.551 | 0.650 | 0.329 |  |  |  | 0.438 |
| `civil_comments` | noul | 2000 | 0.924 | 0.039 | 0.059 | 0.22 | 0.961 | 0.990 | 0.016 |  |  | 0.868 | 0.091 |
| `clinc150` | choice | 1000 | 0.687 | 0.160 | 0.475 | 1.68 | 0.744 | 0.912 | 0.113 |  |  |  |  |
| `fever_evidence` | noul | 1000 | 0.908 | 0.033 | 0.069 | 0.23 | 0.946 | 0.994 | 0.017 |  |  | 0.971 |  |
| `go_emotions` | choice | 1000 | 0.535 | 0.214 | 0.692 | 1.90 | 0.562 | 0.690 | 0.310 |  |  |  | 0.538 |
| `helpsteer2_helpfulness` | score | 1000 | 0.404 | 0.040 | 0.697 | 1.37 | 0.413 | 0.460 | 0.543 | 0.153 | 0.93 |  |  |
| `helpsteer2_verbosity` | score | 1000 | 0.644 | 0.088 | 0.524 | 0.99 | 0.661 | 0.686 | 0.329 | 0.081 | 0.54 |  |  |
| `ledgar` | choice | 1000 | 0.717 | 0.143 | 0.429 | 1.22 | 0.759 | 0.924 | 0.122 |  |  |  |  |
| `massive` | choice | 1000 | 0.802 | 0.114 | 0.323 | 1.03 | 0.851 | 0.950 | 0.070 |  |  |  |  |
| `measuring_hate_speech` | score | 1000 | 0.582 | 0.090 | 0.533 | 0.98 | 0.596 | 0.718 | 0.262 | 0.205 | 0.69 |  | 0.398 |
| `mmlu` | choice | 1000 | 0.587 | 0.235 | 0.609 | 1.27 | 0.628 | 0.798 | 0.199 |  |  |  |  |
| `mnli` | choice | 1000 | 0.830 | 0.114 | 0.287 | 0.62 | 0.868 | 0.952 | 0.062 |  |  |  |  |
| `paws` | noul | 1000 | 0.868 | 0.097 | 0.112 | 0.42 | 0.901 | 0.984 | 0.040 |  |  | 0.955 |  |
| `sms_spam` | noul | 800 | 0.974 | 0.021 | 0.025 | 0.13 | 0.986 | 0.998 | 0.004 |  |  | 0.985 |  |
| `sst5` | score | 1000 | 0.521 | 0.045 | 0.601 | 1.08 | 0.539 | 0.570 | 0.413 | 0.098 | 0.61 |  |  |
| `strategyqa_closed` | noul | 687 | 0.672 | 0.136 | 0.226 | 0.68 | 0.693 | 0.776 | 0.211 |  |  | 0.761 |  |
| `strategyqa_grounded` | noul | 687 | 0.805 | 0.103 | 0.144 | 0.46 | 0.833 | 0.951 | 0.071 |  |  | 0.918 |  |
| `stsb` | score | 1000 | 0.482 | 0.050 | 0.637 | 1.20 | 0.490 | 0.532 | 0.460 | 0.090 | 0.70 |  |  |
| `yelp5` | score | 1000 | 0.398 | 0.209 | 0.704 | 1.21 | 0.409 | 0.426 | 0.601 | 0.111 | 0.67 |  |  |
