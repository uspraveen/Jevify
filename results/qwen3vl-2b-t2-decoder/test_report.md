**model:** `Qwen/Qwen3-VL-2B-Instruct (Tier 2, LoRA decoder)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ai2d` | choice | 1500 | 0.707 | 0.041 | 0.387 | 0.73 | 0.746 | 0.924 | 0.104 |  |  |  |  |
| `aokvqa` | choice | 744 | 0.813 | 0.027 | 0.270 | 0.53 | 0.858 | 0.957 | 0.061 |  |  |  |  |
| `pope` | noul | 2000 | 0.893 | 0.037 | 0.078 | 0.26 | 0.926 | 0.990 | 0.024 |  |  | 0.961 |  |
