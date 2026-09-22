**model:** `Qwen/Qwen3-VL-2B-Instruct (Tier 2, LoRA vision)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ai2d` | choice | 1500 | 0.669 | 0.057 | 0.436 | 0.82 | 0.712 | 0.872 | 0.136 |  |  |  |  |
| `aokvqa` | choice | 744 | 0.792 | 0.046 | 0.288 | 0.57 | 0.848 | 0.957 | 0.075 |  |  |  |  |
| `pope` | noul | 2000 | 0.887 | 0.030 | 0.080 | 0.27 | 0.926 | 0.980 | 0.027 |  |  | 0.956 |  |
