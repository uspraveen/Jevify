**model:** `Qwen/Qwen3-VL-2B-Instruct (Tier 2, LoRA both)`

| source | prim | n | acc | ECE | Brier | NLL | sel@90 | sel@50 | AURC | RPS | MAE | AUROC | TVD→human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ai2d` | choice | 1500 | 0.697 | 0.040 | 0.390 | 0.73 | 0.740 | 0.921 | 0.106 |  |  |  |  |
| `aokvqa` | choice | 744 | 0.819 | 0.040 | 0.262 | 0.52 | 0.866 | 0.962 | 0.060 |  |  |  |  |
| `pope` | noul | 2000 | 0.880 | 0.032 | 0.085 | 0.28 | 0.920 | 0.982 | 0.030 |  |  | 0.953 |  |
