## validation

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.7843 | 0.1369 | 0.2924 | 0.7545 | 0.1433 |  |  | 0.844 | 0.1241 |
| +permutations | 0.7843 | 0.1369 | 0.2924 | 0.7545 | 0.1433 |  |  | 0.844 | 0.1241 |
| +prior | 0.7843 | 0.1369 | 0.2924 | 0.7545 | 0.1433 |  |  | 0.844 | 0.1241 |
| +temperature/bias | 0.789 | 0.0486 | 0.2568 | 0.7545 | 0.0488 |  |  | 0.858 | 0.0482 |

## test

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.8071 | 0.1257 | 0.2716 | 0.7856 | 0.1284 |  |  | 0.85 | 0.1202 |
| +permutations | 0.8071 | 0.1257 | 0.2716 | 0.7856 | 0.1284 |  |  | 0.85 | 0.1202 |
| +prior | 0.8071 | 0.1257 | 0.2716 | 0.7856 | 0.1284 |  |  | 0.85 | 0.1202 |
| +temperature/bias | 0.8131 | 0.0463 | 0.239 | 0.7856 | 0.0517 |  |  | 0.868 | 0.0354 |
