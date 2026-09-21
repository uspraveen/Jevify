## validation

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.7287 | 0.1463 | 0.3978 | 0.7287 | 0.1463 |  |  |  |  |
| +permutations | 0.7287 | 0.1463 | 0.3978 | 0.7287 | 0.1463 |  |  |  |  |
| +prior | 0.7287 | 0.1463 | 0.3978 | 0.7287 | 0.1463 |  |  |  |  |
| +temperature/bias | 0.7287 | 0.0856 | 0.3527 | 0.7287 | 0.0856 |  |  |  |  |

## test

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.7815 | 0.1274 | 0.3104 | 0.7225 | 0.1494 |  |  | 0.8995 | 0.0832 |
| +permutations | 0.7815 | 0.1274 | 0.3104 | 0.7225 | 0.1494 |  |  | 0.8995 | 0.0832 |
| +prior | 0.7815 | 0.1274 | 0.3104 | 0.7225 | 0.1494 |  |  | 0.8995 | 0.0832 |
| +temperature/bias | 0.7815 | 0.0596 | 0.2811 | 0.7225 | 0.0478 |  |  | 0.8995 | 0.0832 |
