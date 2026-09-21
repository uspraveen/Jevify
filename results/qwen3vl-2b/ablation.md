## validation

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.7711 | 0.1381 | 0.3065 | 0.7287 | 0.1463 |  |  | 0.856 | 0.1216 |
| +permutations | 0.7711 | 0.1381 | 0.3065 | 0.7287 | 0.1463 |  |  | 0.856 | 0.1216 |
| +prior | 0.7711 | 0.1381 | 0.3065 | 0.7287 | 0.1463 |  |  | 0.856 | 0.1216 |
| +temperature/bias | 0.7725 | 0.0788 | 0.2665 | 0.7287 | 0.0856 |  |  | 0.86 | 0.065 |

## test

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.7803 | 0.1283 | 0.3116 | 0.7225 | 0.1494 |  |  | 0.896 | 0.0862 |
| +permutations | 0.7803 | 0.1283 | 0.3116 | 0.7225 | 0.1494 |  |  | 0.896 | 0.0862 |
| +prior | 0.7803 | 0.1283 | 0.3116 | 0.7225 | 0.1494 |  |  | 0.896 | 0.0862 |
| +temperature/bias | 0.7787 | 0.0473 | 0.2774 | 0.7225 | 0.0478 |  |  | 0.891 | 0.0462 |
