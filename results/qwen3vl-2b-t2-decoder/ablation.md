## validation

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.8068 | 0.05 | 0.2353 | 0.7772 | 0.0424 |  |  | 0.866 | 0.065 |
| +permutations | 0.8068 | 0.05 | 0.2353 | 0.7772 | 0.0424 |  |  | 0.866 | 0.065 |
| +prior | 0.8068 | 0.05 | 0.2353 | 0.7772 | 0.0424 |  |  | 0.866 | 0.065 |
| +temperature/bias | 0.8081 | 0.0471 | 0.2304 | 0.7772 | 0.0429 |  |  | 0.87 | 0.0555 |

## test

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.8043 | 0.0348 | 0.2476 | 0.7599 | 0.0334 |  |  | 0.893 | 0.0377 |
| +permutations | 0.8043 | 0.0348 | 0.2476 | 0.7599 | 0.0334 |  |  | 0.893 | 0.0377 |
| +prior | 0.8043 | 0.0348 | 0.2476 | 0.7599 | 0.0334 |  |  | 0.893 | 0.0377 |
| +temperature/bias | 0.8043 | 0.035 | 0.245 | 0.7599 | 0.034 |  |  | 0.893 | 0.0369 |
