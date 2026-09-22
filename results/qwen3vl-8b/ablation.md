## validation

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.8244 | 0.1185 | 0.2343 | 0.8056 | 0.1187 |  |  | 0.862 | 0.118 |
| +permutations | 0.8244 | 0.1185 | 0.2343 | 0.8056 | 0.1187 |  |  | 0.862 | 0.118 |
| +prior | 0.8244 | 0.1185 | 0.2343 | 0.8056 | 0.1187 |  |  | 0.862 | 0.118 |
| +temperature/bias | 0.8251 | 0.0542 | 0.2009 | 0.8056 | 0.057 |  |  | 0.864 | 0.0486 |

## test

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.8405 | 0.1089 | 0.2304 | 0.8178 | 0.1154 |  |  | 0.886 | 0.096 |
| +permutations | 0.8405 | 0.1089 | 0.2304 | 0.8178 | 0.1154 |  |  | 0.886 | 0.096 |
| +prior | 0.8405 | 0.1089 | 0.2304 | 0.8178 | 0.1154 |  |  | 0.886 | 0.096 |
| +temperature/bias | 0.8417 | 0.0319 | 0.199 | 0.8178 | 0.0353 |  |  | 0.8895 | 0.0253 |
