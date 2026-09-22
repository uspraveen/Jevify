## validation

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.8091 | 0.0616 | 0.2326 | 0.7817 | 0.0547 |  |  | 0.864 | 0.0754 |
| +permutations | 0.8091 | 0.0616 | 0.2326 | 0.7817 | 0.0547 |  |  | 0.864 | 0.0754 |
| +prior | 0.8091 | 0.0616 | 0.2326 | 0.7817 | 0.0547 |  |  | 0.864 | 0.0754 |
| +temperature/bias | 0.8111 | 0.0464 | 0.2287 | 0.7817 | 0.0487 |  |  | 0.87 | 0.0417 |

## test

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.7999 | 0.0471 | 0.2475 | 0.7576 | 0.0473 |  |  | 0.8845 | 0.0467 |
| +permutations | 0.7999 | 0.0471 | 0.2475 | 0.7576 | 0.0473 |  |  | 0.8845 | 0.0467 |
| +prior | 0.7999 | 0.0471 | 0.2475 | 0.7576 | 0.0473 |  |  | 0.8845 | 0.0467 |
| +temperature/bias | 0.7986 | 0.0374 | 0.2455 | 0.7576 | 0.04 |  |  | 0.8805 | 0.0322 |
