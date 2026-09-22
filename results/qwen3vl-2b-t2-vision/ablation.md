## validation

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.7728 | 0.0967 | 0.2783 | 0.7322 | 0.099 |  |  | 0.854 | 0.0919 |
| +permutations | 0.7728 | 0.0967 | 0.2783 | 0.7322 | 0.099 |  |  | 0.854 | 0.0919 |
| +prior | 0.7728 | 0.0967 | 0.2783 | 0.7322 | 0.099 |  |  | 0.854 | 0.0919 |
| +temperature/bias | 0.7821 | 0.068 | 0.263 | 0.7322 | 0.0828 |  |  | 0.882 | 0.0386 |

## test

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.78 | 0.0716 | 0.2803 | 0.7305 | 0.0747 |  |  | 0.879 | 0.0654 |
| +permutations | 0.78 | 0.0716 | 0.2803 | 0.7305 | 0.0747 |  |  | 0.879 | 0.0654 |
| +prior | 0.78 | 0.0716 | 0.2803 | 0.7305 | 0.0747 |  |  | 0.879 | 0.0654 |
| +temperature/bias | 0.7828 | 0.0447 | 0.268 | 0.7305 | 0.0518 |  |  | 0.8875 | 0.0304 |
