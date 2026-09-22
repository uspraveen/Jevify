## validation

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.8281 | 0.0591 | 0.2101 | 0.8021 | 0.056 |  |  | 0.88 | 0.0651 |
| +permutations | 0.8281 | 0.0591 | 0.2101 | 0.8021 | 0.056 |  |  | 0.88 | 0.0651 |
| +prior | 0.8281 | 0.0591 | 0.2101 | 0.8021 | 0.056 |  |  | 0.88 | 0.0651 |
| +temperature/bias | 0.8288 | 0.054 | 0.2072 | 0.8021 | 0.0557 |  |  | 0.882 | 0.0507 |

## test

| step | macro_acc | macro_ece | macro_brier | choice_acc | choice_ece | score_acc | score_ece | noul_acc | noul_ece |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.8413 | 0.0447 | 0.1956 | 0.8132 | 0.045 |  |  | 0.8975 | 0.0442 |
| +permutations | 0.8413 | 0.0447 | 0.1956 | 0.8132 | 0.045 |  |  | 0.8975 | 0.0442 |
| +prior | 0.8413 | 0.0447 | 0.1956 | 0.8132 | 0.045 |  |  | 0.8975 | 0.0442 |
| +temperature/bias | 0.8406 | 0.0393 | 0.1939 | 0.8132 | 0.0438 |  |  | 0.8955 | 0.0303 |
