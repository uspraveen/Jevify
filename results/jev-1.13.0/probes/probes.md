# Behavioral probes

## Cardinality: same items, gold kept, K options

| source | K | accuracy | ECE | mean P(gold) | mean confidence |
|---|---|---|---|---|---|
| `banking77` | 2 | 0.995 | 0.009 | 0.988 | 0.990 |
| `banking77` | 5 | 0.970 | 0.027 | 0.959 | 0.976 |
| `banking77` | 10 | 0.935 | 0.053 | 0.926 | 0.966 |
| `banking77` | 25 | 0.865 | 0.073 | 0.852 | 0.936 |
| `banking77` | 50 | 0.835 | 0.105 | 0.803 | 0.910 |
| `banking77` | 77 | 0.820 | 0.085 | 0.782 | 0.899 |
| `banking77` | 100 | 0.810 | 0.104 | 0.783 | 0.900 |
| `clinc150` | 2 | 0.995 | 0.008 | 0.990 | 0.993 |
| `clinc150` | 5 | 0.995 | 0.011 | 0.989 | 0.990 |
| `clinc150` | 10 | 0.995 | 0.009 | 0.989 | 0.992 |
| `clinc150` | 25 | 0.980 | 0.018 | 0.963 | 0.977 |
| `clinc150` | 50 | 0.970 | 0.042 | 0.934 | 0.954 |
| `clinc150` | 100 | 0.945 | 0.033 | 0.912 | 0.946 |
| `clinc150` | 151 | 0.910 | 0.049 | 0.872 | 0.921 |
| `go_emotions` | 2 | 0.850 | 0.070 | 0.817 | 0.906 |
| `go_emotions` | 5 | 0.670 | 0.150 | 0.604 | 0.793 |
| `go_emotions` | 10 | 0.470 | 0.297 | 0.417 | 0.750 |
| `go_emotions` | 25 | 0.325 | 0.344 | 0.275 | 0.666 |
| `go_emotions` | 28 | 0.300 | 0.345 | 0.264 | 0.645 |
| `go_emotions` | 50 | 0.300 | 0.350 | 0.264 | 0.645 |
| `go_emotions` | 100 | 0.300 | 0.344 | 0.265 | 0.644 |
| `ledgar` | 2 | 1.000 | 0.003 | 0.997 | 0.997 |
| `ledgar` | 5 | 0.975 | 0.022 | 0.971 | 0.986 |
| `ledgar` | 10 | 0.945 | 0.030 | 0.934 | 0.959 |
| `ledgar` | 25 | 0.875 | 0.065 | 0.869 | 0.937 |
| `ledgar` | 50 | 0.855 | 0.060 | 0.813 | 0.901 |
| `ledgar` | 100 | 0.775 | 0.098 | 0.739 | 0.857 |
| `massive` | 2 | 0.995 | 0.011 | 0.987 | 0.991 |
| `massive` | 5 | 0.980 | 0.031 | 0.963 | 0.978 |
| `massive` | 10 | 0.935 | 0.056 | 0.918 | 0.962 |
| `massive` | 25 | 0.890 | 0.070 | 0.877 | 0.937 |
| `massive` | 50 | 0.845 | 0.068 | 0.824 | 0.903 |
| `massive` | 60 | 0.815 | 0.093 | 0.801 | 0.904 |
| `massive` | 100 | 0.810 | 0.097 | 0.801 | 0.905 |

## Label renaming: opaque option keys

| source | variant | accuracy | ECE | mean P(gold) |
|---|---|---|---|---|
| `banking77` | opaque_keys | 0.810 | 0.104 | 0.775 |
| `banking77` | opaque_no_desc | 0.020 | 0.298 | 0.015 |
| `clinc150` | opaque_keys | 0.915 | 0.056 | 0.869 |
| `clinc150` | opaque_no_desc | 0.005 | 0.250 | 0.005 |
| `massive` | opaque_keys | 0.805 | 0.097 | 0.788 |
| `massive` | opaque_no_desc | 0.015 | 0.377 | 0.016 |

## Order sensitivity: original vs shuffled option order

| source | comparison | n | argmax flip rate | mean TVD between answers | P on distractors |
|---|---|---|---|---|---|
| `arc_challenge` | original->shuffled | 200 | 0.005 | 0.008 |  |
| `banking77` | original->shuffled | 200 | 0.050 | 0.046 |  |
| `chaosnli` | original->shuffled | 200 | 0.045 | 0.030 |  |
| `clinc150` | original->shuffled | 200 | 0.040 | 0.041 |  |
| `go_emotions` | original->shuffled | 200 | 0.130 | 0.107 |  |
| `ledgar` | original->shuffled | 200 | 0.075 | 0.068 |  |
| `mmlu` | original->shuffled | 200 | 0.000 | 0.020 |  |
| `mnli` | original->shuffled | 200 | 0.025 | 0.023 |  |

| config | accuracy | ECE |
|---|---|---|
| `arc_challenge` / original | 0.980 | 0.017 |
| `arc_challenge` / shuffled | 0.985 | 0.017 |
| `banking77` / original | 0.820 | 0.097 |
| `banking77` / shuffled | 0.825 | 0.093 |
| `chaosnli` / original | 0.585 | 0.247 |
| `chaosnli` / shuffled | 0.570 | 0.253 |
| `clinc150` / original | 0.905 | 0.043 |
| `clinc150` / shuffled | 0.925 | 0.041 |
| `go_emotions` / original | 0.300 | 0.343 |
| `go_emotions` / shuffled | 0.290 | 0.353 |
| `ledgar` / original | 0.775 | 0.090 |
| `ledgar` / shuffled | 0.760 | 0.111 |
| `mmlu` / original | 0.930 | 0.048 |
| `mmlu` / shuffled | 0.930 | 0.024 |
| `mnli` / original | 0.865 | 0.058 |
| `mnli` / shuffled | 0.850 | 0.058 |

## Distractor injection: three irrelevant options added

| source | comparison | n | argmax flip rate | mean TVD between answers | P on distractors |
|---|---|---|---|---|---|
| `arc_challenge` | original->injected | 200 | 0.010 | 0.006 | 0.01295 |
| `mmlu` | original->injected | 200 | 0.005 | 0.018 | 0.0333 |
| `mnli` | original->injected | 200 | 0.015 | 0.018 | 0.0045000000000000005 |

| config | accuracy | ECE |
|---|---|---|
| `arc_challenge` / injected | 0.975 | 0.023 |
| `arc_challenge` / original | 0.990 | 0.017 |
| `mmlu` / injected | 0.915 | 0.039 |
| `mmlu` / original | 0.930 | 0.036 |
| `mnli` / injected | 0.845 | 0.072 |
| `mnli` / original | 0.855 | 0.057 |

## Primitive ablation: the same question through a different primitive

| source | comparison | n | argmax flip rate | mean TVD between answers | P on distractors |
|---|---|---|---|---|---|
| `boolq` | noul->choice_yes_no | 200 | 0.010 | 0.047 |  |
| `civil_comments` | noul->choice_yes_no | 200 | 0.050 | 0.088 |  |
| `helpsteer2_helpfulness` | score->choice_levels | 200 | 0.110 | 0.069 |  |
| `paws` | noul->choice_yes_no | 200 | 0.045 | 0.050 |  |
| `sst5` | score->choice_levels | 200 | 0.050 | 0.049 |  |
| `yelp5` | score->choice_levels | 200 | 0.045 | 0.053 |  |

| config | accuracy | ECE |
|---|---|---|
| `boolq` / choice_yes_no | 0.940 | 0.054 |
| `boolq` / noul | 0.930 | 0.028 |
| `civil_comments` / choice_yes_no | 0.730 | 0.126 |
| `civil_comments` / noul | 0.745 | 0.056 |
| `helpsteer2_helpfulness` / choice_levels | 0.340 | 0.246 |
| `helpsteer2_helpfulness` / score | 0.355 | 0.233 |
| `paws` / choice_yes_no | 0.875 | 0.072 |
| `paws` / noul | 0.855 | 0.052 |
| `sst5` / choice_levels | 0.555 | 0.223 |
| `sst5` / score | 0.580 | 0.182 |
| `yelp5` / choice_levels | 0.580 | 0.255 |
| `yelp5` / score | 0.605 | 0.244 |
