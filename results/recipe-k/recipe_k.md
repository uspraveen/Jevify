# One temperature per primitive, or one that varies with the option count?

Each model's recipe refitted on its own validation predictions both ways and scored on test. `K>30` = `banking77`, `clinc150`, `ledgar`, `massive`; `K<=5` = `arc_challenge`, `chaosnli`, `mmlu`, `mnli`. The slope is kept only when it cuts validation NLL by >=1%.

| model | macro ECE scalar | macro ECE T(K) | K>30 scalar | K>30 T(K) | K<=5 scalar | K<=5 T(K) | macro acc | slope kept |
|---|---|---|---|---|---|---|---|---|
| `gemma4-12b` | 0.086 | **0.083** | 0.052 | **0.048** | 0.078 | 0.083 | 0.717 | choice -0.29 |
| `gemma4-e2b` | 0.115 | **0.115** | 0.059 | **0.059** | 0.089 | 0.089 | 0.396 | — |
| `gemma4-e2b-it` | 0.123 | **0.103** | 0.099 | **0.063** | 0.180 | 0.124 | 0.591 | choice -0.58 |
| `gemma4-e4b-it` | 0.092 | **0.090** | 0.065 | **0.040** | 0.061 | 0.090 | 0.658 | choice -0.42 |
| `k2-horizon-0.9b` | 0.119 | **0.105** | 0.129 | **0.114** | 0.110 | 0.062 | 0.448 | choice -0.21 |
| `k2-horizon-7b` | 0.093 | **0.093** | 0.072 | **0.072** | 0.120 | 0.120 | 0.623 | — |
| `olmo3-7b-it` | 0.091 | **0.108** | 0.099 | **0.106** | 0.068 | 0.145 | 0.553 | choice -0.58 |
| `qwen35-0.8b` | 0.112 | **0.112** | 0.131 | **0.114** | 0.059 | 0.100 | 0.526 | choice -0.21 |
| `qwen35-0.8b-base` | 0.105 | **0.105** | 0.113 | **0.113** | 0.068 | 0.068 | 0.466 | — |
| `qwen35-2b` | 0.089 | **0.089** | 0.076 | **0.076** | 0.072 | 0.072 | 0.577 | — |
| `qwen35-4b` | 0.090 | **0.090** | 0.074 | **0.074** | 0.088 | 0.088 | 0.662 | — |
| `qwen35-9b` | 0.092 | **0.085** | 0.054 | **0.064** | 0.142 | 0.088 | 0.690 | choice +0.08 |
| `smollm3-3b` | 0.111 | **0.111** | 0.118 | **0.118** | 0.059 | 0.059 | 0.552 | — |
