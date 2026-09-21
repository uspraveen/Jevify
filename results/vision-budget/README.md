# Vision encoder budget sweep — Qwen/Qwen3-VL-2B-Instruct

Same records, same prompt, same decoder; only the pixel budget moves.
`tokens` is the median image-token count the budget actually produced.

| budget (patches) | tokens | ai2d acc / ECE | aokvqa acc / ECE | pope acc / ECE | macro acc | macro ECE |
|---|---|---|---|---|---|---|
| 64 | 42 | 0.593 / 0.225 | 0.705 / 0.195 | 0.875 / 0.109 | 0.724 | 0.176 |
| 128 | 88 | 0.615 / 0.217 | 0.748 / 0.155 | 0.892 / 0.092 | 0.752 | 0.155 |
| 256 | 187 | 0.623 / 0.208 | 0.785 / 0.136 | 0.907 / 0.073 | 0.772 | 0.139 |
| 512 | 280 | 0.630 / 0.191 | 0.777 / 0.138 | 0.897 / 0.081 | 0.768 | 0.137 |
| 1024 | 280 | 0.637 / 0.195 | 0.780 / 0.136 | 0.897 / 0.082 | 0.772 | 0.137 |
