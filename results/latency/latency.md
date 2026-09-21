# Inference latency on one NVIDIA A40

Single request = one record through the served `ask` path, median over N per K. Batched = records/s at batch 16 over the same mixed set. Jev = its client-observed API round trip on the same sources (network included).

| model | tier | perms | K=2 | K=4 | K=5 | K=27 | K=60 | K=77 | K=151 | rec/s @16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (Tier 0) | Tier 0 | 1 | 86 ms | 80 ms | 81 ms | 88 ms | 143 ms | 174 ms | 234 ms | 18.4 |
| Qwen3.5-0.8B (Tier 0) | Tier 0 | 2 | 86 ms | 82 ms | 80 ms | 91 ms | 150 ms | 187 ms | 271 ms | 10.6 |
| Qwen3.5-2B (Tier 0) | Tier 0 | 1 | 88 ms | 82 ms | 82 ms | 89 ms | 144 ms | 177 ms | 238 ms | 14.6 |
| Qwen3.5-2B (Tier 0) | Tier 0 | 2 | 87 ms | 83 ms | 82 ms | 92 ms | 161 ms | 222 ms | 336 ms | 8.4 |
| Qwen3.5-4B (Tier 0) | Tier 0 | 1 | 116 ms | 109 ms | 109 ms | 119 ms | 191 ms | 267 ms | 378 ms | 6.6 |
| Qwen3.5-4B (Tier 0) | Tier 0 | 2 | 116 ms | 111 ms | 109 ms | 121 ms | 340 ms | 469 ms | 699 ms | 3.7 |
| Qwen3.5-9B (Tier 0) | Tier 0 | 1 | 119 ms | 111 ms | 111 ms | 121 ms | 252 ms | 337 ms | 487 ms | 5.0 |
| Qwen3.5-9B (Tier 0) | Tier 0 | 2 | 119 ms | 120 ms | 113 ms | 158 ms | 444 ms | 620 ms | 898 ms | 2.8 |
| Qwen3.5-2B (Tier 1) | Tier 1 | 1 | 87 ms | 81 ms | 82 ms | 89 ms | 154 ms | 196 ms | 270 ms | 14.6 |
| Qwen3.5-4B (Tier 1) | Tier 1 | 1 | 117 ms | 109 ms | 109 ms | 120 ms | 205 ms | 289 ms | 418 ms | 6.6 |
| Qwen3.5-2B (Tier 2) | Tier 2 | 1 | 88 ms | 82 ms | 82 ms | 91 ms | 155 ms | 195 ms | 274 ms | 14.5 |
| Jev 1.13.0 (API round trip) | API | — | 188 ms | 186 ms | 189 ms | 179 ms | 194 ms | 205 ms | 219 ms | — |
