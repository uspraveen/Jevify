# Inference latency on one NVIDIA A40

Single request = one record through the served `ask` path, median over N per K. Batched = records/s at batch 16 over the same mixed set. Jev = its client-observed API round trip on the same sources (network included).

| model | tier | perms | K=2 | K=4 | K=5 | K=27 | K=60 | K=77 | K=151 | rec/s @16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (Tier 0, vLLM) | Tier 0 | 1 | 43 ms | 42 ms | 42 ms | 44 ms | 92 ms | 98 ms | 108 ms | 66.2 |
| Qwen3.5-0.8B (Tier 0, vLLM) | Tier 0 | 2 | 44 ms | 82 ms | 42 ms | 86 ms | 140 ms | 153 ms | 159 ms | 50.2 |
| Qwen3.5-2B (Tier 0, vLLM) | Tier 0 | 1 | 49 ms | 46 ms | 46 ms | 48 ms | 100 ms | 101 ms | 112 ms | 58.1 |
| Qwen3.5-2B (Tier 0, vLLM) | Tier 0 | 2 | 44 ms | 81 ms | 43 ms | 86 ms | 145 ms | 157 ms | 163 ms | 40.8 |
| Qwen3.5-4B (Tier 0, vLLM) | Tier 0 | 1 | 50 ms | 49 ms | 49 ms | 51 ms | 113 ms | 161 ms | 210 ms | 33.6 |
| Qwen3.5-4B (Tier 0, vLLM) | Tier 0 | 2 | 50 ms | 93 ms | 49 ms | 96 ms | 165 ms | 240 ms | 302 ms | 10.7 |
| Jev 1.13.0 (API round trip) | API | — | 188 ms | 186 ms | 189 ms | 179 ms | 194 ms | 205 ms | 219 ms | — |
