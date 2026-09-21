# Inference latency on one NVIDIA A40

Single request = one record through the served `ask` path, median over N per K. Batched = records/s at batch 16 over the same mixed set. Jev = its client-observed API round trip on the same sources (network included).

| model | tier | perms | K=2 | K=4 | K=5 | K=27 | K=60 | K=77 | K=151 | rec/s @16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B (Tier 0, vLLM, state last) | Tier 0 | 1 | 46 ms | 43 ms | 43 ms | 45 ms | 54 ms | 60 ms | 111 ms | 60.5 |
| Qwen3.5-2B (Tier 0, vLLM, state last) | Tier 0 | 2 | 44 ms | 82 ms | 43 ms | 88 ms | 107 ms | 119 ms | 163 ms | 40.9 |
| Jev 1.13.0 (API round trip) | API | — | 188 ms | 186 ms | 189 ms | 179 ms | 194 ms | 205 ms | 219 ms | — |
