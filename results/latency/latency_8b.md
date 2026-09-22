# Inference latency on one NVIDIA A40

Single request = one record through the served `ask` path, median over N per K. Batched = records/s at batch 8 over the same mixed set. Jev = its client-observed API round trip on the same sources (network included).

| model | tier | perms | K=2 | K=4 | K=5 | K=27 | K=60 | K=77 | K=151 | rec/s @8 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B (Tier 0) | Tier 0 | 1 | 96 ms | 89 ms | 89 ms | 100 ms | 250 ms | 337 ms | 486 ms | 5.1 |
| gemma-4-12B-it (Tier 0) | Tier 0 | 1 | 116 ms | 102 ms | 104 ms | 126 ms | 311 ms | 462 ms | 680 ms | 3.8 |
| Jev 1.13.0 (API round trip) | API | — | 188 ms | 186 ms | 189 ms | 179 ms | 194 ms | 205 ms | 219 ms | — |
