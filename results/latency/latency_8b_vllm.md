# Inference latency on one NVIDIA A40

Single request = one record through the served `ask` path, median over N per K. Batched = records/s at batch 8 over the same mixed set. Jev = its client-observed API round trip on the same sources (network included).

| model | tier | perms | K=2 | K=4 | K=5 | K=27 | K=60 | K=77 | K=151 | rec/s @8 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B (Tier 0, vLLM) | Tier 0 | 1 | 68 ms | 56 ms | 57 ms | 70 ms | 179 ms | 242 ms | 321 ms | 9.1 |
| Jev 1.13.0 (API round trip) | API | — | 188 ms | 186 ms | 189 ms | 179 ms | 194 ms | 205 ms | 219 ms | — |
