# Vision latency on one NVIDIA A40 — Qwen/Qwen3-VL-8B-Instruct

One record at a time through the served `ask` path, median of N real records; the image goes through the processor and the answer is read from one position.

| source | primitive | K | p50 ms | p90 ms | input tokens | of which image |
|---|---|---|---|---|---|---|
| pope | noul | 2 | 123 | 126 | 352 | 300 |
| aokvqa | choice | 4 | 121 | 138 | 351 | 260 |
| ai2d | choice | 4 | 120 | 295 | 358 | 296 |

Batched: **5.3 records/s** at batch 8, mixed sources.
