# Vision latency on one NVIDIA A40 — Qwen/Qwen3-VL-2B-Instruct

One record at a time through the served `ask` path, median of N real records; the image goes through the processor and the answer is read from one position.

| source | primitive | K | p50 ms | p90 ms | input tokens | of which image |
|---|---|---|---|---|---|---|
| pope | noul | 2 | 68 | 80 | 352 | 300 |
| aokvqa | choice | 4 | 79 | 83 | 351 | 260 |
| ai2d | choice | 4 | 79 | 147 | 358 | 296 |

Batched: **12.6 records/s** at batch 8, mixed sources.
