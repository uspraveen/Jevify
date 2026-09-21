# Vision latency on one NVIDIA A40 — Qwen/Qwen3-VL-2B-Instruct

One record at a time through the served `ask` path, median of N real records; the image goes through the processor and the answer is read from one position.

| source | primitive | K | p50 ms | p90 ms | input tokens | of which image |
|---|---|---|---|---|---|---|
| pope | noul | 2 | 67 | 78 | 352 | 260 |
| aokvqa | choice | 4 | 66 | 69 | 351 | 260 |
| ai2d | choice | 4 | 67 | 142 | 358 | 296 |

Batched: **12.8 records/s** at batch 8, mixed sources.
