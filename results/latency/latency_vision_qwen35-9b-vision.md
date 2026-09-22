# Vision latency on one NVIDIA A40 — Qwen/Qwen3.5-9B

One record at a time through the served `ask` path, median of N real records; the image goes through the processor and the answer is read from one position.

| source | primitive | K | p50 ms | p90 ms | input tokens | of which image |
|---|---|---|---|---|---|---|
| pope | noul | 2 | 171 | 177 | 360 | 300 |
| aokvqa | choice | 4 | 169 | 190 | 360 | 260 |
| ai2d | choice | 4 | 172 | 398 | 366 | 296 |

Batched: **3.9 records/s** at batch 8, mixed sources.
