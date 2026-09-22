# Vision latency on one NVIDIA A40 — google/gemma-4-12B-it

One record at a time through the served `ask` path, median of N real records; the image goes through the processor and the answer is read from one position.

| source | primitive | K | p50 ms | p90 ms | input tokens | of which image |
|---|---|---|---|---|---|---|
| pope | noul | 2 | 173 | 174 | 338 | 1 |
| aokvqa | choice | 4 | 178 | 181 | 368 | 1 |
| ai2d | choice | 4 | 179 | 194 | 368 | 1 |

Batched: **7.0 records/s** at batch 4, mixed sources.
