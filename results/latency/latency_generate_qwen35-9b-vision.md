# Readout vs generation on one NVIDIA A40 — Qwen/Qwen3.5-9B

Same records, same prompt (chat template included), one request at a time, medians. The readout is Jevify's `ask`; `prefill` is one forward pass over the prompt with no answer read; `generate N` is Hugging Face greedy decoding of N tokens.

| source | K | prompt tokens | readout | prefill only | generate 1 | generate 8 | generate 32 | per generated token |
|---|---|---|---|---|---|---|---|---|
| pope | 2 | 360 | **171 ms** | 163 ms | 169 ms | 539 ms | 1807 ms | 52.8 ms |
| aokvqa | 4 | 360 | **168 ms** | 161 ms | 166 ms | 537 ms | 1806 ms | 52.9 ms |
