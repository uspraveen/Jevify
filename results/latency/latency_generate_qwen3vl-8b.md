# Readout vs generation on one NVIDIA A40 — Qwen/Qwen3-VL-8B-Instruct

Same records, same prompt (chat template included), one request at a time, medians. The readout is Jevify's `ask`; `prefill` is one forward pass over the prompt with no answer read; `generate N` is Hugging Face greedy decoding of N tokens.

| source | K | prompt tokens | readout | prefill only | generate 1 | generate 8 | generate 32 | per generated token |
|---|---|---|---|---|---|---|---|---|
| pope | 2 | 352 | **128 ms** | 119 ms | 126 ms | 499 ms | 1784 ms | 53.5 ms |
| aokvqa | 4 | 351 | **126 ms** | 117 ms | 124 ms | 498 ms | 1783 ms | 53.5 ms |
