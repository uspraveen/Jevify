# Readout vs generation on one NVIDIA A40 — google/gemma-4-12B-it

Same records, same prompt (chat template included), one request at a time, medians. The readout is Jevify's `ask`; `prefill` is one forward pass over the prompt with no answer read; `generate N` is Hugging Face greedy decoding of N tokens.

| source | K | prompt tokens | readout | prefill only | generate 1 | generate 8 | generate 32 | per generated token |
|---|---|---|---|---|---|---|---|---|
| pope | 2 | 338 | **174 ms** | 154 ms | 161 ms | 720 ms | 2639 ms | 79.9 ms |
| aokvqa | 4 | 368 | **181 ms** | 160 ms | 167 ms | 727 ms | 2645 ms | 79.9 ms |
