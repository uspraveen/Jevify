# Readout vs generation on one NVIDIA A40 — google/gemma-4-12B-it

Same records, same prompt (chat template included), one request at a time, medians. The readout is Jevify's `ask`; `prefill` is one forward pass over the prompt with no answer read; `generate N` is Hugging Face greedy decoding of N tokens.

| source | K | prompt tokens | readout | prefill only | generate 1 | generate 8 | generate 32 | per generated token |
|---|---|---|---|---|---|---|---|---|
| boolq | 2 | 222 | **119 ms** | 109 ms | 124 ms | 859 ms | 3396 ms | 105.6 ms |
| arc_challenge | 4 | 147 | **101 ms** | 88 ms | 105 ms | 687 ms | 2685 ms | 83.2 ms |
| clinc150 | 151 | 1756 | **670 ms** | 675 ms | 650 ms | 1220 ms | 3182 ms | 81.7 ms |
