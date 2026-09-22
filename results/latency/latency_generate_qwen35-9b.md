# Readout vs generation on one NVIDIA A40 — Qwen/Qwen3.5-9B

Same records, same prompt (chat template included), one request at a time, medians. The readout is Jevify's `ask`; `prefill` is one forward pass over the prompt with no answer read; `generate N` is Hugging Face greedy decoding of N tokens.

| source | K | prompt tokens | readout | prefill only | generate 1 | generate 8 | generate 32 | per generated token |
|---|---|---|---|---|---|---|---|---|
| boolq | 2 | 224 | **95 ms** | 93 ms | 96 ms | 460 ms | 1709 ms | 52.0 ms |
| arc_challenge | 4 | 148 | **85 ms** | 81 ms | 86 ms | 448 ms | 1690 ms | 51.7 ms |
| clinc150 | 151 | 1649 | **482 ms** | 480 ms | 458 ms | 820 ms | 2065 ms | 51.8 ms |
